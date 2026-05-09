"""App-facing /ai/* endpoints.

This router is intentionally **schema-compatible** with the legacy
`majiang/backend` that the uni-app frontend already targets. The existing
mobile client (see `majiang/frontend/utils/api.js`) calls:

    POST /ai/recommend            -> discard recommendation
    POST /ai/recommend-response   -> chi/peng/gang/hu/pass response
    POST /ai/recommend-all        -> score every candidate discard
    POST /ai/reset                -> reset the engine state
    GET  /ai/health               -> AI engine self-check

By matching the legacy request/response JSON field-by-field, swapping the
client's `apiUrl` to this V3 backend is a one-line config change (no app
rebuild required). Internally we route through `LinhaiV3Orchestrator`,
which automatically degrades V3 -> V2 -> heuristic.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..core.state import GameState, Meld, PlayerState, Wind
from ..core.tiles import normalize_code
from ..services.orchestrator import get_orchestrator


# 副露类型白名单（与 v3_service / v2_fallback 消费侧 meld.type 字符串一致）
_MELD_TYPES = {"chi", "peng", "ming_gang", "an_gang", "bu_gang"}
# 兼容前端可能用的别名
_MELD_TYPE_ALIAS: Dict[str, str] = {
    "chii": "chi", "pon": "peng", "kan": "ming_gang",
    "minkan": "ming_gang", "ankan": "an_gang", "kakan": "bu_gang",
    "ming_kan": "ming_gang", "an_kan": "an_gang", "bu_kan": "bu_gang",
    # 中文/拼音别名
    "吾": "peng", "碰": "peng", "吃": "chi",
    "明杠": "ming_gang", "暗杠": "an_gang", "补杠": "bu_gang",
}

router = APIRouter(prefix="/ai", tags=["AI"])


# ---------- tile-code aliasing ------------------------------------------- #
# The legacy `majiang/backend/app/services/linhai_ai_service.py` accepts
# many tile-code dialects (mjai-style `1m/1p/1s`, the in-house `1w/1t`,
# the single-letter honors `E/S/W/N/P/F/C`, and the spelled-out
# `east/south/white/...`). To make the new V3 backend a drop-in
# replacement for the uni-app client, we translate all of those into the
# canonical codes that `backend/app/core/tiles.TILES` expects (`1w..9w`,
# `1t..9t`, lowercase English honors).
_TILE_ALIAS: Dict[str, str] = {}
for _rank in range(1, 10):
    for _legacy in (f"{_rank}m", f"{_rank}w"):
        _TILE_ALIAS[_legacy] = f"{_rank}w"
    for _legacy in (f"{_rank}s", f"{_rank}t"):
        _TILE_ALIAS[_legacy] = f"{_rank}t"
    # Pin tiles don't exist in linhai -- legacy code still sometimes
    # emits them from generic mahjong code paths; fail loudly if we see
    # one rather than silently remap.
    _TILE_ALIAS[f"{_rank}p"] = f"__UNSUPPORTED_PIN_{_rank}__"
_TILE_ALIAS.update(
    {
        "east": "east", "south": "south", "west": "west", "north": "north",
        "white": "white", "green": "green", "red": "red",
        "E": "east", "S": "south", "W": "west", "N": "north",
        "P": "white", "F": "green", "C": "red",
        "1z": "east", "2z": "south", "3z": "west", "4z": "north",
        "5z": "white", "6z": "green", "7z": "red",
    }
)


def _canonical_tile(code: str) -> str:
    """Accept mjai / legacy / native tile codes; return the canonical one
    core.tiles.normalize_code understands. Raises HTTPException 400 with a
    useful message if the tile cannot be represented in linhai (e.g. pin
    suit)."""
    if not isinstance(code, str) or not code.strip():
        raise HTTPException(status_code=400, detail=f"empty or non-string tile: {code!r}")
    raw = code.strip()
    mapped = _TILE_ALIAS.get(raw, raw)
    if mapped.startswith("__UNSUPPORTED_PIN_"):
        raise HTTPException(
            status_code=400,
            detail=f"pin-suit tile {raw!r} is not supported by linhai-majiang",
        )
    try:
        return normalize_code(mapped)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"unknown tile code {raw!r}: {exc}")


# ---------- legacy-payload normalization ------------------------------ #
# The uni-app frontend (`majiang/frontend/pages/index/index.vue`) still
# speaks the old wire format: `wind: "east"` instead of `wind_seat: 0`,
# `action_buttons` instead of `available_actions`, and ships extra fields
# (`melds`, `dora_indicators`, `opponent_discards`, `latest_action_kind`,
# `action_buttons`, `game_id`, `record_data`) that the strict V3 schema
# doesn't know about. Rather than require an app rebuild, we translate
# those at the edge so the existing binary keeps working.
_WIND_STR_TO_SEAT: Dict[str, int] = {
    "east": 0, "south": 1, "west": 2, "north": 3,
    "E": 0, "S": 1, "W": 2, "N": 3,
    "e": 0, "s": 1, "w": 2, "n": 3,
    "0": 0, "1": 1, "2": 2, "3": 3,
    "1z": 0, "2z": 1, "3z": 2, "4z": 3,
}


def _coerce_wind_seat(raw) -> Optional[int]:
    if raw is None:
        return None
    if isinstance(raw, bool):  # bool is a subclass of int; reject it
        return None
    if isinstance(raw, int):
        return raw if 0 <= raw <= 3 else None
    if isinstance(raw, str):
        mapped = _WIND_STR_TO_SEAT.get(raw.strip())
        if mapped is not None:
            return mapped
        try:
            n = int(raw.strip())
            return n if 0 <= n <= 3 else None
        except ValueError:
            return None
    return None


# Some legacy action codes from the app map to V3 engine actions.
# "guo" is the pinyin for 过 (pass); everything else is pass-through.
_ACTION_ALIAS: Dict[str, str] = {
    "guo": "pass",
    "过": "pass",
    "skip": "pass",
    "none": "pass",
}


def _normalize_actions(raw) -> List[str]:
    if raw is None:
        return ["pass"]
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return ["pass"]
    out: List[str] = []
    seen = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        token = item.strip().lower()
        if not token:
            continue
        token = _ACTION_ALIAS.get(token, token)
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out or ["pass"]


def _legacy_translate(values):
    """Pre-validator that rewrites legacy uni-app field names into the V3
    canonical ones. Runs before Pydantic type validation so the request
    schema sees the translated dict."""
    if not isinstance(values, dict):
        return values
    # wind -> wind_seat
    if "wind_seat" not in values or values.get("wind_seat") in (None, ""):
        for alt in ("wind", "seat", "player_wind"):
            if alt in values:
                coerced = _coerce_wind_seat(values[alt])
                if coerced is not None:
                    values["wind_seat"] = coerced
                break
    # Even when wind_seat is provided as a string (e.g. "0", "east"), coerce it.
    if isinstance(values.get("wind_seat"), str):
        coerced = _coerce_wind_seat(values["wind_seat"])
        if coerced is not None:
            values["wind_seat"] = coerced
    # action_buttons -> available_actions
    if "available_actions" not in values and "action_buttons" in values:
        values["available_actions"] = _normalize_actions(values["action_buttons"])
    # `opponent_discards` and `discards` are kept and exposed as first-class
    # fields so the engine can use the opponent's river (risk / defense
    # features benefit from this). `melds` / `opponent_melds` are now also
    # first-class — keep them so副露信息能进 V3 search。Other legacy fields
    # are harmless metadata and can be quietly dropped.
    # opp_melds -> opponent_melds 别名
    if "opponent_melds" not in values and "opp_melds" in values:
        values["opponent_melds"] = values.get("opp_melds")
    for legacy_key in (
        "dora_indicators",
        "latest_action_kind", "action_buttons", "game_id", "record_data",
        "player_wind", "seat", "wind", "opp_melds",
    ):
        values.pop(legacy_key, None)
    return values


# ---------- request / response schemas (kept byte-compatible) -------------- #


class OpponentState(BaseModel):
    wind_seat: int = Field(..., ge=0, le=3, description="风位 (0=东, 1=南, 2=西, 3=北)")
    reach: bool = Field(default=False, description="是否立直")
    melds_count: int = Field(default=0, ge=0, le=4, description="副露数量")


class ChiPonRecord(BaseModel):
    type: str = Field(..., description="类型: chi 或 pon")
    target_player: int = Field(..., ge=0, le=3, description="目标玩家")


class MeldInput(BaseModel):
    """副露输入。与 core.state.Meld(type, tiles) 一致，额外允许一个可选 from_player 提示对手座位。"""

    model_config = ConfigDict(extra="ignore")

    type: str = Field(..., description="chi/peng/ming_gang/an_gang/bu_gang")
    tiles: List[str] = Field(..., min_length=1, max_length=4, description="副露牌列表")
    from_player: Optional[int] = Field(default=None, ge=0, le=3, description="被吾玩家座位（可选）")

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy(cls, values):
        if not isinstance(values, dict):
            return values
        # 兼容字段名：kind -> type；consumed/pai -> tiles
        if "type" not in values and "kind" in values:
            values["type"] = values["kind"]
        if "tiles" not in values:
            for alt in ("consumed", "pai", "hai"):
                if alt in values:
                    values["tiles"] = values[alt]
                    break
        # 类型别名归一
        raw_type = values.get("type")
        if isinstance(raw_type, str):
            t = raw_type.strip().lower()
            values["type"] = _MELD_TYPE_ALIAS.get(t, t)
        return values


class AIRecommendRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hand: List[str] = Field(..., min_length=13, max_length=14, description="手牌列表")
    wind_seat: int = Field(..., ge=0, le=3, description="风位")
    has_kan: bool = Field(default=False, description="是否有杠")
    wall_remaining: int = Field(default=70, ge=0, le=136, description="剩余牌墙数量")
    round_number: int = Field(default=0, ge=0, description="局数")
    opponents_state: Optional[List[OpponentState]] = Field(default=None)
    chi_pon_history: Optional[List[ChiPonRecord]] = Field(default=None)
    strategy: str = Field(default="balanced", description="balanced/aggressive/defensive")
    discards: Optional[List[str]] = Field(default=None, description="自家牌河")
    opponent_discards: Optional[List[str]] = Field(default=None, description="对手牌河")
    melds: Optional[List[MeldInput]] = Field(default=None, description="自家副露")
    opponent_melds: Optional[List[MeldInput]] = Field(default=None, description="对手副露")
    mode: Optional[str] = Field(default=None, description="玩法模式，如 'linhai_2p'")
    missing_suit_self: Optional[str] = Field(default=None, description="自家缺一门: w/t/z 或别名")
    missing_suit_opp: Optional[str] = Field(default=None, description="对手缺一门")

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_payload(cls, values):
        return _legacy_translate(values)


class AIRecommendResponse(BaseModel):
    success: bool
    recommendation: Dict
    message: Optional[str] = None


class AIAllRecommendationsRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hand: List[str] = Field(..., min_length=13, max_length=14)
    wind_seat: int = Field(..., ge=0, le=3)
    has_kan: bool = Field(default=False)
    wall_remaining: int = Field(default=70, ge=0, le=136)
    round_number: int = Field(default=0, ge=0)
    discards: Optional[List[str]] = Field(default=None)
    opponent_discards: Optional[List[str]] = Field(default=None)
    melds: Optional[List[MeldInput]] = Field(default=None)
    opponent_melds: Optional[List[MeldInput]] = Field(default=None)
    mode: Optional[str] = Field(default=None)
    missing_suit_self: Optional[str] = Field(default=None)
    missing_suit_opp: Optional[str] = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_payload(cls, values):
        return _legacy_translate(values)


class AIResponseRequest(BaseModel):
    """Request for /ai/recommend-response (NEW in V3).

    The frontend already calls this endpoint via api.js
    `getResponseRecommendation`, but the legacy backend never exposed it.
    Accepts both new-style (`wind_seat`, `available_actions`) and
    legacy uni-app-style (`wind`, `action_buttons`) payloads.
    """

    model_config = ConfigDict(extra="ignore")

    hand: List[str] = Field(..., min_length=0, max_length=14, description="手牌")
    wind_seat: int = Field(..., ge=0, le=3)
    discarded_tile: str = Field(..., description="最后被打出的牌")
    from_player: int = Field(..., ge=0, le=3, description="打牌人的风位")
    available_actions: List[str] = Field(
        default_factory=lambda: ["pass"],
        description="允许的响应动作列表: pass/peng/chi/gang/hu",
    )
    wall_remaining: int = Field(default=70, ge=0, le=136)
    round_number: int = Field(default=0, ge=0)
    opponents_state: Optional[List[OpponentState]] = Field(default=None)
    passed_hu_this_round: bool = Field(default=False, description="本局已经过胡")
    discards: Optional[List[str]] = Field(default=None)
    opponent_discards: Optional[List[str]] = Field(default=None)
    melds: Optional[List[MeldInput]] = Field(default=None)
    opponent_melds: Optional[List[MeldInput]] = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_payload(cls, values):
        return _legacy_translate(values)


# ---------- helpers: V3 result -> legacy shape -------------------------- #


_WIND_ORDER = [Wind.EAST, Wind.SOUTH, Wind.WEST, Wind.NORTH]


def _wind_of(seat: int) -> Wind:
    return _WIND_ORDER[max(0, min(seat, 3))]


def _confidence_from_shanten(shanten: Optional[int]) -> float:
    """Match the legacy LinHaiAIService._calculate_confidence mapping so
    confidence scores are comparable across backends."""
    if shanten is None:
        return 0.5
    if shanten <= -1:
        return 1.0
    if shanten == 0:
        return 0.9
    if shanten == 1:
        return 0.8
    return 0.7


def _meld_input_to_meld(m: "MeldInput") -> Optional[Meld]:
    """将路由层的 MeldInput 转成 core.state.Meld，不合法的跳过。"""
    raw_type = (m.type or "").strip().lower()
    meld_type = _MELD_TYPE_ALIAS.get(raw_type, raw_type)
    if meld_type not in _MELD_TYPES:
        return None
    try:
        canonical_tiles = [_canonical_tile(t) for t in m.tiles]
    except HTTPException:
        return None
    if not canonical_tiles:
        return None
    return Meld(tiles=canonical_tiles, type=meld_type)


def _normalize_meld_list(raw: Optional[List["MeldInput"]]) -> List[Meld]:
    if not raw:
        return []
    out: List[Meld] = []
    for item in raw:
        meld = _meld_input_to_meld(item)
        if meld is not None:
            out.append(meld)
    return out


def _build_state(
    hand: List[str],
    wind_seat: int,
    *,
    wall_remaining: int = 70,
    opponents_state: Optional[List[OpponentState]] = None,
    passed_hu_this_round: bool = False,
    opponent_discards_per_seat: Optional[Dict[int, List[str]]] = None,
    own_melds: Optional[List[Meld]] = None,
    opponent_melds_per_seat: Optional[Dict[int, List[Meld]]] = None,
) -> GameState:
    my_wind = _wind_of(wind_seat)
    my_hand = [_canonical_tile(tile) for tile in hand]

    players: Dict[Wind, PlayerState] = {
        my_wind: PlayerState(
            wind=my_wind,
            hand=my_hand,
            melds=list(own_melds or []),
            discards=[_canonical_tile(t) for t in (opponent_discards_per_seat or {}).get(wind_seat, [])],
        )
    }
    for seat_idx, wind in enumerate(_WIND_ORDER):
        if wind is my_wind:
            continue
        discards: List[str] = []
        if opponent_discards_per_seat:
            raw = opponent_discards_per_seat.get(seat_idx)
            if raw:
                discards = [_canonical_tile(t) for t in raw]
        melds: List[Meld] = []
        if opponent_melds_per_seat:
            melds = list(opponent_melds_per_seat.get(seat_idx) or [])
        players[wind] = PlayerState(wind=wind, hand=[], melds=melds, discards=discards)

    state = GameState(
        round_wind=Wind.EAST,
        dealer_wind=Wind.EAST,
        active_wind=my_wind,
        players=players,
        wall_remaining=int(wall_remaining),
    )
    if passed_hu_this_round:
        # Exposed on both PlayerState and GameState for legacy compatibility.
        me = state.current_player()
        setattr(me, "passed_hu_this_round", True)
        setattr(state, "passed_hu_this_round", True)
    return state


def _to_app_tile(tile_raw) -> str:
    """Translate a tile emitted by the V3 engine (`E`/`W`/`S`/`N`/`P`/`F`/`C`
    for honors, `1m`/`1s` variants, ...) into the canonical codes that
    `majiang/frontend` understands (`east/south/west/north/white/green/red`,
    `1w..9w`, `1t..9t`). Unknown tokens pass through unchanged so a bad
    mapping never takes the whole response down."""
    if not tile_raw:
        return ""
    raw = str(tile_raw).strip()
    if not raw:
        return ""
    try:
        return _canonical_tile(raw)
    except HTTPException:
        return raw


def _map_candidate_scores_for_app(candidate_scores) -> Dict[str, float]:
    """`result.vue` calls `score.toFixed(2)` on every value in
    `meta.scores`, so values must be numeric. Keys are translated the
    same way tiles are so 'E' doesn't show up in the UI."""
    out: Dict[str, float] = {}
    if not isinstance(candidate_scores, dict):
        return out
    for raw_key, raw_val in candidate_scores.items():
        if not isinstance(raw_val, (int, float)):
            continue
        key = str(raw_key)
        # "pass->9t" / "pass" / tile-only candidates: only try to
        # translate the tile portion when it's a single tile code.
        if "->" in key:
            head, _, tail = key.partition("->")
            tail_app = _to_app_tile(tail) if tail else tail
            key_out = f"{head}->{tail_app}" if tail_app else head
        else:
            key_out = _to_app_tile(key) or key
        out[key_out] = round(float(raw_val), 3)
    return out


def _discard_explanation(tile: str, agari_prob: float, houjuu_prob: float, chosen_by: str) -> str:
    bits = [chosen_by] if chosen_by else []
    if tile:
        bits.append(f"打{tile}")
    if agari_prob > 0:
        bits.append(f"和牌概率 {agari_prob:.1%}")
    if houjuu_prob > 0.01:
        bits.append(f"放炮概率 {houjuu_prob:.1%}")
    return ", ".join(bit for bit in bits if bit) or "v3 recommendation"


def _response_explanation(action: str, tile: str, agari_prob: float, houjuu_prob: float) -> str:
    bits = [f"action={action}"]
    if tile:
        bits.append(f"tile={tile}")
    if agari_prob > 0:
        bits.append(f"agari={agari_prob:.1%}")
    if houjuu_prob > 0.01:
        bits.append(f"houjuu={houjuu_prob:.1%}")
    return ", ".join(bits)


def _shape_discard(result: Dict) -> Dict:
    """Translate orchestrator's SearchResult into the flat shape the
    uni-app frontend expects. The app assigns this dict directly to
    `lastRecommendation`, so fields like `confidence`, `action`,
    `explanation`, and `meta.scores` must exist at the top level."""
    tile = _to_app_tile(result.get("tile"))
    total_ev = float(result.get("total_ev", 0.0))
    shanten = result.get("shanten")
    if shanten is None:
        shanten = int(result.get("search_depth", 0) or 0)
    agari_prob = float(result.get("agari_prob", 0.0) or 0.0)
    houjuu_prob = float(result.get("houjuu_prob", 0.0) or 0.0)
    defense_score = float(result.get("defense_score", 0.0) or 0.0)
    candidates = result.get("candidate_scores") or {}

    return {
        # Fields the uni-app reads directly from `lastRecommendation`.
        "success": True,
        "action": "discard",
        "tile": tile,
        "confidence": _confidence_from_shanten(int(shanten) if shanten is not None else None),
        "explanation": _discard_explanation(tile, agari_prob, houjuu_prob, result.get("chosen_by", "")),
        "ai_type": "linhai-v3",
        # Extra V3-native fields kept for debugging / richer UI.
        "score": round(total_ev, 3),
        "shanten": int(shanten),
        "agari_prob": round(agari_prob, 6),
        "houjuu_prob": round(houjuu_prob, 6),
        "defense_score": round(defense_score, 3),
        "bonus_value": round(agari_prob * 1000.0, 3),
        "risk_value": round(houjuu_prob * 1000.0, 3),
        "engine": result.get("engine", "v3"),
        "fallback_reason": result.get("fallback_reason", ""),
        "meta": {
            # `result.vue` renders this block; keys must be numeric scores.
            "scores": _map_candidate_scores_for_app(candidates),
            "engine": result.get("engine", "v3"),
            "shanten": int(shanten),
            "agari_prob": round(agari_prob, 6),
            "houjuu_prob": round(houjuu_prob, 6),
            "defense_score": round(defense_score, 3),
            "fallback_reason": result.get("fallback_reason", ""),
        },
    }


def _shape_response(result: Dict, available_actions: List[str]) -> Dict:
    """Translate the orchestrator's response-action result into the
    `{actions: [...], recommended: ..., meta: {...}}` shape expected by
    `normalizeResponseResult` on the frontend. We must list one entry
    per action the player can pick so the app can highlight the one
    matching `recommended`."""
    recommended = (result.get("action") or "pass").strip().lower() or "pass"
    tile = _to_app_tile(result.get("tile"))
    shanten = result.get("shanten", 0)
    if shanten is None:
        shanten = 0
    agari_prob = float(result.get("agari_prob", 0.0) or 0.0)
    houjuu_prob = float(result.get("houjuu_prob", 0.0) or 0.0)
    defense_score = float(result.get("defense_score", 0.0) or 0.0)
    total_ev = float(result.get("total_ev", 0.0) or 0.0)
    candidate_scores = result.get("candidate_scores") or {}

    # Build one `{action, tile, confidence, explanation}` entry per
    # action that was originally offered. The uni-app's
    # `normalizeResponseResult` looks up the entry whose `action`
    # matches `recommended`.
    actions_out: List[Dict] = []
    seen = set()
    ordered = list(available_actions) or ["pass"]
    if recommended not in ordered:
        ordered.append(recommended)
    for act_raw in ordered:
        act = (act_raw or "").strip().lower()
        if not act or act in seen:
            continue
        seen.add(act)
        is_chosen = act == recommended
        entry_tile = tile if is_chosen and tile else ""
        entry_conf = _confidence_from_shanten(int(shanten)) if is_chosen else 0.3
        entry_expl = (
            _response_explanation(act, entry_tile, agari_prob, houjuu_prob)
            if is_chosen
            else f"action={act}"
        )
        actions_out.append(
            {
                "action": act,
                "tile": entry_tile,
                "confidence": entry_conf,
                "explanation": entry_expl,
            }
        )

    return {
        "success": True,
        "recommended": recommended,
        "actions": actions_out,
        "ai_type": "linhai-v3",
        # Keep these at the top too so code paths that skip
        # normalizeResponseResult still render cleanly.
        "action": recommended,
        "tile": tile,
        "confidence": _confidence_from_shanten(int(shanten)),
        "explanation": _response_explanation(recommended, tile, agari_prob, houjuu_prob),
        "score": round(total_ev, 3),
        "shanten": int(shanten),
        "agari_prob": round(agari_prob, 6),
        "houjuu_prob": round(houjuu_prob, 6),
        "defense_score": round(defense_score, 3),
        "engine": result.get("engine", "v3"),
        "fallback_reason": result.get("fallback_reason", ""),
        "meta": {
            "scores": _map_candidate_scores_for_app(candidate_scores),
            "engine": result.get("engine", "v3"),
            "agari_prob": round(agari_prob, 6),
            "houjuu_prob": round(houjuu_prob, 6),
            "defense_score": round(defense_score, 3),
            "fallback_reason": result.get("fallback_reason", ""),
        },
    }


# ---------- routes ----------------------------------------------------- #


def _opponent_discards_map(
    wind_seat: int,
    own_discards: Optional[List[str]],
    opp_discards: Optional[List[str]],
    opp_seat_hint: Optional[int] = None,
) -> Dict[int, List[str]]:
    """Build the `opponent_discards_per_seat` dict from the legacy flat
    lists. Opponent discards go to the first non-my seat, or to
    `opp_seat_hint` when provided (e.g. `from_player` in response calls)."""
    mapping: Dict[int, List[str]] = {}
    if own_discards:
        mapping[int(wind_seat)] = list(own_discards)
    if opp_discards:
        if opp_seat_hint is not None and int(opp_seat_hint) != int(wind_seat):
            target_seat = int(opp_seat_hint)
        else:
            target_seat = next((s for s in range(4) if s != int(wind_seat)), 1)
        mapping.setdefault(target_seat, []).extend(opp_discards)
    return mapping


def _opponent_melds_map(
    wind_seat: int,
    opp_melds_input: Optional[List["MeldInput"]],
    opp_seat_hint: Optional[int] = None,
) -> Dict[int, List[Meld]]:
    """将平哈列表的对手副露分配到座位：优先用 MeldInput.from_player，
    其次用 opp_seat_hint，最后退化到第一个非本座位。"""
    if not opp_melds_input:
        return {}
    fallback_seat = next((s for s in range(4) if s != int(wind_seat)), 1)
    if opp_seat_hint is not None and int(opp_seat_hint) != int(wind_seat):
        fallback_seat = int(opp_seat_hint)
    mapping: Dict[int, List[Meld]] = {}
    for raw in opp_melds_input:
        meld = _meld_input_to_meld(raw)
        if meld is None:
            continue
        seat = (
            int(raw.from_player)
            if raw.from_player is not None and 0 <= int(raw.from_player) <= 3 and int(raw.from_player) != int(wind_seat)
            else fallback_seat
        )
        mapping.setdefault(seat, []).append(meld)
    return mapping


@router.post("/recommend")
def recommend_discard(request: AIRecommendRequest) -> Dict:
    """Best discard recommendation.

    Returns the flat `{action, tile, confidence, explanation, meta: {...}}`
    shape that the uni-app frontend uses directly (no `recommendation`
    wrapper). See `majiang/frontend/pages/index/index.vue` ->
    `requestRecommendation`, where `result = await
    getDiscardRecommendation(...)` is assigned straight to
    `lastRecommendation` and consumed as-is by the templates.
    """
    try:
        state = _build_state(
            request.hand,
            request.wind_seat,
            wall_remaining=request.wall_remaining,
            opponents_state=request.opponents_state,
            opponent_discards_per_seat=_opponent_discards_map(
                request.wind_seat, request.discards, request.opponent_discards
            ),
            own_melds=_normalize_meld_list(request.melds),
            opponent_melds_per_seat=_opponent_melds_map(
                request.wind_seat, request.opponent_melds
            ),
        )
        result = get_orchestrator().recommend(
            state,
            mode=request.mode,
            missing_suit_self=request.missing_suit_self,
            missing_suit_opp=request.missing_suit_opp,
        )
        return _shape_discard(result)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"AI推荐失败: {exc}")


@router.post("/recommend-response")
def recommend_response(request: AIResponseRequest) -> Dict:
    """Response-action recommendation (chi / peng / gang / hu / pass).

    This endpoint is NEW in V3. The legacy backend didn't expose it even
    though the frontend already calls it via `getResponseRecommendation`.
    """
    try:
        # Merge opponent-discards list (if the app sent one) with the freshly
        # discarded tile, so the engine sees the full river when computing
        # defense / risk features.
        opp_river = list(request.opponent_discards or [])
        opp_river.append(request.discarded_tile)
        discard_map = _opponent_discards_map(
            request.wind_seat,
            request.discards,
            opp_river,
            opp_seat_hint=request.from_player,
        )
        state = _build_state(
            request.hand or [],
            request.wind_seat,
            wall_remaining=request.wall_remaining,
            opponents_state=request.opponents_state,
            passed_hu_this_round=request.passed_hu_this_round,
            opponent_discards_per_seat=discard_map,
            own_melds=_normalize_meld_list(request.melds),
            opponent_melds_per_seat=_opponent_melds_map(
                request.wind_seat, request.opponent_melds, opp_seat_hint=request.from_player
            ),
        )
        discarded = _canonical_tile(request.discarded_tile)
        actions = [action.strip().lower() for action in request.available_actions if action.strip()]
        if not actions:
            actions = ["pass"]
        result = get_orchestrator().recommend_response(
            state, discarded, int(request.from_player), actions
        )
        return _shape_response(result, actions)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"AI响应推荐失败: {exc}")


@router.post("/recommend-all")
def recommend_all(request: AIAllRecommendationsRequest) -> Dict:
    """Return every candidate discard's score.

    Order preserved from orchestrator.candidate_scores (best first).
    """
    try:
        state = _build_state(
            request.hand,
            request.wind_seat,
            wall_remaining=request.wall_remaining,
            opponent_discards_per_seat=_opponent_discards_map(
                request.wind_seat, request.discards, request.opponent_discards
            ),
            own_melds=_normalize_meld_list(request.melds),
            opponent_melds_per_seat=_opponent_melds_map(
                request.wind_seat, request.opponent_melds
            ),
        )
        result = get_orchestrator().recommend(
            state,
            mode=request.mode,
            missing_suit_self=request.missing_suit_self,
            missing_suit_opp=request.missing_suit_opp,
        )
        ranked = []
        for tile, score in (result.get("candidate_scores") or {}).items():
            if not isinstance(score, (int, float)):
                continue
            ranked.append(
                {
                    "tile": _to_app_tile(tile),
                    "score": round(float(score), 3),
                    "confidence": _confidence_from_shanten(result.get("shanten")),
                }
            )
        return {"success": True, "recommendations": ranked, "engine": result.get("engine", "v3")}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"获取推荐失败: {exc}")


@router.post("/reset")
def reset_ai() -> Dict:
    """Kept for legacy API compatibility. V3 engine is stateless between
    calls (cache is bounded and self-invalidates), so this is effectively
    a no-op but we still return success so the client's flow is happy.
    """
    return {"success": True, "message": "v3 engine is stateless; reset acknowledged"}


@router.get("/health")
def ai_health() -> Dict:
    """AI engine self-check. Mirrors the legacy /ai/health route."""
    try:
        orch = get_orchestrator()
        status = orch.engine_status()
        test_hand = ["1w", "1w", "1w", "2w", "2w", "2w", "3w", "3w", "3w", "4w", "5w", "6w", "7w", "8w"]
        state = _build_state(test_hand, 0, wall_remaining=70)
        result = orch.recommend(state)
        return {
            "success": True,
            "status": "healthy",
            "message": "AI引擎运行正常",
            "test_result": {"tile": result.get("tile"), "shanten": result.get("shanten", 0)},
            "engine_status": status,
        }
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "status": "unhealthy", "message": f"AI引擎异常: {exc}"}
