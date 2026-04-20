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
from pydantic import BaseModel, Field

from ..core.state import GameState, Meld, PlayerState, Wind
from ..core.tiles import normalize_code
from ..services.orchestrator import get_orchestrator

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


# ---------- request / response schemas (kept byte-compatible) -------------- #


class OpponentState(BaseModel):
    wind_seat: int = Field(..., ge=0, le=3, description="\u98ce\u4f4d (0=\u4e1c, 1=\u5357, 2=\u897f, 3=\u5317)")
    reach: bool = Field(default=False, description="\u662f\u5426\u7acb\u76f4")
    melds_count: int = Field(default=0, ge=0, le=4, description="\u526f\u9732\u6570\u91cf")


class ChiPonRecord(BaseModel):
    type: str = Field(..., description="\u7c7b\u578b: chi \u6216 pon")
    target_player: int = Field(..., ge=0, le=3, description="\u76ee\u6807\u73a9\u5bb6")


class AIRecommendRequest(BaseModel):
    hand: List[str] = Field(..., min_length=13, max_length=14, description="\u624b\u724c\u5217\u8868")
    wind_seat: int = Field(..., ge=0, le=3, description="\u98ce\u4f4d")
    has_kan: bool = Field(default=False, description="\u662f\u5426\u6709\u6760")
    wall_remaining: int = Field(default=70, ge=0, le=136, description="\u5269\u4f59\u724c\u5899\u6570\u91cf")
    round_number: int = Field(default=0, ge=0, description="\u5c40\u6570")
    opponents_state: Optional[List[OpponentState]] = Field(default=None)
    chi_pon_history: Optional[List[ChiPonRecord]] = Field(default=None)
    strategy: str = Field(default="balanced", description="balanced/aggressive/defensive")


class AIRecommendResponse(BaseModel):
    success: bool
    recommendation: Dict
    message: Optional[str] = None


class AIAllRecommendationsRequest(BaseModel):
    hand: List[str] = Field(..., min_length=13, max_length=14)
    wind_seat: int = Field(..., ge=0, le=3)
    has_kan: bool = Field(default=False)
    wall_remaining: int = Field(default=70, ge=0, le=136)
    round_number: int = Field(default=0, ge=0)


class AIResponseRequest(BaseModel):
    """Request for /ai/recommend-response (NEW in V3).

    The frontend already calls this endpoint via api.js
    `getResponseRecommendation`, but the legacy backend never exposed it.
    """

    hand: List[str] = Field(..., min_length=0, max_length=14, description="\u624b\u724c")
    wind_seat: int = Field(..., ge=0, le=3)
    discarded_tile: str = Field(..., description="\u6700\u540e\u88ab\u6253\u51fa\u7684\u724c")
    from_player: int = Field(..., ge=0, le=3, description="\u6253\u724c\u4eba\u7684\u98ce\u4f4d")
    available_actions: List[str] = Field(
        default_factory=lambda: ["pass"],
        description="\u5141\u8bb8\u7684\u54cd\u5e94\u52a8\u4f5c\u5217\u8868: pass/peng/chi/gang/hu",
    )
    wall_remaining: int = Field(default=70, ge=0, le=136)
    round_number: int = Field(default=0, ge=0)
    opponents_state: Optional[List[OpponentState]] = Field(default=None)
    passed_hu_this_round: bool = Field(default=False, description="\u672c\u5c40\u5df2\u7ecf\u8fc7\u80e1")


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


def _build_state(
    hand: List[str],
    wind_seat: int,
    *,
    wall_remaining: int = 70,
    opponents_state: Optional[List[OpponentState]] = None,
    passed_hu_this_round: bool = False,
    opponent_discards_per_seat: Optional[Dict[int, List[str]]] = None,
) -> GameState:
    my_wind = _wind_of(wind_seat)
    my_hand = [_canonical_tile(tile) for tile in hand]

    players: Dict[Wind, PlayerState] = {
        my_wind: PlayerState(
            wind=my_wind,
            hand=my_hand,
            discards=[_canonical_tile(t) for t in (opponent_discards_per_seat or {}).get(wind_seat, [])],
        )
    }
    for seat_idx, wind in enumerate(_WIND_ORDER):
        if wind is my_wind:
            continue
        discards = []
        if opponent_discards_per_seat:
            raw = opponent_discards_per_seat.get(seat_idx)
            if raw:
                discards = [_canonical_tile(t) for t in raw]
        players[wind] = PlayerState(wind=wind, hand=[], discards=discards)

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


def _shape_discard(result: Dict) -> Dict:
    """Translate orchestrator's rich SearchResult into the legacy
    `recommendation` dict shape that the app expects."""
    tile = result.get("tile") or ""
    # total_ev can be large; keep it as-is but expose a normalized "score"
    # similar to the legacy V2 field.
    total_ev = float(result.get("total_ev", 0.0))
    shanten = result.get("shanten")
    if shanten is None:
        # Older V2/heuristic responses don't include shanten; derive a
        # rough value from search_depth so the UI still has something.
        shanten = int(result.get("search_depth", 0) or 0)
    agari_prob = float(result.get("agari_prob", 0.0) or 0.0)
    houjuu_prob = float(result.get("houjuu_prob", 0.0) or 0.0)
    defense_score = float(result.get("defense_score", 0.0) or 0.0)

    candidates = result.get("candidate_scores") or {}
    explanation_bits = [result.get("chosen_by", "")]
    if tile:
        explanation_bits.append(f"\u6253{tile}")
    if agari_prob > 0:
        explanation_bits.append(f"\u548c\u724c\u6982\u7387 {agari_prob:.1%}")
    if houjuu_prob > 0.01:
        explanation_bits.append(f"\u653e\u70ae\u6982\u7387 {houjuu_prob:.1%}")
    explanation = ", ".join(bit for bit in explanation_bits if bit)

    return {
        "tile": tile,
        "score": round(total_ev, 3),
        "shanten": int(shanten),
        "bonus_value": round(agari_prob * 1000.0, 3),
        "risk_value": round(houjuu_prob * 1000.0, 3),
        "defense_score": round(defense_score, 3),
        "agari_prob": round(agari_prob, 6),
        "houjuu_prob": round(houjuu_prob, 6),
        "explanation": explanation or "v3 recommendation",
        "confidence": _confidence_from_shanten(int(shanten) if shanten is not None else None),
        "engine": result.get("engine", "v3"),
        "candidate_scores": candidates,
        "fallback_reason": result.get("fallback_reason", ""),
    }


def _shape_response(result: Dict) -> Dict:
    action = result.get("action") or "pass"
    tile = result.get("tile") or ""
    shanten = result.get("shanten", 0)
    if shanten is None:
        shanten = 0
    agari_prob = float(result.get("agari_prob", 0.0) or 0.0)
    houjuu_prob = float(result.get("houjuu_prob", 0.0) or 0.0)
    defense_score = float(result.get("defense_score", 0.0) or 0.0)
    total_ev = float(result.get("total_ev", 0.0) or 0.0)

    # Keep a text explanation short; UI can always display candidate_scores
    # for full detail.
    bits = [f"action={action}"]
    if tile:
        bits.append(f"tile={tile}")
    if agari_prob > 0:
        bits.append(f"agari={agari_prob:.1%}")
    if houjuu_prob > 0.01:
        bits.append(f"houjuu={houjuu_prob:.1%}")
    explanation = ", ".join(bits)

    return {
        "action": action,
        "tile": tile,
        "score": round(total_ev, 3),
        "shanten": int(shanten),
        "bonus_value": round(agari_prob * 1000.0, 3),
        "risk_value": round(houjuu_prob * 1000.0, 3),
        "defense_score": round(defense_score, 3),
        "agari_prob": round(agari_prob, 6),
        "houjuu_prob": round(houjuu_prob, 6),
        "explanation": explanation,
        "confidence": _confidence_from_shanten(int(shanten)),
        "engine": result.get("engine", "v3"),
        "candidate_scores": result.get("candidate_scores", {}),
        "fallback_reason": result.get("fallback_reason", ""),
    }


# ---------- routes ----------------------------------------------------- #


@router.post("/recommend", response_model=AIRecommendResponse)
def recommend_discard(request: AIRecommendRequest) -> AIRecommendResponse:
    """Best discard recommendation.

    Byte-compatible with legacy `majiang/backend` /ai/recommend: same
    request schema, same top-level `{success, recommendation, message}`
    envelope. The `recommendation` dict is a superset of the legacy
    fields (we add `agari_prob`, `houjuu_prob`, `defense_score`,
    `candidate_scores`, `engine`, `fallback_reason`) so the existing UI
    keeps working while new UI can surface richer info.
    """
    try:
        state = _build_state(
            request.hand,
            request.wind_seat,
            wall_remaining=request.wall_remaining,
            opponents_state=request.opponents_state,
        )
        result = get_orchestrator().recommend(state)
        return AIRecommendResponse(success=True, recommendation=_shape_discard(result))
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"AI\u63a8\u8350\u5931\u8d25: {exc}")


@router.post("/recommend-response")
def recommend_response(request: AIResponseRequest) -> Dict:
    """Response-action recommendation (chi / peng / gang / hu / pass).

    This endpoint is NEW in V3. The legacy backend didn't expose it even
    though the frontend already calls it via `getResponseRecommendation`.
    """
    try:
        state = _build_state(
            request.hand or [],
            request.wind_seat,
            wall_remaining=request.wall_remaining,
            opponents_state=request.opponents_state,
            passed_hu_this_round=request.passed_hu_this_round,
            opponent_discards_per_seat={request.from_player: [request.discarded_tile]},
        )
        discarded = _canonical_tile(request.discarded_tile)
        actions = [action.strip().lower() for action in request.available_actions if action.strip()]
        if not actions:
            actions = ["pass"]
        result = get_orchestrator().recommend_response(
            state, discarded, int(request.from_player), actions
        )
        return {"success": True, "recommendation": _shape_response(result)}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"AI\u54cd\u5e94\u63a8\u8350\u5931\u8d25: {exc}")


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
        )
        result = get_orchestrator().recommend(state)
        ranked = []
        for tile, score in (result.get("candidate_scores") or {}).items():
            ranked.append(
                {
                    "tile": tile,
                    "score": round(float(score), 3),
                    "confidence": _confidence_from_shanten(result.get("shanten")),
                }
            )
        return {"success": True, "recommendations": ranked, "engine": result.get("engine", "v3")}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"\u83b7\u53d6\u63a8\u8350\u5931\u8d25: {exc}")


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
            "message": "AI\u5f15\u64ce\u8fd0\u884c\u6b63\u5e38",
            "test_result": {"tile": result.get("tile"), "shanten": result.get("shanten", 0)},
            "engine_status": status,
        }
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "status": "unhealthy", "message": f"AI\u5f15\u64ce\u5f02\u5e38: {exc}"}
