"""ReferenceHumanPolicy — 离线代理对手，模拟"中等熟练真人"水平。

按 spec §2.2：
- 基础动作：V2 baseline（EV 搜索，非简单启发式）
- 防守层：houjuu_prob > threshold 时按概率 betaori
- 番数倾向：清一色/混一色 雏形（≥6 张同色）时偏好成型路线
- 抓冲偏好：抓冲红头存在时优先满足门风牌
- 错牌噪声：3% 概率从 top-2 随机选

强度校准目标（`tools/calibrate_reference.py`）：
- vs heuristic 胜率 ≥ 70%
- 放炮率 ≤ 12%

校准后的超参写入 `engine/params/v3/reference_human_config.json`，
本文件 __init__ 会自动加载（如果存在）。
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from ..core.state import GameState
from ..core.tiles import Suit, normalize_code, require_tile
from .strategy_fallback import StrategyFallbackService
from .v2_fallback import V2FallbackAdapter

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "engine" / "params" / "v3" / "reference_human_config.json"


# ---------- 默认超参（spec §2.2） ----------

DEFAULT_PARAMS = {
    "betaori_threshold": 0.20,    # houjuu_prob > 这个值才考虑 betaori
    "betaori_prob": 0.50,         # 触发后 50% 概率执行
    "hunyise_bonus": 0.25,        # 清一色/混一色 雏形时 EV 加成
    "hunyise_min_same_suit": 6,   # 至少 6 张同色才认定为雏形
    "mistake_prob": 0.03,         # 3% 概率从 top-2 随机选
    "redhead_preference_weight": 0.5,  # 抓冲红头牌的偏好权重
}


def _load_calibrated_params(path: Optional[Path] = None) -> Dict[str, float]:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    params = dict(DEFAULT_PARAMS)
    if p.exists():
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
            override = cfg.get("params", {})
            for k, v in override.items():
                if k in params:
                    params[k] = v
        except (json.JSONDecodeError, OSError):
            pass
    return params


# ---------- 牌色判断（用于番数倾向） ----------

def _suit_of(tile: str) -> str:
    """Return short-suit token: 'm' = 万, 's' = 条, 'p' = 筒, 'z' = 字牌.

    Uses the canonical TileDescriptor (not string-suffix matching) because
    "west", "east" etc. would falsely match endswith("t") / endswith("st").
    """
    desc = require_tile(normalize_code(tile))
    if desc.suit is Suit.CHARACTERS:
        return "m"
    if desc.suit is Suit.BAMBOO:
        return "s"
    # CHARACTERS / BAMBOO covered; HONOR → "z"
    return "z"


def _normalize_missing_suit(s: Optional[str]) -> Optional[str]:
    """缺一门别名归一：w/m/万/wan/characters → 'm'； t/s/条/索/bamboo → 's'.
    None / 未知 → None（不启用）。"""
    if not s:
        return None
    v = s.strip().lower()
    if v in ("m", "w", "wan", "char", "characters", "万"):
        return "m"
    if v in ("s", "t", "tiao", "bamboo", "sozu", "条", "索"):
        return "s"
    return None


def _detect_hunyise_target(hand: List[str]) -> Optional[str]:
    """Return suit prefix ('m'|'p'|'s') if hand has ≥ N same-suit tiles, else None."""
    suit_counts: Counter = Counter()
    for tile in hand:
        s = _suit_of(tile)
        if s != "z":
            suit_counts[s] += 1
    if not suit_counts:
        return None
    best_suit, best_count = suit_counts.most_common(1)[0]
    return best_suit


# ---------- 主类 ----------

class ReferenceHumanPolicy:
    """Acts as opponent for offline acceptance test.

    Behavior contract:
    - recommend_discard(state) -> dict (same shape as V2/Strategy fallbacks return)
    - recommend_response(state, discarded_tile, from_player, action_buttons) -> dict
    """

    def __init__(self,
                 seed: int = 0,
                 config_path: Optional[Path] = None,
                 **overrides):
        self.rng = random.Random(seed)
        self.params = _load_calibrated_params(config_path)
        # CLI / test overrides
        for k, v in overrides.items():
            if k in self.params:
                self.params[k] = v

        self.v2 = V2FallbackAdapter()
        self.strategy = StrategyFallbackService()

    # --- introspection ---

    def status(self) -> Dict[str, object]:
        return {
            "engine": "reference_human",
            "params": dict(self.params),
            "v2_available": self.v2.available(),
        }

    # --- core: discard ---

    def recommend_discard(self, state: GameState, *, missing_suit: Optional[str] = None) -> Dict:
        # Step 0: 缺一门强制清门——真人在这条规则下没有选择，必须先把缺门牌打掉。
        # 优先级最高，跳过防守 / 番数倾向 / 错牌噪声等所有后续步骤。
        ms = _normalize_missing_suit(missing_suit)
        if ms:
            hand_codes = [normalize_code(t) for t in state.current_player().hand]
            ms_hand = [t for t in hand_codes if _suit_of(t) == ms]
            if ms_hand:
                # B1 现物优先：清缺门时若有对手河里出过的现物（绝对不会放炮），先打它。
                player_wind = state.current_player().wind
                opp_discards = set()
                for w, snap in state.players.items():
                    if w == player_wind:
                        continue
                    for t in snap.discards:
                        opp_discards.add(normalize_code(t))
                ms_genbutsu = [t for t in ms_hand if t in opp_discards]
                pool = ms_genbutsu if ms_genbutsu else ms_hand

                # 在候选池中选 V2 EV 最低（最不可惜）的；退化字典序保证确定性。
                v2_seed = self.v2.recommend_discard(state) or self.strategy.recommend_discard(state)
                cand = (v2_seed.get("candidate_scores", {}) if v2_seed else {})
                cand_pool = {
                    t: float(score) for t, score in cand.items()
                    if normalize_code(t) in set(pool)
                }
                pick = min(cand_pool, key=cand_pool.get) if cand_pool else sorted(pool)[0]
                return {
                    "engine": "reference_human",
                    "tile": normalize_code(pick),
                    "chosen_by": "missing_suit_clear" + ("_genbutsu" if ms_genbutsu else ""),
                    "candidate_scores": {pick: cand_pool.get(pick, 0.0)},
                    "houjuu_prob": 0.0,
                }

        # Step 1: ask V2 for baseline EV decision; fall back to strategy heuristic
        # if V2 .so is not loaded (e.g. dev box without boost).
        result = self.v2.recommend_discard(state)
        if result is None:
            result = self.strategy.recommend_discard(state)

        result["engine"] = "reference_human"
        result.setdefault("chosen_by", "v2_baseline")
        candidate_scores: Dict[str, float] = dict(result.get("candidate_scores", {}))

        # Step 2: 防守层 — 高放炮风险时 betaori
        houjuu_prob = float(result.get("houjuu_prob", 0))
        if houjuu_prob > self.params["betaori_threshold"]:
            if self.rng.random() < self.params["betaori_prob"]:
                # P1-5: 优先打"现物"（手牌中已经在对手河出现过的牌），
                # 这是真人最容易识别的安全策略，比按 EV 升序排更接近真人。
                hand = state.current_player().hand
                opp_discards: set = set()
                player_wind = state.current_player().wind
                for w, snap in state.players.items():
                    if w != player_wind:
                        for t in snap.discards:
                            opp_discards.add(normalize_code(t))
                genbutsu_in_hand = [
                    t for t in hand if normalize_code(t) in opp_discards
                ]
                if genbutsu_in_hand:
                    # 现物里再按手牌出现频次取第一个（避免拆对子）
                    counts: Counter = Counter(genbutsu_in_hand)
                    pick = min(counts, key=counts.get)  # 取频次最低（孤张优先）
                    result["tile"] = pick
                    result["chosen_by"] = "betaori_genbutsu"
                    return result
                # 没现物时退化到按 EV 升序选最低
                if candidate_scores:
                    safest = min(candidate_scores, key=candidate_scores.get)
                    result["tile"] = safest
                    result["chosen_by"] = "betaori"
                    return result

        # Step 3: 番数倾向 — 清一色/混一色 雏形时优先成型
        hand = [normalize_code(tile) for tile in state.current_player().hand]
        target_suit = _detect_hunyise_target(hand)
        if target_suit and candidate_scores:
            same_suit_count = sum(1 for tile in hand if _suit_of(tile) == target_suit)
            if same_suit_count >= self.params["hunyise_min_same_suit"]:
                # Re-rank: discarding off-suit tiles gets a bonus (prefer keeping target suit)
                bonus = self.params["hunyise_bonus"]
                reranked = {
                    tile: ev + (bonus if _suit_of(tile) != target_suit else 0)
                    for tile, ev in candidate_scores.items()
                }
                top = max(reranked, key=reranked.get)
                if top != result.get("tile"):
                    result["tile"] = top
                    result["chosen_by"] = "hunyise_target"
                    result["candidate_scores"] = {
                        k: round(v, 3) for k, v in reranked.items()
                    }

        # Step 4: 错牌噪声
        if self.rng.random() < self.params["mistake_prob"]:
            # Sort candidates by EV descending; pick uniformly from top-2
            sorted_cand = sorted(
                result.get("candidate_scores", {}).items(),
                key=lambda kv: -kv[1],
            )
            top2 = sorted_cand[:2]
            if len(top2) >= 2:
                pick = self.rng.choice(top2)
                result["tile"] = pick[0]
                result["chosen_by"] = "noise"

        return result

    # --- response (chi/peng/gang/hu/pass) ---

    def recommend_response(
        self,
        state: GameState,
        discarded_tile: str,
        from_player: int,
        action_buttons: Optional[List[str]] = None,
    ) -> Dict:
        result = self.v2.recommend_response(state, discarded_tile, from_player, action_buttons)
        if result is None:
            result = self.strategy.recommend_response(state, discarded_tile, from_player, action_buttons)
        result["engine"] = "reference_human"
        return result
