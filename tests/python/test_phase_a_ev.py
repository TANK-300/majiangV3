"""Phase A spec §3.5.1 — 风险厌恶 EV 在高压力局面下偏好防守。

测试逻辑：
- 通过 LinhaiV3Orchestrator 构造 GameState（与 test_orchestrator.py 对齐的真实 schema）。
- 直接调用底层 V3SearchService 的引擎以拿到完整 SearchCandidate 列表（含每个候选
  的 houjuu_prob），因为 orchestrator.recommend() 只返回 `{tile: ev}` 摘要。
- 高压场景比较 top vs second 候选的 houjuu_prob，验证 A4 EV 重加权确实把
  ranking 往低放炮方向推。
"""
from __future__ import annotations

from typing import List

import pytest

from backend.app.core.state import GameState, Meld, PlayerState, Wind
from backend.app.services.orchestrator import LinhaiV3Orchestrator


def _make_state_low_pressure() -> GameState:
    """对手无副露、对手弃牌干净 → opp_pressure_score ≈ 0。"""
    players = {
        Wind.EAST: PlayerState(
            wind=Wind.EAST,
            hand=[
                "1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w",
                "1t", "2t", "3t", "5t", "white",
            ],
            discards=[],
        ),
        Wind.SOUTH: PlayerState(
            wind=Wind.SOUTH,
            hand=[],
            discards=["9t", "8t"],
        ),
    }
    return GameState(
        round_wind=Wind.EAST,
        dealer_wind=Wind.EAST,
        active_wind=Wind.EAST,
        players=players,
        wall_remaining=50,
    )


def _make_state_high_pressure() -> GameState:
    """对手 3 副露（含白板）+ 字牌刻子 + 大量万子可见 → opp_pressure_score > 0.3。

    设计要点：
    - 对手 3 副露 → meld_term = 0.30 * clamp01(3/3) = 0.30
    - 含 1 副白板刻子 → white_term = 0.10
    - 含 1 副字牌刻子 → ziyise_term = 0.20 * clamp01(1/2) = 0.10
    - 对手弃牌 6 张万子 → qingyise_term = 0.25 * clamp01((6-5)/4) = 0.0625
    - 合计 ≈ 0.5625，远超 0.3 阈值，确保 μ 分支与 λ 增量都触发。
    """
    players = {
        Wind.EAST: PlayerState(
            wind=Wind.EAST,
            hand=[
                "1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w",
                "1t", "2t", "3t", "5t", "white",
            ],
            discards=[],
        ),
        Wind.SOUTH: PlayerState(
            wind=Wind.SOUTH,
            hand=[],
            melds=[
                Meld(tiles=["white", "white", "white"], type="peng"),
                Meld(tiles=["east", "east", "east"], type="peng"),
                Meld(tiles=["3w", "3w", "3w"], type="peng"),
            ],
            discards=["1w", "2w", "5w", "8w", "9w", "4w"],
        ),
    }
    return GameState(
        round_wind=Wind.EAST,
        dealer_wind=Wind.EAST,
        active_wind=Wind.EAST,
        players=players,
        wall_remaining=50,
    )


def _engine_candidates(orch: LinhaiV3Orchestrator, state: GameState) -> List:
    """直接调引擎，取完整候选列表（含 houjuu_prob 等字段）。"""
    svc = orch.v3
    assert svc.available(), "V3 engine must be available for this test"
    canonical = svc._build_canonical_state(state)
    result = svc._engine.recommend_discard_v3(canonical)
    return list(result.candidate_scores)


def test_low_pressure_ev_unchanged():
    """低压力下 Phase A EV 仍能产出有效弃牌推荐（recommend 返回 action='discard'）。"""
    orch = LinhaiV3Orchestrator()
    rec = orch.recommend(_make_state_low_pressure())
    assert rec["action"] == "discard", f"expected discard, got {rec['action']}"
    assert rec.get("tile"), "expected a chosen tile"


def test_high_pressure_pressure_score_above_threshold():
    """High-pressure scenario should produce opp_pressure_score above the
    spec threshold (>0.3) so the μ branch in EV is exercised."""
    orch = LinhaiV3Orchestrator()
    svc = orch.v3
    assert svc.available()
    canonical = svc._build_canonical_state(_make_state_high_pressure())
    pressure = svc._engine.compute_opp_pressure_score(canonical)
    assert pressure > 0.3, f"expected high pressure > 0.3, got {pressure:.3f}"


def test_high_pressure_prefers_low_houjuu():
    """高压力下 EV 重加权应让 top 候选的 houjuu_prob 不高于次优 +0.02 容差。

    若 A4 的动态 λ/μ 没接进 EV，top 仍可能选高放炮但攻击好的牌，导致比较失败。
    """
    orch = LinhaiV3Orchestrator()
    candidates = _engine_candidates(orch, _make_state_high_pressure())
    assert len(candidates) >= 2, "expected multiple candidates"
    top = candidates[0]
    second = candidates[1]
    assert top.houjuu_prob <= second.houjuu_prob + 0.02, (
        f"top houjuu={top.houjuu_prob:.4f} > second={second.houjuu_prob:.4f} "
        f"(top.hai={top.hai}, second.hai={second.hai})"
    )


# 不再做"高压 top houjuu < 低压 top houjuu"的跨 state 比较：
# 不同 state 的 baseline heuristic_houjuu_prob 本就不同（高压含 +0.016 副露项），
# 而"宁愿输 shanten 也要保命"是 A5 硬 betaori 门的责任，不是 A4。
# A4 的正确行为已由 test_high_pressure_prefers_low_houjuu 覆盖（同 state 内
# top vs second 比较）。


def _make_state_extreme_pressure() -> GameState:
    """极端高压：对手 3 副露 + 字牌刻子 + 万子全可见，opp_pressure 应 > 0.6。

    所有非现物的弃牌候选 houjuu_prob 都 > betaori_thresholds.base（默认 0.15），
    应触发硬 betaori 门，把 ranking 翻为低 houjuu 优先。
    """
    players = {
        Wind.EAST: PlayerState(
            wind=Wind.EAST,
            hand=[
                # 故意构造没有立即 tenpai 路径的手牌：白板 + 散张 + 字牌
                # 强迫引擎在所有候选都不安全时做选择
                "1w", "5w", "9w", "1t", "3t", "5t", "7t", "9t",
                "east", "south", "west", "north", "white", "red",
            ],
            discards=[],
        ),
        Wind.SOUTH: PlayerState(
            wind=Wind.SOUTH,
            hand=[],
            melds=[
                Meld(tiles=["white", "white", "white"], type="peng"),
                Meld(tiles=["east", "east", "east"], type="peng"),
                Meld(tiles=["3w", "3w", "3w"], type="peng"),
            ],
            # 弃牌覆盖较少 tile，让大部分候选不是 genbutsu
            discards=["2w", "8w"],
        ),
    }
    return GameState(
        round_wind=Wind.EAST,
        dealer_wind=Wind.EAST,
        active_wind=Wind.EAST,
        players=players,
        wall_remaining=50,
    )


def test_extreme_pressure_forces_lowest_houjuu():
    """spec §3.5.2 — 极端高压下，全候选 houjuu_prob 都超过阈值时，
    硬 betaori 门必须把 top 翻为 houjuu_prob 最低的候选。
    """
    orch = LinhaiV3Orchestrator()
    candidates = _engine_candidates(orch, _make_state_extreme_pressure())
    assert len(candidates) >= 2, "expected multiple candidates"
    min_houjuu = min(c.houjuu_prob for c in candidates)
    top_houjuu = candidates[0].houjuu_prob
    # 容差 0.005 — 浮点精度，且允许并列时引擎选其中任一
    assert top_houjuu <= min_houjuu + 0.005, (
        f"hard betaori gate failed: top houjuu={top_houjuu:.4f} > "
        f"min={min_houjuu:.4f} (top.hai={candidates[0].hai})"
    )
