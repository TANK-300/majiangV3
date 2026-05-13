from __future__ import annotations

import os
from typing import Dict, List, Optional

from ..core.state import GameState
from .strategy_fallback import StrategyFallbackService
from .two_player_adapter import adjust_for_2p_linhai, rebalance_for_2p_houjuu
from .v2_fallback import V2FallbackAdapter
from .v3_service import V3SearchService


def _rebalance_factor() -> float:
    """读取环境变量 `LINHAI_V3_REBALANCE_FACTOR`，>0 启用 #4 EV 重平衡。
    默认 0（不影响生产 / 既有测试）。"""
    raw = os.environ.get("LINHAI_V3_REBALANCE_FACTOR", "").strip()
    if not raw:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


class LinhaiV3Orchestrator:
    """新项目主编排器：V3 -> V2 -> heuristic。"""

    def __init__(self) -> None:
        self.v3 = V3SearchService()
        self.v2 = V2FallbackAdapter()
        self.strategy = StrategyFallbackService()

    def engine_status(self) -> Dict[str, object]:
        """Return which engine is actually wired up. Useful for ops to check if the
        service is silently running on the heuristic fallback."""
        v3_status = self.v3.status()
        v2_status = self.v2.status()
        if v3_status.get("available"):
            active = "v3"
        elif v2_status.get("available"):
            active = "v2"
        else:
            active = "heuristic"
        return {
            "active_engine": active,
            "v3": v3_status,
            "v2": v2_status,
            "heuristic_available": True,
        }

    def recommend(
        self,
        state: GameState,
        *,
        mode: Optional[str] = None,
        missing_suit_self: Optional[str] = None,
        missing_suit_opp: Optional[str] = None,
    ) -> Dict:
        """出牌推荐。

        额外 kwargs（默认 None 不影响历史行为）：
            mode: "linhai_2p" 强制启用二人后置适配；None 时适配器走启发式推断。
            missing_suit_self: 自家缺一门（w / t / z 或别名）。
            missing_suit_opp: 对手缺一门。
        """
        # 2026-05-13 实验 #1：无任何 2P 信号时跳过 adapter，直接返回 V3 原始结果。
        # 假设：always-on 的"孤张字牌 +800" / betaori 等规则在 V3 已有 EV 上是双重计分，
        # 把 V3 训练好的字牌策略污染了——4-28 基线 84.8% → 5-12 当前 76.8% (-8pp)。
        # 缺一门 / 显式 mode="linhai_2p" 时 adapter 仍然运行。
        no_adapter_signal = (
            mode is None
            and missing_suit_self is None
            and missing_suit_opp is None
        )
        v3_result = self.v3.recommend_discard(state)
        if v3_result is not None:
            # 2026-05-13 实验 #4：环境变量 LINHAI_V3_REBALANCE_FACTOR > 0 时
            # 用 per-candidate houjuu_prob 做 EV 重平衡（独立于 mode / missing_suit）。
            # 默认 0 = 关闭，行为与之前一致。
            factor = _rebalance_factor()
            if factor > 0:
                v3_result = rebalance_for_2p_houjuu(v3_result, factor)
            if no_adapter_signal:
                return v3_result
            return adjust_for_2p_linhai(
                v3_result,
                state,
                mode=mode,
                missing_suit_self=missing_suit_self,
                missing_suit_opp=missing_suit_opp,
            )
        v2_result = self.v2.recommend_discard(state)
        if v2_result is not None:
            if no_adapter_signal:
                return v2_result
            return adjust_for_2p_linhai(
                v2_result,
                state,
                mode=mode,
                missing_suit_self=missing_suit_self,
                missing_suit_opp=missing_suit_opp,
            )
        return self.strategy.recommend_discard(state)

    def recommend_response(
        self,
        state: GameState,
        discarded_tile: str,
        from_player: int,
        action_buttons: Optional[List[str]] = None,
    ) -> Dict:
        v3_result = self.v3.recommend_response(state, discarded_tile, from_player, action_buttons)
        if v3_result is not None:
            return v3_result
        v2_result = self.v2.recommend_response(state, discarded_tile, from_player, action_buttons)
        if v2_result is not None:
            return v2_result
        return self.strategy.recommend_response(state, discarded_tile, from_player, action_buttons)


_orchestrator: Optional[LinhaiV3Orchestrator] = None


def get_orchestrator() -> LinhaiV3Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = LinhaiV3Orchestrator()
    return _orchestrator

