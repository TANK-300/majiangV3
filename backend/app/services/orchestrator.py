from __future__ import annotations

from typing import Dict, List, Optional

from ..core.state import GameState
from .strategy_fallback import StrategyFallbackService
from .v2_fallback import V2FallbackAdapter
from .v3_service import V3SearchService


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

    def recommend(self, state: GameState) -> Dict:
        v3_result = self.v3.recommend_discard(state)
        if v3_result is not None:
            return v3_result
        v2_result = self.v2.recommend_discard(state)
        if v2_result is not None:
            return v2_result
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

