from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

from ..core.state import GameState
from ..core.tiles import normalize_code, require_tile


class StrategyFallbackService:
    """轻量启发式兜底，不依赖任何 C++ 扩展。"""

    def recommend_discard(self, state: GameState) -> Dict:
        player = state.current_player()
        hand = [normalize_code(tile) for tile in player.hand]
        counts = Counter(hand)

        def score(tile: str) -> float:
            descriptor = require_tile(tile)
            base = 0.0
            if tile == "white":
                return -9999.0
            if counts[tile] == 1:
                base += 3.0
            elif counts[tile] >= 3:
                base -= 2.0
            if descriptor.is_honor:
                base += 2.0
            elif descriptor.rank in (1, 9):
                base += 1.5
            elif descriptor.rank is not None and 3 <= descriptor.rank <= 7:
                base -= 1.0
            return base

        best = max(hand, key=score)
        candidate_scores = {tile: round(score(tile), 3) for tile in sorted(set(hand))}
        total_ev = round(candidate_scores[best], 3)
        return {
            "engine": "heuristic",
            "chosen_by": "heuristic",
            "action": "discard",
            "tile": best,
            "candidate_scores": candidate_scores,
            "total_ev": total_ev,
            "agari_prob": 0.0,
            "houjuu_prob": 0.0,
            "defense_score": 0.0,
            "search_depth": 0,
            "nodes_expanded": len(candidate_scores),
            "search_nodes": len(candidate_scores),
            "root_candidates_total": len(candidate_scores),
            "root_candidates_evaluated": len(candidate_scores),
            "configured_time_budget_ms": -1,
            "configured_node_budget": -1,
            "cache_hits": 0,
            "truncated": False,
            "fallback_reason": "v3_and_v2_unavailable",
            "truncate_reason": "",
        }

    def recommend_response(
        self,
        state: GameState,
        discarded_tile: str,
        from_player: int,
        action_buttons: Optional[List[str]] = None,
    ) -> Dict:
        actions = [str(action).strip().lower() for action in (action_buttons or ["pass"]) if str(action).strip()]
        if "hu" in actions:
            return {
                "engine": "heuristic",
                "chosen_by": "heuristic",
                "action": "hu",
                "tile": normalize_code(discarded_tile),
                "candidate_scores": {"hu": 1.0},
                "total_ev": 1.0,
                "agari_prob": 1.0,
                "houjuu_prob": 0.0,
                "defense_score": 0.0,
                "search_depth": 0,
                "nodes_expanded": 1,
                "search_nodes": 1,
                "root_candidates_total": 1,
                "root_candidates_evaluated": 1,
                "configured_time_budget_ms": -1,
                "configured_node_budget": -1,
                "cache_hits": 0,
                "truncated": False,
                "fallback_reason": "v3_and_v2_unavailable",
                "truncate_reason": "",
            }
        candidate_scores = {action: 0.0 for action in actions or ["pass"]}
        return {
            "engine": "heuristic",
            "chosen_by": "heuristic",
            "action": "pass",
            "tile": None,
            "candidate_scores": candidate_scores,
            "total_ev": 0.0,
            "agari_prob": 0.0,
            "houjuu_prob": 0.0,
            "defense_score": 0.0,
            "search_depth": 0,
            "nodes_expanded": len(candidate_scores),
            "search_nodes": len(candidate_scores),
            "root_candidates_total": len(candidate_scores),
            "root_candidates_evaluated": len(candidate_scores),
            "configured_time_budget_ms": -1,
            "configured_node_budget": -1,
            "cache_hits": 0,
            "truncated": False,
            "fallback_reason": "v3_and_v2_unavailable",
            "truncate_reason": "",
        }
