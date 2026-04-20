from __future__ import annotations

from backend.app.core.state import GameState, PlayerState, Wind
from backend.app.services.orchestrator import LinhaiV3Orchestrator
from backend.app.services.strategy_fallback import StrategyFallbackService


COMMON_DEBUG_FIELDS = {
    "total_ev",
    "agari_prob",
    "houjuu_prob",
    "defense_score",
    "search_depth",
    "nodes_expanded",
    "search_nodes",
    "root_candidates_total",
    "root_candidates_evaluated",
    "configured_time_budget_ms",
    "configured_node_budget",
    "cache_hits",
    "truncated",
    "fallback_reason",
    "truncate_reason",
}


def build_state() -> GameState:
    players = {
        Wind.EAST: PlayerState(
            wind=Wind.EAST,
            hand=["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w", "1t", "2t", "3t", "white", "east"],
            discards=["9t"],
        ),
        Wind.SOUTH: PlayerState(
            wind=Wind.SOUTH,
            hand=[],
            discards=["1w", "2w", "east"],
        ),
    }
    return GameState(
        round_wind=Wind.EAST,
        dealer_wind=Wind.EAST,
        active_wind=Wind.EAST,
        players=players,
        wall_remaining=24,
    )


def assert_common_debug_fields(result: dict) -> None:
    for key in COMMON_DEBUG_FIELDS:
        assert key in result
    assert result["search_nodes"] >= 0
    assert result["nodes_expanded"] >= 0
    assert result["root_candidates_total"] >= 0
    assert result["root_candidates_evaluated"] >= 0
    assert result["root_candidates_evaluated"] <= result["root_candidates_total"]
    assert result["cache_hits"] >= 0


def test_orchestrator_recommend_discard_returns_engine_tag() -> None:
    orchestrator = LinhaiV3Orchestrator()
    result = orchestrator.recommend(build_state())
    assert result["engine"] in {"v3", "v2", "heuristic"}
    assert result["action"] == "discard"
    assert "candidate_scores" in result
    assert_common_debug_fields(result)


def test_orchestrator_recommend_response_returns_action() -> None:
    orchestrator = LinhaiV3Orchestrator()
    result = orchestrator.recommend_response(build_state(), "3w", 1, ["pass", "peng"])
    assert result["engine"] in {"v3", "v2", "heuristic"}
    assert result["action"] in {"pass", "peng", "hu", "gang", "chi"}
    assert_common_debug_fields(result)


def test_strategy_fallback_discard_returns_uniform_fields() -> None:
    fallback = StrategyFallbackService()
    result = fallback.recommend_discard(build_state())
    assert result["engine"] == "heuristic"
    assert result["action"] == "discard"
    assert_common_debug_fields(result)


def test_strategy_fallback_response_returns_uniform_fields() -> None:
    fallback = StrategyFallbackService()
    result = fallback.recommend_response(build_state(), "3w", 1, ["pass", "peng"])
    assert result["engine"] == "heuristic"
    assert result["action"] in {"pass", "hu"}
    assert_common_debug_fields(result)
