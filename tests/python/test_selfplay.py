from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.selfplay_eval import (  # noqa: E402
    HeuristicPolicy,
    RandomPolicy,
    is_winning_hand,
    run_match,
)


def test_is_winning_hand_detects_standard_hand() -> None:
    # 1w2w3w, 4w5w6w, 7w8w9w, 1t2t3t, east east -> valid 4 melds + pair
    hand = ["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w", "1t", "2t", "3t", "east", "east"]
    assert is_winning_hand(hand) is True


def test_is_winning_hand_rejects_clearly_bad_hand() -> None:
    hand = ["1w", "2w", "4w", "5w", "7w", "8w", "1t", "3t", "5t", "7t", "east", "south", "west", "north"]
    assert is_winning_hand(hand) is False


def test_white_acts_as_wildcard_for_completion() -> None:
    # Missing 3w in the first run; one white covers it.
    hand = ["1w", "2w", "white", "4w", "5w", "6w", "7w", "8w", "9w", "1t", "2t", "3t", "east", "east"]
    assert is_winning_hand(hand) is True


def test_run_match_symmetric_two_heuristics() -> None:
    # With enough games and side swapping, heuristic vs heuristic should stay
    # within a reasonable band around 50%.
    stats = run_match(
        HeuristicPolicy(),
        HeuristicPolicy(),
        games=200,
        seed=1234,
        swap_sides=True,
        max_turns=120,
    )
    assert stats.games_played == 200
    summary = stats.as_dict("h1", "h2")
    assert 0.35 <= summary["winrate_a"] <= 0.65
    assert 0.35 <= summary["winrate_b"] <= 0.65
    # Results must sum to 1.0 (wins_a + wins_b + draws == games_played)
    assert stats.wins_a + stats.wins_b + stats.draws == 200


def test_heuristic_beats_random_clearly() -> None:
    rng = random.Random(7)
    stats = run_match(
        HeuristicPolicy(),
        RandomPolicy(rng),
        games=60,
        seed=7,
        swap_sides=True,
        max_turns=120,
    )
    summary = stats.as_dict("h", "r")
    # Heuristic should win clearly against random. We give a generous floor
    # to avoid flakiness; in practice it wins ~85-95%.
    assert summary["winrate_a"] >= 0.60


def test_selfplay_eval_cli_emits_summary(tmp_path: Path) -> None:
    tool = TOOLS_DIR / "selfplay_eval.py"
    output_path = tmp_path / "stats.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(tool),
            "--policy-a",
            "heuristic",
            "--policy-b",
            "random",
            "--games",
            "10",
            "--seed",
            "99",
            "--max-turns",
            "80",
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    summary = json.loads(result.stdout)
    assert summary["games_played"] == 10
    assert 0.0 <= summary["winrate_a"] <= 1.0
    assert 0.0 <= summary["winrate_b"] <= 1.0
    assert summary["policy_a"] == "heuristic"
    assert summary["policy_b"] == "random"
    assert output_path.exists()
    line = output_path.read_text(encoding="utf-8").strip()
    assert json.loads(line)["games_played"] == 10


def test_orchestrator_policy_routes_through_engine_status() -> None:
    pytest.importorskip("fastapi")
    from tools.selfplay_eval import OrchestratorPolicy

    policy = OrchestratorPolicy()
    # On a fresh CI VM the C++ extensions are unavailable, so the active engine
    # should be "heuristic". This test documents the expected fallback chain.
    assert policy._active in {"v3", "v2", "heuristic"}


def test_engine_status_endpoint_reports_fallback_chain() -> None:
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from backend.app.main import app

    client = TestClient(app)
    response = client.get("/debug/engine_status")
    assert response.status_code == 200
    body = response.json()
    assert body["active_engine"] in {"v3", "v2", "heuristic"}
    assert "v3" in body and "v2" in body
    assert body["heuristic_available"] is True
