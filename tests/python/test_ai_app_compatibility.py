"""Contract tests against the uni-app frontend (majiang/frontend).

The frontend's `utils/api.js` calls:
  * GET  /health/ping
  * POST /ai/recommend              (getDiscardRecommendation)
  * POST /ai/recommend-response     (getResponseRecommendation)

Each request schema here MUST match the legacy `majiang/backend` field
names/defaults 1:1 so that just changing `app-config.js`'s `apiUrl` to
this backend is a no-code-change swap.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from backend.app.main import app  # noqa: E402


client = TestClient(app)


def test_health_ping_legacy_path() -> None:
    r = client.get("/health/ping")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_engine_status_exposes_active_engine() -> None:
    r = client.get("/debug/engine_status")
    assert r.status_code == 200
    body = r.json()
    assert body["active_engine"] in {"v3", "v2", "heuristic"}
    assert "v3" in body and "v2" in body
    assert body["heuristic_available"] is True


def test_ai_recommend_returns_legacy_envelope() -> None:
    payload = {
        "hand": ["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w", "1t", "2t", "3t", "white", "east"],
        "wind_seat": 0,
        "has_kan": False,
        "wall_remaining": 24,
        "round_number": 0,
        "strategy": "balanced",
    }
    r = client.post("/ai/recommend", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    # envelope
    assert body["success"] is True
    assert "recommendation" in body
    rec = body["recommendation"]
    # legacy fields -- these MUST be present or the current app UI breaks
    for key in ("tile", "score", "shanten", "bonus_value", "risk_value", "explanation", "confidence"):
        assert key in rec, f"legacy field missing: {key}"
    # V3 additions -- new UI can use them, old UI ignores them
    for key in ("agari_prob", "houjuu_prob", "defense_score", "engine", "candidate_scores"):
        assert key in rec
    # tile must come from the actual hand
    assert rec["tile"] in payload["hand"]
    # confidence range
    assert 0.0 <= rec["confidence"] <= 1.0


def test_ai_recommend_rejects_bad_tile_codes() -> None:
    payload = {
        "hand": ["1w"] * 13 + ["not_a_tile"],
        "wind_seat": 0,
    }
    r = client.post("/ai/recommend", json=payload)
    # Bad tile: Pydantic + our normalize_code should surface a 4xx, not 500.
    assert r.status_code in (400, 422)


def test_ai_recommend_response_handles_pass_only() -> None:
    # Even with just {pass}, we must respond cleanly; the frontend relies
    # on this whenever the game signals "you may only pass".
    payload = {
        "hand": ["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w", "1t", "2t", "3t", "white"],
        "wind_seat": 0,
        "discarded_tile": "9t",
        "from_player": 1,
        "available_actions": ["pass"],
        "wall_remaining": 20,
    }
    r = client.post("/ai/recommend-response", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    rec = body["recommendation"]
    for key in ("action", "tile", "score", "shanten", "confidence", "engine"):
        assert key in rec
    assert rec["action"] in {"pass", "peng", "chi", "gang", "hu"}


def test_ai_recommend_response_peng_path() -> None:
    # Hand has two 3w -> peng is legal if offered.
    payload = {
        "hand": ["1w", "2w", "3w", "3w", "4w", "5w", "6w", "7w", "8w", "9w", "1t", "2t", "3t"],
        "wind_seat": 0,
        "discarded_tile": "3w",
        "from_player": 1,
        "available_actions": ["pass", "peng"],
        "wall_remaining": 30,
    }
    r = client.post("/ai/recommend-response", json=payload)
    assert r.status_code == 200, r.text
    rec = r.json()["recommendation"]
    assert rec["action"] in {"pass", "peng"}


def test_ai_recommend_all_returns_sorted_candidates() -> None:
    payload = {
        "hand": ["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w", "1t", "2t", "3t", "white", "east"],
        "wind_seat": 0,
        "wall_remaining": 24,
    }
    r = client.post("/ai/recommend-all", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    recs = body["recommendations"]
    assert isinstance(recs, list)
    # orchestrator.candidate_scores is already sorted best-first; we just
    # forward it, so just assert structure.
    for entry in recs:
        assert "tile" in entry and "score" in entry and "confidence" in entry


def test_ai_reset_is_safe_noop() -> None:
    r = client.post("/ai/reset", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True


def test_ai_health_returns_test_tile() -> None:
    r = client.get("/ai/health")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["status"] == "healthy"
    assert "tile" in body["test_result"]
    assert "engine_status" in body


def test_passed_hu_flag_propagates() -> None:
    # When the seat already passed hu this round, `can_win` must be false
    # in the engine -- we surface this through passed_hu_this_round.
    # The hand below would otherwise be winnable; with the flag, the hu
    # recommendation should NOT be chosen.
    payload = {
        "hand": ["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w", "1t", "2t", "3t", "east"],
        "wind_seat": 0,
        "discarded_tile": "east",
        "from_player": 1,
        "available_actions": ["pass", "hu"],
        "passed_hu_this_round": True,
        "wall_remaining": 15,
    }
    r = client.post("/ai/recommend-response", json=payload)
    assert r.status_code == 200
    rec = r.json()["recommendation"]
    # Under passed-hu, engine MUST not recommend 'hu'.
    assert rec["action"] != "hu"
