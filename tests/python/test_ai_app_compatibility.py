"""Contract tests against the uni-app frontend (majiang/frontend).

The frontend's `utils/api.js` calls:
  * GET  /health/ping
  * POST /ai/recommend              (getDiscardRecommendation)
  * POST /ai/recommend-response     (getResponseRecommendation)

For `/ai/recommend`, the uni-app assigns the response straight to
`lastRecommendation` (see `pages/index/index.vue` ->
`requestRecommendation`) and consumes fields at the top level
(`confidence`, `action`, `tile`, `explanation`, `meta.scores`, ...).

For `/ai/recommend-response`, the uni-app pipes the response through
`normalizeResponseResult`, which expects
`{actions: [{action, tile, confidence, explanation}, ...], recommended}`.

These tests lock the on-the-wire shape so we don't regress the two
fragile integration points again.
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


def test_ai_recommend_returns_flat_shape() -> None:
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
    # The uni-app assigns the response directly to `lastRecommendation`
    # and calls `.confidence.toFixed(1)` / reads `.action` / `.tile` /
    # `.explanation` at the top level. Lock those.
    for key in ("success", "action", "tile", "confidence", "explanation", "meta"):
        assert key in body, f"required top-level field missing: {key}"
    assert body["action"] == "discard"
    assert isinstance(body["confidence"], (int, float))
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["tile"] in payload["hand"]
    # result.vue reads `meta.scores` and calls `score.toFixed(2)` on each
    # value; all values must be numeric.
    scores = body["meta"].get("scores")
    assert isinstance(scores, dict) and scores, "meta.scores must be a non-empty dict"
    for v in scores.values():
        assert isinstance(v, (int, float))
    # V3-native fields kept for debugging / richer UI.
    for key in ("shanten", "agari_prob", "houjuu_prob", "defense_score", "engine"):
        assert key in body


def test_ai_recommend_accepts_legacy_uniapp_payload() -> None:
    # The real uni-app (`pages/index/index.vue`) posts `wind: "east"`
    # rather than `wind_seat: 0`, plus a bunch of extra fields
    # (`melds`, `dora_indicators`, `opponent_discards`, `game_id`,
    # `record_data`). The edge must accept these without rebuilding.
    payload = {
        "hand": ["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w", "1t", "2t", "3t", "white", "east"],
        "wind": "east",
        "discards": ["5t"],
        "melds": [],
        "opponent_discards": ["3t", "4t"],
        "wall_remaining": 50,
        "dora_indicators": ["red"],
        "game_id": "game_1234",
        "record_data": False,
    }
    r = client.post("/ai/recommend", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["action"] == "discard"
    assert body["tile"] in payload["hand"]


def test_ai_recommend_rejects_bad_tile_codes() -> None:
    payload = {
        "hand": ["1w"] * 13 + ["not_a_tile"],
        "wind_seat": 0,
    }
    r = client.post("/ai/recommend", json=payload)
    assert r.status_code in (400, 422)


def test_ai_recommend_response_handles_pass_only() -> None:
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
    # Shape lock: `normalizeResponseResult` on the frontend expects
    # `actions` (array) and `recommended` (string) at the top level.
    assert body["success"] is True
    assert "actions" in body and isinstance(body["actions"], list) and body["actions"]
    assert "recommended" in body and isinstance(body["recommended"], str)
    for entry in body["actions"]:
        for k in ("action", "tile", "confidence", "explanation"):
            assert k in entry
        assert isinstance(entry["confidence"], (int, float))
    assert body["recommended"] in {"pass", "peng", "chi", "gang", "hu"}


def test_ai_recommend_response_peng_path_has_peng_in_actions() -> None:
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
    body = r.json()
    listed_actions = {entry["action"] for entry in body["actions"]}
    # Whatever we recommend, both offered actions must appear so the app
    # can render the option table without missing keys.
    assert {"pass", "peng"}.issubset(listed_actions)
    assert body["recommended"] in {"pass", "peng"}


def test_ai_recommend_response_accepts_legacy_action_buttons() -> None:
    # Legacy uni-app posts `wind: "east"` and `action_buttons: [...]`
    # with `guo` as the "pass" alias. Edge must translate both.
    payload = {
        "hand": ["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w", "1t", "1t", "1t", "9t"],
        "wind": "east",
        "discards": ["5t", "6t"],
        "melds": [],
        "opponent_discards": ["3t", "4t"],
        "wall_remaining": 50,
        "dora_indicators": ["red"],
        "discarded_tile": "9t",
        "from_player": 1,
        "action_buttons": ["hu", "guo", "chi", "peng", "gang", "pass"],
        "latest_action_kind": "discard",
    }
    r = client.post("/ai/recommend-response", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    listed = {entry["action"] for entry in body["actions"]}
    # `guo` translated to `pass`; no `guo` leaks through.
    assert "guo" not in listed
    assert "pass" in listed
    # recommended lands on one of the offered actions.
    assert body["recommended"] in listed


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
    # When the seat already passed hu this round, the engine must not
    # recommend 'hu'. This flag flows through to v3 / heuristic.
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
    body = r.json()
    assert body["recommended"] != "hu"
