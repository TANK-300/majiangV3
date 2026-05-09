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


def test_ai_recommend_passes_own_melds_to_engine() -> None:
    """同一 hand+路必须因为 melds 不同而产生不同 EV/防御评分，
    以证明副露真的走到了 V3 引擎而不是被丢弃。"""
    base = {
        "hand": ["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w", "1t", "2t", "3t", "4t"],
        "wind_seat": 0,
        "wall_remaining": 40,
    }
    r1 = client.post("/ai/recommend", json=base)
    r2 = client.post(
        "/ai/recommend",
        json={**base, "melds": [{"type": "peng", "tiles": ["green", "green", "green"]}]},
    )
    assert r1.status_code == 200 and r2.status_code == 200
    s1 = r1.json().get("score")
    s2 = r2.json().get("score")
    assert isinstance(s1, (int, float)) and isinstance(s2, (int, float))
    # 副露会改变实际手牌长度、完型距离与防御特征→ EV 必变化；若相等则代表被丢弃。
    assert s1 != s2, f"melds passthrough failed: same EV {s1} vs {s2}"


def test_ai_recommend_meld_type_aliases_accepted() -> None:
    """前端可能使用日式别名 (pon/chii/kan)，路由层必须能归一。"""
    payload = {
        "hand": ["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w", "1t", "2t", "3t", "4t"],
        "wind_seat": 0,
        "wall_remaining": 40,
        "melds": [
            {"type": "pon", "tiles": ["green", "green", "green"]},  # 别名 -> peng
        ],
    }
    r = client.post("/ai/recommend", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["action"] == "discard"
    assert body["tile"] in payload["hand"]


def test_ai_recommend_unknown_meld_type_dropped_safely() -> None:
    """未知 type 不该崩服务，只被静默跳过。"""
    payload = {
        "hand": ["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w", "1t", "2t", "3t", "4t"],
        "wind_seat": 0,
        "wall_remaining": 40,
        "melds": [
            {"type": "garbage_meld", "tiles": ["green", "green", "green"]},
        ],
    }
    r = client.post("/ai/recommend", json=payload)
    assert r.status_code == 200, r.text


def test_ai_recommend_response_accepts_opponent_melds() -> None:
    """响应推荐接受 opponent_melds + own melds 同时存在。"""
    payload = {
        "hand": ["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w", "1t", "2t"],
        "wind": "east",
        "discarded_tile": "green",
        "from_player": 1,
        "action_buttons": ["guo"],
        "wall_remaining": 30,
        "melds": [{"type": "peng", "tiles": ["red", "red", "red"]}],
        "opponent_melds": [
            {"type": "chi", "tiles": ["1t", "2t", "3t"], "from_player": 1},
        ],
    }
    r = client.post("/ai/recommend-response", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert body["recommended"] in {"pass", "peng", "chi", "gang", "hu"}


def test_ai_recommend_response_opp_melds_legacy_alias() -> None:
    """兼容旧名 opp_melds -> opponent_melds。"""
    payload = {
        "hand": ["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w", "1t", "2t"],
        "wind_seat": 0,
        "discarded_tile": "9w",
        "from_player": 1,
        "available_actions": ["pass"],
        "wall_remaining": 30,
        "opp_melds": [{"type": "peng", "tiles": ["red", "red", "red"]}],
    }
    r = client.post("/ai/recommend-response", json=payload)
    assert r.status_code == 200, r.text


def test_2p_adapter_promotes_lone_honor_for_discard() -> None:
    """二人临海：字牌不是 yakuhai，孤张应优先打。

    hand: 1w-6w + 1t-7t + east — east 是孤张字牌，优先打才能多听。
    """
    payload = {
        "hand": ["1w", "2w", "3w", "4w", "5w", "6w", "1t", "2t", "3t", "4t", "5t", "6t", "7t", "east"],
        "wind_seat": 0,
        "wall_remaining": 50,
        # 未带 mode → 启发式推断应识别为 2p（其他座位无牌河/副露）
    }
    r = client.post("/ai/recommend", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tile"] == "east", f"孤张 east 未被 adapter 提为 top: {body['tile']}, scores={body['meta']['scores']}"


def test_2p_adapter_explicit_mode_lowers_houjuu() -> None:
    """mode=linhai_2p 时 houjuu_prob 应被折低（0.4 倍）。"""
    payload = {
        "hand": ["1w", "2w", "3w", "4w", "5w", "6w", "1t", "2t", "3t", "4t", "5t", "6t", "7t", "east"],
        "wind_seat": 0,
        "wall_remaining": 50,
        "mode": "linhai_2p",
    }
    r = client.post("/ai/recommend", json=payload)
    assert r.status_code == 200, r.text
    h = r.json()["houjuu_prob"]
    # 原始 4p 估计通常 0.2-0.3，折后应 < 0.15。
    assert h < 0.15, f"linhai_2p 未调低放炮率: {h}"


def test_2p_adapter_missing_suit_self_forces_discard() -> None:
    """缺一门：该门牌必须被 adapter 发上优先。"""
    base = {
        "hand": ["1w", "2w", "3w", "4w", "5w", "6w", "1t", "2t", "3t", "4t", "5t", "6t", "7t", "east"],
        "wind_seat": 0,
        "wall_remaining": 50,
    }
    # 缺万 → 推万牌
    r = client.post("/ai/recommend", json={**base, "missing_suit_self": "w"})
    assert r.status_code == 200
    assert r.json()["tile"].endswith("w"), f"缺万但未推万: {r.json()['tile']}"
    # 缺字 → 推字牌
    r = client.post("/ai/recommend", json={**base, "missing_suit_self": "z"})
    assert r.status_code == 200
    assert r.json()["tile"] in {"east", "south", "west", "north", "white", "green", "red"}


def test_2p_adapter_missing_suit_alias_accepted() -> None:
    """missing_suit 接受别名 wan/万/m 等。"""
    base = {
        "hand": ["1w", "2w", "3w", "4w", "5w", "6w", "1t", "2t", "3t", "4t", "5t", "6t", "7t", "east"],
        "wind_seat": 0,
        "wall_remaining": 50,
    }
    for alias in ("wan", "万", "m"):
        r = client.post("/ai/recommend", json={**base, "missing_suit_self": alias})
        assert r.status_code == 200, r.text
        assert r.json()["tile"].endswith("w"), f"alias={alias} 未生效: {r.json()['tile']}"


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
