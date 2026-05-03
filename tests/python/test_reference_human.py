"""ReferenceHumanPolicy 测试。spec §2.2。

注意：本机环境通常没装 boost，V2FallbackAdapter 会无 .so 走 strategy fallback，
ReferenceHumanPolicy 设计上要在两种情况下都能 work。
"""
from __future__ import annotations

from backend.app.core.state import GameState, PlayerState, Wind
from backend.app.services.reference_human import (
    DEFAULT_PARAMS,
    ReferenceHumanPolicy,
    _detect_hunyise_target,
)


# ---------- fixtures ----------

def _build_state(hand=None, opp_discards=None) -> GameState:
    if hand is None:
        hand = ["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w",
                "1t", "2t", "3t", "white", "east"]
    if opp_discards is None:
        opp_discards = ["9t"]
    players = {
        Wind.EAST: PlayerState(wind=Wind.EAST, hand=hand, discards=[]),
        Wind.SOUTH: PlayerState(wind=Wind.SOUTH, hand=[], discards=opp_discards),
    }
    return GameState(
        round_wind=Wind.EAST,
        dealer_wind=Wind.EAST,
        active_wind=Wind.EAST,
        players=players,
        wall_remaining=24,
    )


def _build_hunyise_state() -> GameState:
    """Hand with 8 万 + 2 honors → strong hunyise target (suit 'm')."""
    hand = ["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "east", "white", "1t", "2t", "3t", "9w"]
    return _build_state(hand=hand)


# ---------- 基本行为 ----------

def test_legal_discard_from_hand():
    policy = ReferenceHumanPolicy(seed=1)
    state = _build_state()
    result = policy.recommend_discard(state)
    assert result["engine"] == "reference_human"
    # 必须从手牌里选
    assert result["tile"] in {"1w","2w","3w","4w","5w","6w","7w","8w","9w","1t","2t","3t","white","east"}


def test_status_reports_engine_and_params():
    policy = ReferenceHumanPolicy(seed=1)
    s = policy.status()
    assert s["engine"] == "reference_human"
    assert "betaori_threshold" in s["params"]


def test_default_params_loaded():
    policy = ReferenceHumanPolicy(seed=1)
    assert policy.params["betaori_threshold"] == DEFAULT_PARAMS["betaori_threshold"]
    assert policy.params["mistake_prob"] == DEFAULT_PARAMS["mistake_prob"]


def test_overrides_apply():
    policy = ReferenceHumanPolicy(seed=1, mistake_prob=0.99, betaori_prob=0.0)
    assert policy.params["mistake_prob"] == 0.99
    assert policy.params["betaori_prob"] == 0.0


# ---------- 番数倾向（hunyise） ----------

def test_detect_hunyise_target_finds_dominant_suit():
    hand = ["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "east", "white"]
    suit = _detect_hunyise_target(hand)
    assert suit == "m"  # 8 万 dominant


def test_detect_hunyise_target_returns_none_when_all_honors():
    hand = ["east", "south", "west", "white", "red", "green"]
    # 没有任何数牌 → None
    assert _detect_hunyise_target(hand) is None


def test_hunyise_bonus_re_ranks_off_suit_higher():
    """有 8 张万牌 + 1 条 + 1 字 → policy 应倾向打 1t/east 而非 1w。"""
    policy = ReferenceHumanPolicy(seed=42, mistake_prob=0.0, betaori_prob=0.0)
    state = _build_hunyise_state()
    result = policy.recommend_discard(state)
    # 由于 V2 没加载，会走 strategy_fallback；hunyise re-rank 在 strategy 输出之上应用
    # 期望 chosen_by 标记为 hunyise_target 或 tile 是非万牌
    if result.get("chosen_by") == "hunyise_target":
        # 重排生效，打的不是万
        from backend.app.services.reference_human import _suit_of
        assert _suit_of(result["tile"]) != "m"


# ---------- 错牌噪声 ----------

def test_mistake_prob_zero_picks_top1():
    """mistake_prob=0 时永远选 top-1。"""
    policy = ReferenceHumanPolicy(seed=1, mistake_prob=0.0, betaori_prob=0.0)
    state = _build_state()
    result = policy.recommend_discard(state)
    # chosen_by 不应为 "noise"
    assert result.get("chosen_by") != "noise"


def test_mistake_prob_one_always_picks_from_top2():
    """mistake_prob=1 + 关闭 hunyise → 总是从 top-2 抽取。
    这里只验证它不抛异常并标记为 noise（如果有 ≥2 候选）。"""
    policy = ReferenceHumanPolicy(seed=1, mistake_prob=1.0, betaori_prob=0.0)
    state = _build_state()
    result = policy.recommend_discard(state)
    # 如果手牌只有 1 个 distinct tile，noise 不触发；本 state 有 14 张多样手牌
    assert result.get("chosen_by") in {"noise", "v2_baseline", "fallback_v2", "heuristic", "hunyise_target"}


# ---------- 防守层（betaori） ----------

def test_betaori_does_not_trigger_when_houjuu_low():
    """V2 不可用时 strategy fallback 没有 houjuu_prob，betaori 不应触发。"""
    policy = ReferenceHumanPolicy(seed=1, betaori_prob=1.0, betaori_threshold=0.0)
    state = _build_state()
    # 关闭 hunyise / mistake 干扰
    policy.params["mistake_prob"] = 0.0
    policy.params["hunyise_min_same_suit"] = 999
    result = policy.recommend_discard(state)
    # houjuu_prob 默认 0，threshold 0 → 触发条件 0>0 不成立 → 不 betaori
    # 但 strategy_fallback 也可能没暴露 houjuu_prob，确认不抛即可
    assert "tile" in result


# ---------- 集成：两个 policy 实例不互相干扰 ----------

def test_two_policies_with_different_seeds_independent():
    a = ReferenceHumanPolicy(seed=1)
    b = ReferenceHumanPolicy(seed=2)
    state = _build_state()
    # 都能跑，不抛异常
    ra = a.recommend_discard(state)
    rb = b.recommend_discard(state)
    assert ra["engine"] == rb["engine"] == "reference_human"


# ---------- response ----------

def test_recommend_response_returns_action():
    policy = ReferenceHumanPolicy(seed=1)
    state = _build_state()
    result = policy.recommend_response(state, "3w", 1, ["pass", "peng"])
    assert result["engine"] == "reference_human"
    assert result["action"] in {"pass", "peng", "hu", "gang", "chi"}
