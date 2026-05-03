"""特征名一致性测试。

确保训练侧 (tools/extract_canonical_states.py) 和推理侧
(engine/share/linhai_search_v3.cpp::build_state_features) 产出的 safety_*
特征键完全一致。

如果 C++ .so 不可用（boost 缺失环境），用源码静态扫描兜底。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from tools.extract_canonical_states import (
    PER_TILE_FEATURE_ORDER,
    build_model_features,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CPP_FILE = REPO_ROOT / "engine" / "share" / "linhai_search_v3.cpp"


def _make_minimal_state(opp_discards=None, hand_counts=None, remaining_counts=None):
    return {
        "active_player": {
            "wind": "east",
            "hand": [],
            "hand_counts": hand_counts or {code: 0 for code in PER_TILE_FEATURE_ORDER},
            "melds": [],
            "discards": [],
            "tree_revealed": False,
            "grab_charge_hits": 0,
            "grab_charge_limit": 0,
            "contract_targets": [],
            "contract_counter": 0,
            "passed_hu_this_round": False,
        },
        "remaining_counts": remaining_counts or {code: 4 for code in PER_TILE_FEATURE_ORDER},
        "available_actions": ["discard"],
        "can_win": True,
        "wall_remaining": 24,
        "from_player": 1,
        "target_hai": None,
        "tree_active": False,
        "grab_charge_active": False,
        "contract_target_count": 0,
        "opponent_meld_count": 0,
        "opponent_discard_count": 0,
        "opponent_discards": opp_discards or [],
    }


# ---------- Python 端：feature names 存在性 ----------

def test_python_emits_safety_genbutsu_for_all_tiles():
    state = _make_minimal_state()
    feats = build_model_features(**state)
    for tile in PER_TILE_FEATURE_ORDER:
        assert f"safety_t_{tile}" in feats


def test_python_emits_safety_suji_for_all_tiles():
    state = _make_minimal_state()
    feats = build_model_features(**state)
    for tile in PER_TILE_FEATURE_ORDER:
        assert f"safety_suji_t_{tile}" in feats


def test_python_emits_safety_kabe_for_all_tiles():
    state = _make_minimal_state()
    feats = build_model_features(**state)
    for tile in PER_TILE_FEATURE_ORDER:
        assert f"safety_kabe_t_{tile}" in feats


# ---------- 行为正确性 ----------

def test_suji_4w_marks_1w_and_7w_safe():
    state = _make_minimal_state(opp_discards=["4w"])
    feats = build_model_features(**state)
    assert feats["safety_suji_t_1w"] == 1.0
    assert feats["safety_suji_t_7w"] == 1.0
    assert feats["safety_suji_t_4w"] == 0.0  # 4w 自己不算 suji


def test_suji_5t_marks_2t_and_8t():
    state = _make_minimal_state(opp_discards=["5t"])
    feats = build_model_features(**state)
    assert feats["safety_suji_t_2t"] == 1.0
    assert feats["safety_suji_t_8t"] == 1.0


def test_suji_6w_marks_3w_and_9w():
    state = _make_minimal_state(opp_discards=["6w"])
    feats = build_model_features(**state)
    assert feats["safety_suji_t_3w"] == 1.0
    assert feats["safety_suji_t_9w"] == 1.0


def test_suji_does_not_cross_suits():
    """4w 出过 → 1w/7w safe，但 1t/7t 不应被标记。"""
    state = _make_minimal_state(opp_discards=["4w"])
    feats = build_model_features(**state)
    assert feats["safety_suji_t_1t"] == 0.0
    assert feats["safety_suji_t_7t"] == 0.0


def test_kabe_when_3_copies_visible():
    """5w 可见 3 张 → 4w/6w 标 kabe。"""
    # remaining=1 表示已可见 3 张
    rem = {code: 4 for code in PER_TILE_FEATURE_ORDER}
    rem["5w"] = 1  # 4 - 1 = 3 visible
    state = _make_minimal_state(remaining_counts=rem)
    feats = build_model_features(**state)
    assert feats["safety_kabe_t_4w"] == 1.0
    assert feats["safety_kabe_t_6w"] == 1.0


def test_kabe_edge_tile():
    """1w 可见 3 张 → 仅 2w 标 kabe（无 0w）。"""
    rem = {code: 4 for code in PER_TILE_FEATURE_ORDER}
    rem["1w"] = 1
    state = _make_minimal_state(remaining_counts=rem)
    feats = build_model_features(**state)
    assert feats["safety_kabe_t_2w"] == 1.0


def test_kabe_does_not_apply_to_honors():
    """字牌不参与 kabe 计算。"""
    state = _make_minimal_state()
    feats = build_model_features(**state)
    for honor in ("east", "south", "west", "north", "white", "green", "red"):
        assert feats[f"safety_kabe_t_{honor}"] == 0.0


# ---------- C++ 静态扫描（双侧名字契约） ----------

@pytest.mark.skipif(not CPP_FILE.exists(), reason="linhai_search_v3.cpp not present")
def test_cpp_emits_safety_suji_string():
    """静态扫描 C++ 源码确认 safety_suji_t_ 前缀被发出。"""
    content = CPP_FILE.read_text(encoding="utf-8")
    assert 'features[std::string("safety_suji_t_")' in content


@pytest.mark.skipif(not CPP_FILE.exists(), reason="linhai_search_v3.cpp not present")
def test_cpp_emits_safety_kabe_string():
    content = CPP_FILE.read_text(encoding="utf-8")
    assert 'features[std::string("safety_kabe_t_")' in content


@pytest.mark.skipif(not CPP_FILE.exists(), reason="linhai_search_v3.cpp not present")
def test_cpp_emits_safety_genbutsu_string():
    content = CPP_FILE.read_text(encoding="utf-8")
    # genbutsu 走 safety_t_ (没有 _genbutsu 后缀，与 Python 端一致)
    assert 'features[std::string("safety_t_")' in content
