"""score_calculator.py 单元测试。

覆盖与 tests/cpp/test_linhai_score.cpp 等价的全部条款，保证 Python 与 C++
两端在同样输入下产出相同 score。spec §3 全条款。
"""
from __future__ import annotations

import pytest

from backend.app.services.score_calculator import (
    calc_chong,
    calc_score_from_game_event,
    load_score_table,
    reload_score_table,
)


# ---------- 基础 fixtures ----------

@pytest.fixture
def base_event():
    """A minimal hu event; tests override fields they care about."""
    return {
        "is_hu": True,
        "is_tsumo": False,
        "is_qiang_gang": False,
        "has_tree_active": False,
        "white_count_in_hand": 0,
        "white_count_in_melds": 0,
        "is_qingyise": False,
        "is_hunyise": False,
        "is_ziyise": False,
        "redhead_caught": [],
        "grab_charge_caught": 0,
        "contract_active": False,
        "contract_counter": 0,
        "contract_target": 3,
    }


# ---------- 普通胡 ----------

def test_ordinary_hu_with_white(base_event):
    base_event["white_count_in_hand"] = 1  # 非硬碰硬
    b = calc_chong(base_event)
    assert b["hu_type"] == "ordinary"
    assert b["base_chong"] == 1
    assert b["multiplier_2x"] == 0
    assert b["capped_base"] == 1
    assert b["final_chong"] == 1
    assert b["score"] == 1


def test_ordinary_hard(base_event):
    """无白板 → 硬碰硬 ×2。"""
    b = calc_chong(base_event)
    assert b["hu_type"] == "ordinary"
    assert b["multiplier_2x"] == 1
    assert b["capped_base"] == 2
    assert b["final_chong"] == 2


# ---------- 牌型 ----------

def test_qingyise(base_event):
    base_event["is_qingyise"] = True
    b = calc_chong(base_event)
    assert b["hu_type"] == "qingyise"
    assert b["base_chong"] == 8
    assert b["capped_base"] == 8  # 封顶 8


def test_ziyise(base_event):
    base_event["is_ziyise"] = True
    base_event["white_count_in_hand"] = 1  # 字一色可能有白板
    b = calc_chong(base_event)
    assert b["hu_type"] == "ziyise"
    assert b["base_chong"] == 8
    assert b["capped_base"] == 8


def test_hunyise_with_tree(base_event):
    base_event["is_hunyise"] = True
    base_event["white_count_in_hand"] = 1
    b = calc_chong(base_event)
    assert b["hu_type"] == "hunyise"
    assert b["base_chong"] == 2
    assert b["multiplier_2x"] == 0
    assert b["capped_base"] == 2


def test_hunyise_hard(base_event):
    base_event["is_hunyise"] = True
    base_event["white_count_in_hand"] = 0
    b = calc_chong(base_event)
    assert b["hu_type"] == "hunyise"
    assert b["base_chong"] == 4
    assert b["multiplier_2x"] == 0  # 混一色已经在 base 区分，避免重复翻倍


# ---------- 修饰条款 ----------

def test_qianggang(base_event):
    base_event["is_qiang_gang"] = True
    base_event["white_count_in_hand"] = 1  # 非硬碰硬
    b = calc_chong(base_event)
    assert b["qianggang_bonus"] == 2
    assert b["final_chong"] == 1 + 2


def test_tree_restore(base_event):
    base_event["has_tree_active"] = True
    base_event["white_count_in_hand"] = 3
    b = calc_chong(base_event)
    assert b["multiplier_4x"] == 1
    assert b["capped_base"] == 4  # base 1 * 4 = 4，封顶 8 内


def test_qingyise_with_grab_charge(base_event):
    base_event["is_qingyise"] = True
    base_event["grab_charge_caught"] = 2
    b = calc_chong(base_event)
    assert b["extra_chong"] == 2
    assert b["final_chong"] == 8 + 2  # capped 8 + extra 2，extra 不计入封顶


# ---------- 翻屁股 ----------

def test_redhead_yiwan(base_event):
    base_event["white_count_in_hand"] = 1
    base_event["redhead_caught"] = ["1m"]
    b = calc_chong(base_event)
    assert b["redhead_bonus"] == 1
    assert b["final_chong"] == 1 + 1


def test_redhead_jiuwan(base_event):
    base_event["white_count_in_hand"] = 1
    base_event["redhead_caught"] = ["9m"]
    b = calc_chong(base_event)
    assert b["redhead_bonus"] == 9


def test_redhead_zhong(base_event):
    """中（C） → +5 冲。"""
    base_event["white_count_in_hand"] = 1
    base_event["redhead_caught"] = ["C"]
    b = calc_chong(base_event)
    assert b["redhead_bonus"] == 5


def test_redhead_multi(base_event):
    base_event["white_count_in_hand"] = 1
    base_event["redhead_caught"] = ["1m", "9m", "E"]
    b = calc_chong(base_event)
    assert b["redhead_bonus"] == 1 + 9 + 5


def test_redhead_unknown_tile(base_event):
    """未知 tile（不在表里）→ 0 分，不抛异常。"""
    base_event["white_count_in_hand"] = 1
    base_event["redhead_caught"] = ["1p"]  # 筒不在翻屁股表里
    b = calc_chong(base_event)
    assert b["redhead_bonus"] == 0


# ---------- 三键承包 ----------

def test_contract_split_triggers(base_event):
    base_event["is_tsumo"] = True
    base_event["contract_active"] = True
    base_event["contract_counter"] = 3
    base_event["contract_target"] = 3
    b = calc_chong(base_event)
    # base 1, white=0 硬碰硬 → mult 2 → capped 2 → final 2 → split 1
    assert b["final_chong"] == 2
    assert b["contract_split"] == 1


def test_contract_inactive_below_target(base_event):
    base_event["is_tsumo"] = True
    base_event["contract_active"] = True
    base_event["contract_counter"] = 2
    base_event["contract_target"] = 3
    b = calc_chong(base_event)
    assert b["contract_split"] == 0


def test_contract_only_on_tsumo(base_event):
    """放炮胡时不触发承包。"""
    base_event["is_tsumo"] = False
    base_event["contract_active"] = True
    base_event["contract_counter"] = 3
    base_event["contract_target"] = 3
    b = calc_chong(base_event)
    assert b["contract_split"] == 0


# ---------- 组合场景 ----------

def test_combination_qingyise_qianggang(base_event):
    base_event["is_qingyise"] = True
    base_event["is_qiang_gang"] = True
    b = calc_chong(base_event)
    assert b["final_chong"] == 8 + 2  # 清一色封顶 + 抢杠


# ---------- 流局 ----------

def test_draw_returns_zero(base_event):
    base_event["is_hu"] = False
    b = calc_chong(base_event)
    assert b["score"] == 0
    assert b["hu_type"] == "draw"


# ---------- role-based scoring ----------

def test_winner_gets_positive_score(base_event):
    base_event["white_count_in_hand"] = 1
    s = calc_score_from_game_event(base_event, role="winner")
    assert s == 1


def test_loser_gets_negative_score(base_event):
    base_event["white_count_in_hand"] = 1
    s = calc_score_from_game_event(base_event, role="loser")
    assert s == -1


def test_observer_gets_zero(base_event):
    base_event["white_count_in_hand"] = 1
    s = calc_score_from_game_event(base_event, role="observer")
    assert s == 0


# ---------- score_table 加载 ----------

def test_load_default_table_succeeds():
    table = load_score_table()
    assert table["chong_to_score"] >= 1
    assert table["cap"]["max_base_chong"] == 8


def test_chong_to_score_ratio(base_event):
    """改 chong_to_score = 2 → score 翻倍。"""
    custom = load_score_table()
    custom["chong_to_score"] = 2
    base_event["white_count_in_hand"] = 1
    b = calc_chong(base_event, table=custom)
    assert b["score"] == b["final_chong"] * 2


def test_score_table_default_phase_a_fields_when_missing():
    """spec §3.5 — score_table.json 不含 ev_risk_weights 时回退到默认。"""
    from backend.app.services.score_calculator import load_score_table
    import tempfile, json, pathlib
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"chong_to_score": 1, "base_chong": {"ordinary": 1}}, f)
        path = f.name
    table = load_score_table(path)
    # Phase A 默认值（spec §3.5.1 / §3.5.2）
    assert table["ev_risk_weights"]["lambda_base"] == 1.5
    assert table["ev_risk_weights"]["mu_base"] == 2.0
    assert table["ev_risk_weights"]["lambda_pressure_step"] == 0.5
    assert table["betaori_thresholds"]["base"] == 0.15
    assert table["betaori_thresholds"]["meld_drop"] == 0.03
    assert table["betaori_thresholds"]["qingyise_drop"] == 0.05
    assert table["feature_weights"]["opp_meld_pressure_alpha"] == 1.0
    pathlib.Path(path).unlink()


def test_score_table_phase_a_fields_loaded_when_present():
    """spec §3.5 — 显式提供新字段时读取生效。"""
    from backend.app.services.score_calculator import load_score_table
    import tempfile, json, pathlib
    cfg = {
        "ev_risk_weights": {
            "lambda_base": 1.5, "mu_base": 2.5,
            "lambda_pressure_step": 0.4, "lambda_max": 3.0,
        },
        "betaori_thresholds": {
            "base": 0.13, "meld_drop": 0.025, "qingyise_drop": 0.04,
        },
        "feature_weights": {"opp_meld_pressure_alpha": 1.2},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        path = f.name
    table = load_score_table(path)
    assert table["ev_risk_weights"]["lambda_base"] == 1.5
    assert table["betaori_thresholds"]["base"] == 0.13
    assert table["feature_weights"]["opp_meld_pressure_alpha"] == 1.2
    pathlib.Path(path).unlink()


def test_phase_a_defaults_match_canonical():
    """spec §3.5 — 单一 truth source for Phase A 默认值。

    任何一处（C++ struct / Python setdefault / score_table.json）改动
    都必须同步另外两处，本测试是漂移护栏。
    """
    from backend.app.services.score_calculator import load_score_table
    import tempfile, json, pathlib
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({}, f)  # 完全空 JSON，全部回退默认
        path = f.name
    try:
        table = load_score_table(path)
        # ev_risk_weights — 8 个字段
        evr = table["ev_risk_weights"]
        assert evr["lambda_base"] == 1.5, "spec §3.5.1 λ default"
        assert evr["lambda_pressure_step"] == 0.5
        assert evr["lambda_max"] == 3.0
        assert evr["mu_base"] == 2.0, "spec §3.5.1 μ default"
        assert evr["mu_pressure_step"] == 0.5
        assert evr["mu_max"] == 5.0
        assert evr["houjuu_base_coeff"] == 5500.0
        assert evr["high_chong_threshold"] == 4.0
        # betaori_thresholds — 4 个字段
        bt = table["betaori_thresholds"]
        assert bt["base"] == 0.15, "spec §3.5.2 base default"
        assert bt["meld_drop"] == 0.03
        assert bt["qingyise_drop"] == 0.05
        assert bt["ev_loss_multiplier"] == 1.5
        # feature_weights — 2 个字段
        fw = table["feature_weights"]
        assert fw["opp_meld_pressure_alpha"] == 1.0
        assert fw["opp_qingyise_alarm_alpha"] == 1.0
    finally:
        pathlib.Path(path).unlink()
