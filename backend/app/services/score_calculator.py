"""临海麻将番数 → 冲数 → 积分计算（Python 端）。

设计目标：
1. 与 C++ engine/share/linhai_score.cpp 行为完全一致（pure Python fallback）
2. 不依赖 .so，selfplay / acceptance / sample 脚本在没构建 C++ 的环境也能跑
3. 读取 engine/params/v3/score_table.json，所有番数映射可参数化

调用方式：
    from backend.app.services.score_calculator import calc_score_from_game_event
    score = calc_score_from_game_event(event_dict, role="winner")

`event_dict` 字段（spec §3.3）：
    is_hu                 bool   是否胡牌（流局 False）
    is_tsumo              bool   是否自摸
    is_qiang_gang         bool   是否抢杠胡
    has_tree_active       bool   树活
    white_count_in_hand   int
    white_count_in_melds  int
    is_qingyise           bool
    is_hunyise            bool
    is_ziyise             bool
    redhead_caught        list[str]  抓到的红头牌（mjai 名，如 ["1m", "9m", "C"]）
    grab_charge_caught    int   本局抓冲牌数
    contract_active       bool
    contract_counter      int
    contract_target       int
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

# ---------- 默认番数表（与 engine/params/v3/score_table.json 一致） ----------

_DEFAULT_TABLE = {
    "chong_to_score": 1,
    "base_chong": {
        "ordinary": 1,
        "qingyise": 8,
        "ziyise": 8,
        "hunyise_with_tree": 2,
        "hunyise_hard": 4,
    },
    "multipliers": {
        "hard_collision": 2,
        "tree_restore_white_anko": 4,
    },
    "bonuses": {
        "qianggang_hu": 2,
        "grab_charge_per_tile": 1,
    },
    "redhead": {
        "tile_to_chong": {
            "1m": 1, "2m": 2, "3m": 3, "4m": 4, "5m": 5, "6m": 6, "7m": 7, "8m": 8, "9m": 9,
            "E": 5, "S": 5, "W": 5, "N": 5,
            "P": 5, "F": 5, "C": 5,
        }
    },
    "contract": {"split_ratio": [0.5, 0.5]},
    "cap": {"max_base_chong": 8, "extra_chong_uncapped": True},
}

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCORE_TABLE_PATH = REPO_ROOT / "engine" / "params" / "v3" / "score_table.json"

_CACHED_TABLE: Optional[Dict] = None


def _deep_update(base: Dict, override: Dict) -> Dict:
    """Recursive merge — used to patch defaults with values from JSON file."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_score_table(path: Optional[Path] = None) -> Dict:
    """Load score_table.json from disk; fall back to defaults if missing/malformed."""
    p = Path(path) if path else DEFAULT_SCORE_TABLE_PATH
    table = dict(_DEFAULT_TABLE)
    if p.exists():
        try:
            override = json.loads(p.read_text(encoding="utf-8"))
            table = _deep_update(table, override)
        except (json.JSONDecodeError, OSError):
            pass

    # === Phase A 默认字段填充（spec §3.5）===
    table.setdefault("ev_risk_weights", {})
    evr = table["ev_risk_weights"]
    evr.setdefault("lambda_base", 1.5)
    evr.setdefault("lambda_pressure_step", 0.5)
    evr.setdefault("lambda_max", 3.0)
    evr.setdefault("mu_base", 2.0)
    evr.setdefault("mu_pressure_step", 0.5)
    evr.setdefault("mu_max", 5.0)
    evr.setdefault("houjuu_base_coeff", 5500.0)
    evr.setdefault("high_chong_threshold", 4.0)
    table.setdefault("betaori_thresholds", {})
    bt = table["betaori_thresholds"]
    bt.setdefault("base", 0.15)
    bt.setdefault("meld_drop", 0.03)
    bt.setdefault("qingyise_drop", 0.05)
    bt.setdefault("ev_loss_multiplier", 1.5)
    table.setdefault("feature_weights", {})
    fw = table["feature_weights"]
    fw.setdefault("opp_meld_pressure_alpha", 1.0)
    fw.setdefault("opp_qingyise_alarm_alpha", 1.0)

    return table


def _get_table() -> Dict:
    global _CACHED_TABLE
    if _CACHED_TABLE is None:
        _CACHED_TABLE = load_score_table()
    return _CACHED_TABLE


def reload_score_table(path: Optional[Path] = None) -> None:
    """Force re-read score_table.json (for tests / dev)."""
    global _CACHED_TABLE
    _CACHED_TABLE = load_score_table(path)


# ---------- 番数计算核心 ----------

def _classify_hu(event: Dict) -> str:
    if event.get("is_qingyise"):
        return "qingyise"
    if event.get("is_ziyise"):
        return "ziyise"
    if event.get("is_hunyise"):
        return "hunyise"
    return "ordinary"


def calc_chong(event: Dict, table: Optional[Dict] = None) -> Dict[str, int]:
    """Return a breakdown dict mirroring C++ ChongBreakdown.

    Keys: hu_type, base_chong, multiplier_2x, multiplier_4x, capped_base,
    extra_chong, qianggang_bonus, redhead_bonus, contract_split, final_chong, score.
    """
    if not event.get("is_hu", True):
        return {
            "hu_type": "draw",
            "base_chong": 0, "multiplier_2x": 0, "multiplier_4x": 0,
            "capped_base": 0, "extra_chong": 0, "qianggang_bonus": 0,
            "redhead_bonus": 0, "contract_split": 0,
            "final_chong": 0, "score": 0,
        }

    t = table or _get_table()
    base = t["base_chong"]
    cap = int(t["cap"]["max_base_chong"])
    qianggang = int(t["bonuses"]["qianggang_hu"])
    grab_per_tile = int(t["bonuses"].get("grab_charge_per_tile", 1))
    chong_to_score = int(t["chong_to_score"])
    redhead_table = t["redhead"]["tile_to_chong"]

    hu_type = _classify_hu(event)
    white_in_hand = int(event.get("white_count_in_hand", 0))
    white_in_melds = int(event.get("white_count_in_melds", 0))
    white_total = white_in_hand + white_in_melds

    # 1. base
    if hu_type == "qingyise":
        base_chong = int(base["qingyise"])
    elif hu_type == "ziyise":
        base_chong = int(base["ziyise"])
    elif hu_type == "hunyise":
        base_chong = int(base["hunyise_with_tree"]) if white_total > 0 else int(base["hunyise_hard"])
    else:
        base_chong = int(base["ordinary"])

    # 2. multipliers
    multiplier_2x = 0
    if white_total == 0 and hu_type != "hunyise":
        multiplier_2x = 1  # 硬碰硬

    multiplier_4x = 0
    if event.get("has_tree_active") and white_in_hand >= 3:
        multiplier_4x = 1  # 树掉还原（白板暗刻）

    mult = (2 ** multiplier_2x) * (4 if multiplier_4x else 1)
    capped_base = min(base_chong * mult, cap)

    # 3. bonuses
    qianggang_bonus = qianggang if event.get("is_qiang_gang") else 0
    extra_chong = int(event.get("grab_charge_caught", 0)) * grab_per_tile
    redhead_bonus = sum(int(redhead_table.get(tile, 0)) for tile in event.get("redhead_caught", []))

    # 4. final
    final_chong = capped_base + extra_chong + qianggang_bonus + redhead_bonus

    # 5. contract split
    contract_split = 0
    if (event.get("is_tsumo")
            and event.get("contract_active")
            and int(event.get("contract_counter", 0)) >= int(event.get("contract_target", 3))):
        contract_split = final_chong // 2

    return {
        "hu_type": hu_type,
        "base_chong": base_chong,
        "multiplier_2x": multiplier_2x,
        "multiplier_4x": multiplier_4x,
        "capped_base": capped_base,
        "extra_chong": extra_chong,
        "qianggang_bonus": qianggang_bonus,
        "redhead_bonus": redhead_bonus,
        "contract_split": contract_split,
        "final_chong": final_chong,
        "score": final_chong * chong_to_score,
    }


def calc_score_from_game_event(event: Dict, role: str = "winner",
                               table: Optional[Dict] = None) -> int:
    """Returns signed score from `role`'s perspective.

    role:
        "winner"   — 胡牌方，正分
        "loser"    — 放炮方（仅非自摸时），负分
        "observer" — 旁观者（含自摸时其他玩家），返回 0
    """
    breakdown = calc_chong(event, table)
    s = breakdown["score"]
    if role == "winner":
        return s
    if role == "loser":
        return -s
    return 0
