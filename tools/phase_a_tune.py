#!/usr/bin/env python3
"""Phase A spec §3.5.4 — 网格搜索 EV 风险加权与 betaori 阈值。

输出：
- 每组参数对应的 N-seed × M 局验收指标（25/25 全正窗口数、最差窗口、放炮率）
- 最优参数候选写入 engine/params/v3/score_table_phase_a.json
- 全过程把原 score_table.json 备份到 score_table.json.bak，结束时恢复

Usage:
    # 冒烟（≈30s）
    python3 tools/phase_a_tune.py --grid small --games-per-seed 16 --seeds 1

    # small grid（约 1 小时）
    python3 tools/phase_a_tune.py --grid small

    # full grid（约 3 小时）
    python3 tools/phase_a_tune.py --grid full
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
SCORE_TABLE = REPO_ROOT / "engine/params/v3/score_table.json"
SCORE_TABLE_PHASE_A = REPO_ROOT / "engine/params/v3/score_table_phase_a.json"
ACCEPTANCE_REPORT = REPO_ROOT / "docs/release/acceptance_report.json"

GRIDS: Dict[str, Dict[str, List[float]]] = {
    "small": {
        "lambda_base": [1.0, 1.5, 2.0],
        "betaori_base": [0.13, 0.15],
        "qingyise_drop": [0.05],
    },
    "full": {
        "lambda_base": [1.0, 1.3, 1.6, 2.0],
        "mu_base": [1.5, 2.0, 3.0],
        "betaori_base": [0.12, 0.14, 0.16],
        "qingyise_drop": [0.04, 0.06],
    },
}


def write_score_table(params: dict) -> None:
    """把当前网格点参数写回 score_table.json（合并到已有结构，不破坏其他段）。"""
    with open(SCORE_TABLE) as f:
        cfg = json.load(f)
    cfg.setdefault("ev_risk_weights", {})
    cfg["ev_risk_weights"]["lambda_base"] = params["lambda_base"]
    if "mu_base" in params:
        cfg["ev_risk_weights"]["mu_base"] = params["mu_base"]
    cfg.setdefault("betaori_thresholds", {})
    cfg["betaori_thresholds"]["base"] = params["betaori_base"]
    cfg["betaori_thresholds"]["qingyise_drop"] = params["qingyise_drop"]
    with open(SCORE_TABLE, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def run_acceptance(games_per_seed: int, seeds: List[int]) -> dict:
    """跑 acceptance_25_8.py，返回汇总 dict。

    放宽门槛跑（仅为获取窗口数据用于排序），实际 Phase A 验证由 Task A7
    用严苛门槛重跑最优参数。
    """
    cmd = [
        sys.executable, str(REPO_ROOT / "tools/acceptance_25_8.py"),
        "--games-per-seed", str(games_per_seed),
        "--seeds", *map(str, seeds),
        "--min-window-chong", "-1000",
        "--min-avg-chong", "-1000",
        "--max-houjuu-rate", "1.0",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    with open(ACCEPTANCE_REPORT) as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", choices=["small", "full"], default="small")
    ap.add_argument("--games-per-seed", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--output", default=str(SCORE_TABLE_PHASE_A))
    ap.add_argument("--results-log", default=str(REPO_ROOT / "docs/release/phase_a_tune_results.json"),
                    help="所有网格点结果导出到这个 JSON 用于事后分析")
    args = ap.parse_args()

    grid = GRIDS[args.grid]
    keys = list(grid.keys())
    backup = SCORE_TABLE.with_suffix(".json.bak")
    shutil.copy(SCORE_TABLE, backup)
    print(f"Backed up score_table.json to {backup}")

    n_points = 1
    for v in grid.values():
        n_points *= len(v)
    print(f"Grid '{args.grid}' has {n_points} points; "
          f"{args.games_per_seed} games × {len(args.seeds)} seeds per point.\n")

    results = []
    try:
        for i, vals in enumerate(product(*grid.values()), 1):
            params = dict(zip(keys, vals))
            print(f"\n=== [{i}/{n_points}] Trying {params} ===")
            write_score_table(params)
            report = run_acceptance(args.games_per_seed, args.seeds)
            row = {
                "params": params,
                "all_positive_count": report["all_positive_window_count"],
                "total": report["total_window_count"],
                "min_window": report["min_window_chong"],
                "avg_chong": report["avg_chong_per_game"],
                "houjuu_rate": report["avg_houjuu_rate"],
            }
            results.append(row)
            print(f"  -> {row['all_positive_count']}/{row['total']} "
                  f"min={row['min_window']} avg={row['avg_chong']:.3f} "
                  f"houjuu={row['houjuu_rate']:.3f}")
    finally:
        # 即便中途异常也 dump 已得结果与恢复 score_table
        Path(args.results_log).parent.mkdir(parents=True, exist_ok=True)
        with open(args.results_log, "w") as f:
            json.dump({"grid": args.grid, "results": results}, f, indent=2, ensure_ascii=False)
        print(f"\nAll grid results logged to {args.results_log}")

    # 排序：先 all_positive_count 降序，再 min_window 降序，再 houjuu_rate 升序
    results.sort(key=lambda r: (-r["all_positive_count"], -r["min_window"], r["houjuu_rate"]))
    best = results[0] if results else None

    print("\n=== TOP 5 ===")
    for r in results[:5]:
        print(f"{r['params']} -> {r['all_positive_count']}/{r['total']} "
              f"min={r['min_window']} avg={r['avg_chong']:.3f} "
              f"houjuu={r['houjuu_rate']:.3f}")

    if best:
        write_score_table(best["params"])
        shutil.copy(SCORE_TABLE, args.output)
        print(f"\nBest params written to {args.output}")

    # 恢复原 score_table（用户决定是否替换为最优）
    shutil.copy(backup, SCORE_TABLE)
    print(f"Restored original score_table.json from {backup}")
    print(f"To apply best params: cp {args.output} {SCORE_TABLE}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
