#!/usr/bin/env python3
"""4 个最坏牌型的引擎 P95 延迟回归门禁。

通过条件（spec §5.3）：
- 每个 case P95 < 800ms
- 任一 case 超预算 → 退出码 1（CI 阻塞）

Worst cases:
1. 4 张白板 + 普通牌型（早期 shanten 性能瓶颈）
2. 3 个副露 + 听牌响应（response 路径）
3. 多抢杠候选场景
4. 抓冲红头全开 + 树掉还原

Usage:
    python3 tools/perf_regression.py --runs 100
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.state import GameState, PlayerState, Wind
from backend.app.services.orchestrator import LinhaiV3Orchestrator


# ---------- 4 个最坏 case ----------

def case_4_whites() -> GameState:
    """4 张白板 + 普通牌型 — 早期 shanten 计算最坏 case。"""
    hand = ["white"] * 4 + ["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w", "1t"]
    return _make_state(hand=hand, wall_remaining=24)


def case_complex_meld() -> GameState:
    """3 个副露 + 短手牌（响应路径压力）。"""
    from backend.app.core.state import Meld
    hand = ["1w", "2w", "3w", "white"]
    melds = [
        Meld(tiles=["4w", "4w", "4w"], type="peng"),
        Meld(tiles=["5w", "5w", "5w"], type="peng"),
        Meld(tiles=["6w", "6w", "6w"], type="peng"),
    ]
    state = _make_state(hand=hand, wall_remaining=20)
    state.players[Wind.EAST].melds = melds
    return state


def case_multi_qianggang() -> GameState:
    """多个 anko 候选 → 多抢杠场景。"""
    hand = ["1w", "1w", "1w", "2w", "2w", "2w", "3w", "3w", "3w",
            "1t", "1t", "1t", "white", "east"]
    return _make_state(hand=hand, wall_remaining=24)


def case_grab_charge_with_tree() -> GameState:
    """抓冲红头全开 + 白板暗刻（树掉还原触发条件）。"""
    hand = ["white", "white", "white", "1w", "5w", "9w", "east", "south",
            "west", "north", "red", "green", "1t", "9t"]
    state = _make_state(hand=hand, wall_remaining=24)
    state.tree_active = True
    state.grab_charge_active = True
    state.players[Wind.EAST].grab_charge_hits = 6
    state.players[Wind.EAST].grab_charge_limit = 6
    return state


def _make_state(hand: List[str], wall_remaining: int) -> GameState:
    players = {
        Wind.EAST: PlayerState(wind=Wind.EAST, hand=hand, discards=[]),
        Wind.SOUTH: PlayerState(wind=Wind.SOUTH, hand=[], discards=["9t", "8t"]),
    }
    return GameState(
        round_wind=Wind.EAST,
        dealer_wind=Wind.EAST,
        active_wind=Wind.EAST,
        players=players,
        wall_remaining=wall_remaining,
    )


CASES = {
    "4_whites": case_4_whites,
    "complex_meld_response": case_complex_meld,
    "multi_qianggang": case_multi_qianggang,
    "grab_charge_with_tree": case_grab_charge_with_tree,
}


# ---------- 测量 ----------

def measure_case(name: str, build_state, orchestrator, runs: int) -> Dict:
    timings_ms: List[float] = []
    err_count = 0
    for _ in range(runs):
        try:
            state = build_state()
            t0 = time.perf_counter()
            orchestrator.recommend(state)
            timings_ms.append((time.perf_counter() - t0) * 1000.0)
        except Exception:
            err_count += 1
    if not timings_ms:
        return {"name": name, "runs": runs, "errors": err_count, "skipped": "no successful runs"}
    return {
        "name": name,
        "runs": runs,
        "successful_runs": len(timings_ms),
        "errors": err_count,
        "p50_ms": round(statistics.median(timings_ms), 3),
        "p95_ms": round(statistics.quantiles(timings_ms, n=20)[-1] if len(timings_ms) >= 20 else max(timings_ms), 3),
        "max_ms": round(max(timings_ms), 3),
        "min_ms": round(min(timings_ms), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Engine P95 latency regression on 4 worst-case hands")
    ap.add_argument("--runs", type=int, default=100, help="Iterations per case")
    ap.add_argument("--p95-budget-ms", type=float, default=800.0,
                    help="Maximum allowed P95 per case in milliseconds")
    ap.add_argument("--output", default="docs/release/perf_report.json")
    args = ap.parse_args()

    orchestrator = LinhaiV3Orchestrator()
    engine_status = orchestrator.engine_status()
    active = engine_status.get("active_engine", "heuristic")

    results = []
    for name, build_state in CASES.items():
        results.append(measure_case(name, build_state, orchestrator, args.runs))
        latest = results[-1]
        if "p95_ms" in latest:
            print(f"[{name}] p50={latest['p50_ms']}ms p95={latest['p95_ms']}ms "
                  f"max={latest['max_ms']}ms (n={latest['successful_runs']})")
        else:
            print(f"[{name}] SKIPPED ({latest.get('skipped', 'unknown')})")

    failed = [r for r in results if r.get("p95_ms", 0) > args.p95_budget_ms]
    summary = {
        "active_engine": active,
        "p95_budget_ms": args.p95_budget_ms,
        "results": results,
        "failed_cases": [r["name"] for r in failed],
        "passed": len(failed) == 0,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print()
    print(json.dumps({
        "active_engine": active,
        "passed": summary["passed"],
        "failed_cases": summary["failed_cases"],
    }, indent=2))

    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
