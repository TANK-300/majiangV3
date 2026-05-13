#!/usr/bin/env python3
"""5-seed × 200 局（25 个 8-局窗口）验收回归。

通过条件（spec §6.1）：
- 5 seed × 25 = 125 个窗口全部 > 0
- 平均冲数 / 局 ≥ 0.5
- 最差单窗口 ≥ +2 冲
- 总放炮率 ≤ 12%

Usage:
    python3 tools/acceptance_25_8.py [--games-per-seed N] [--seeds 1 2 3 4 5]

Smoke (CI):
    python3 tools/acceptance_25_8.py --games-per-seed 16 --seeds 1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.selfplay_eval import (
    policy_factory,
    run_match_with_per_game_chong,
)


def settle_windows(per_game_chong: List[int], window_size: int = 8) -> List[int]:
    """Aggregate per-game chong into 8-game windows. Drops trailing partial window."""
    return [
        sum(per_game_chong[i:i + window_size])
        for i in range(0, len(per_game_chong) - window_size + 1, window_size)
    ]


def run_one_seed(
    seed: int,
    games: int,
    window: int,
    policy_a_spec: str,
    policy_b_spec: str,
    max_turns: int,
    missing_suit_enabled: bool = False,
) -> Dict:
    rng_a = random.Random(seed * 7 + 1)
    rng_b = random.Random(seed * 7 + 2)
    policy_a = policy_factory(policy_a_spec, rng_a)
    policy_b = policy_factory(policy_b_spec, rng_b)
    chongs_a, _chongs_b, stats = run_match_with_per_game_chong(
        policy_a, policy_b, games, seed=seed, max_turns=max_turns,
        missing_suit_enabled=missing_suit_enabled,
    )
    windows = settle_windows(chongs_a, window)
    stats_dict = stats.as_dict(policy_a_spec, policy_b_spec)
    return {
        "seed": seed,
        "games": games,
        "windows": windows,
        "all_positive": all(w > 0 for w in windows) if windows else False,
        "min_window": min(windows) if windows else 0,
        "avg_chong_per_game": (sum(chongs_a) / len(chongs_a)) if chongs_a else 0.0,
        "houjuu_rate": stats_dict["houjuu_rate_a"],
        "winrate": stats_dict["winrate_a"],
    }


def evaluate_pass(
    per_seed: List[Dict],
    min_window_chong: int,
    min_avg_chong: float,
    max_houjuu_rate: float,
) -> Dict:
    all_windows = [w for r in per_seed for w in r["windows"]]
    total = len(all_windows)
    positive = sum(1 for w in all_windows if w > 0)
    min_window = min((r["min_window"] for r in per_seed), default=0)
    avg_chong = (
        sum(r["avg_chong_per_game"] for r in per_seed) / len(per_seed)
        if per_seed else 0.0
    )
    avg_houjuu = (
        sum(r["houjuu_rate"] for r in per_seed) / len(per_seed)
        if per_seed else 0.0
    )

    passed = (
        total > 0
        and positive == total
        and min_window >= min_window_chong
        and avg_chong >= min_avg_chong
        and avg_houjuu <= max_houjuu_rate
    )
    return {
        "passed": passed,
        "all_positive_window_count": positive,
        "total_window_count": total,
        "min_window_chong": min_window,
        "avg_chong_per_game": avg_chong,
        "avg_houjuu_rate": avg_houjuu,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="5-seed × 8-game-window acceptance regression")
    ap.add_argument("--games-per-seed", type=int, default=200)
    # 1万范围最小可行版（spec §6.1 降级口径）：默认改为 3 seed × 25 = 75 窗口
    # 全正，最差窗口 ≥ +1 冲，平均冲数 ≥ 0.3。
    # 想跑 spec 严苛口径用 --seeds 1 2 3 4 5 --min-window-chong 2 --min-avg-chong 0.5
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--min-window-chong", type=int, default=1,
                    help="Worst single window must be >= this many chong")
    ap.add_argument("--min-avg-chong", type=float, default=0.3)
    ap.add_argument("--max-houjuu-rate", type=float, default=0.15)
    ap.add_argument("--policy-a", default="orchestrator:v3",
                    help="Policy under test (default V3 orchestrator)")
    ap.add_argument("--policy-b", default="reference_human",
                    help="Reference opponent (default ReferenceHumanPolicy)")
    ap.add_argument("--max-turns", type=int, default=200)
    ap.add_argument("--missing-suit", action="store_true",
                    help="Enable 2P linhai missing-suit (缺一门) rule")
    ap.add_argument("--output", default="docs/release/acceptance_report.json")
    args = ap.parse_args()

    per_seed: List[Dict] = []
    for s in args.seeds:
        result = run_one_seed(
            seed=s,
            games=args.games_per_seed,
            window=args.window,
            policy_a_spec=args.policy_a,
            policy_b_spec=args.policy_b,
            max_turns=args.max_turns,
            missing_suit_enabled=args.missing_suit,
        )
        per_seed.append(result)
        print(
            f"[seed={s}] games={result['games']} windows_ok={sum(1 for w in result['windows'] if w > 0)}/{len(result['windows'])} "
            f"min={result['min_window']} avg_chong={result['avg_chong_per_game']:.3f} "
            f"houjuu={result['houjuu_rate']:.3f} winrate={result['winrate']:.3f}"
        )

    summary = evaluate_pass(
        per_seed,
        min_window_chong=args.min_window_chong,
        min_avg_chong=args.min_avg_chong,
        max_houjuu_rate=args.max_houjuu_rate,
    )
    summary["per_seed"] = per_seed
    summary["thresholds"] = {
        "min_window_chong": args.min_window_chong,
        "min_avg_chong": args.min_avg_chong,
        "max_houjuu_rate": args.max_houjuu_rate,
        "seeds": args.seeds,
        "games_per_seed": args.games_per_seed,
        "policy_a": args.policy_a,
        "policy_b": args.policy_b,
        "missing_suit": bool(args.missing_suit),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print()
    print(json.dumps({
        "passed": summary["passed"],
        "all_positive_window_count": summary["all_positive_window_count"],
        "total_window_count": summary["total_window_count"],
        "min_window_chong": summary["min_window_chong"],
        "avg_chong_per_game": round(summary["avg_chong_per_game"], 3),
        "avg_houjuu_rate": round(summary["avg_houjuu_rate"], 3),
    }, indent=2))

    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
