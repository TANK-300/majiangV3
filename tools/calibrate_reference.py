#!/usr/bin/env python3
"""ReferenceHumanPolicy 强度校准（spec §2.2）。

通过条件：reference_human vs heuristic 200 局，胜率 ≥ 70% 且放炮率 ≤ 12%。
不达标说明代理对手不够"中等熟练真人"，离线 25/25 的真人安全余量站不住，
应回到 backend/app/services/reference_human.py 调超参重跑。

Usage:
    python3 tools/calibrate_reference.py
    python3 tools/calibrate_reference.py --games 200 --seed 1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.selfplay_eval import policy_factory, run_match_with_per_game_chong


def evaluate_pass(
    stats_dict: Dict,
    min_winrate: float,
    max_houjuu_rate: float,
) -> Dict:
    """Apply spec §2.2 calibration thresholds to a MatchStats.as_dict() output."""
    winrate = float(stats_dict["winrate_a"])
    houjuu = float(stats_dict["houjuu_rate_a"])
    passed = winrate >= min_winrate and houjuu <= max_houjuu_rate
    return {
        "passed": passed,
        "winrate": winrate,
        "houjuu_rate": houjuu,
        "thresholds": {
            "min_winrate": min_winrate,
            "max_houjuu_rate": max_houjuu_rate,
        },
    }


def run_calibration(
    seed: int,
    games: int,
    max_turns: int,
    policy_a_spec: str,
    policy_b_spec: str,
) -> Tuple[Dict, List[int]]:
    rng_a = random.Random(seed * 7 + 1)
    rng_b = random.Random(seed * 7 + 2)
    policy_a = policy_factory(policy_a_spec, rng_a)
    policy_b = policy_factory(policy_b_spec, rng_b)
    chongs_a, _chongs_b, stats = run_match_with_per_game_chong(
        policy_a, policy_b, games, seed=seed, max_turns=max_turns,
    )
    return stats.as_dict(policy_a_spec, policy_b_spec), chongs_a


def main() -> int:
    ap = argparse.ArgumentParser(
        description="ReferenceHumanPolicy strength calibration vs heuristic (spec §2.2)"
    )
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--policy-a", default="reference_human",
                    help="Policy under calibration (default reference_human)")
    ap.add_argument("--policy-b", default="heuristic",
                    help="Baseline opponent (default heuristic)")
    ap.add_argument("--min-winrate", type=float, default=0.70)
    ap.add_argument("--max-houjuu-rate", type=float, default=0.12)
    ap.add_argument("--max-turns", type=int, default=200)
    ap.add_argument("--output", default="docs/release/reference_calibration.json")
    args = ap.parse_args()

    stats, chongs_a = run_calibration(
        seed=args.seed,
        games=args.games,
        max_turns=args.max_turns,
        policy_a_spec=args.policy_a,
        policy_b_spec=args.policy_b,
    )
    summary = evaluate_pass(stats, args.min_winrate, args.max_houjuu_rate)
    summary["stats"] = stats
    summary["avg_chong_per_game"] = (
        sum(chongs_a) / len(chongs_a) if chongs_a else 0.0
    )
    summary["games"] = args.games
    summary["seed"] = args.seed

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(json.dumps({
        "passed": summary["passed"],
        "winrate": round(summary["winrate"], 3),
        "houjuu_rate": round(summary["houjuu_rate"], 3),
        "avg_chong_per_game": round(summary["avg_chong_per_game"], 3),
        "policy_a": args.policy_a,
        "policy_b": args.policy_b,
        "games": args.games,
        "thresholds": summary["thresholds"],
    }, indent=2, ensure_ascii=False))

    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())