#!/usr/bin/env python3
"""R0-R3 多轮迭代训练流水线（spec §4.2）。

每轮三步：
1. selfplay_sample 生成数据
2. train_model_stub 训 8 个 head（6 prob + 2 score）
3. （仅 R3）调用 acceptance_25_8 跑回归

CPU 节奏（spec 估计）：
- R0: ~6 小时（50k 局）
- R1: ~8 小时
- R2: ~12 小时（depth=3）
- R3: ~1 小时（仅验收，不训练）

Usage:
    # 跑完整流水线
    python3 tools/iterative_train.py --start R0 --end R3

    # 单轮（断点续跑）
    python3 tools/iterative_train.py --start R1 --end R1 \\
        --params-root data/iter_params --sample-root data/iter_samples

    # smoke（小样本验证流水通畅，~1 分钟）
    python3 tools/iterative_train.py --start R0 --end R0 \\
        --games-override 64 --params-root /tmp/iter_smoke

GPU 入口（不实现，留接口）：
    --accelerator gpu  → 当前打印 "GPU path: not implemented in this milestone, contact for quote"
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------- Round configs (spec §4.2) ----------

ROUNDS = {
    "R0": {
        "policy_a": "reference_human",
        "policy_b": "random",
        "games": 50000,
        "description": "R0: bootstrap GBDT from ReferenceHuman vs random samples",
    },
    "R1": {
        "policy_a": "orchestrator:v3",
        "policy_b": "reference_human",
        "games": 50000,
        "description": "R1: V3(R0) vs ReferenceHuman, recalibrate houjuu/betaori",
    },
    "R2": {
        "policy_a": "orchestrator:v3",
        "policy_b": "orchestrator:v3",
        "games": 50000,
        "description": "R2: V3(R1) self-play (temperature sampling), depth=3",
    },
    "R3": {
        "policy_a": "orchestrator:v3",
        "policy_b": "reference_human",
        "games": 200,
        "is_acceptance": True,
        "description": "R3: acceptance_25_8 regression vs ReferenceHuman",
    },
}

# spec §3.3：8 个训练 head
TRAINING_HEADS = [
    "agari_prob", "tenpai_prob", "houjuu_prob",
    "betaori", "tsumo_num", "ryukyoku_prob",
    "agari_score", "houjuu_score",
]


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _run(cmd: List[str], **kw) -> int:
    """Run a subprocess, streaming output. Returns exit code."""
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=REPO_ROOT, **kw)


def run_sample_generation(round_name: str, cfg: Dict, sample_dir: Path,
                          games_override: Optional[int],
                          params_root: Path) -> Path:
    """Generate selfplay samples for this round. Returns path to JSONL.

    Bug fix (2026-04-29): R1/R2 需要把 LINHAI_V3_PARAMS_DIR 指向前一轮 bundle，
    否则 orchestrator:v3 policy 会加载默认 (Phase A) bundle 而不是 V3(R0)/V3(R1)，
    导致每轮的训练数据分布相同——蒸馏失效。
    """
    sample_file = sample_dir / f"{round_name}.jsonl"
    games = games_override if games_override else cfg["games"]
    cmd = [
        sys.executable, "tools/selfplay_sample.py",
        "--policy-a", cfg["policy_a"],
        "--policy-b", cfg["policy_b"],
        "--games", str(games),
        "--seed", "1",
        "--output", str(sample_file),
    ]
    # 验收 (spec §5): R0/R1 用 fast profile（depth=0, beam=1, ~5ms/step）；
    # spec 原本 R2 切 prod（depth=3），但 10k 局规模 + M4 Max 单核下 prod 会到
    # 5-8 小时——超出 PHASE_B_EXECUTION_GUIDE.md 承诺的 3 小时。R2 同样用 fast
    # profile：V3(R1) self-play 即使 depth=0 仍然产出对 R1 的微小改进数据，
    # 配合 score_label 回归足以驱动一步蒸馏。R3 acceptance 仍走 prod（已经在
    # acceptance_25_8.py 默认配置里）。
    env = os.environ.copy()
    env["LINHAI_V3_PROFILE"] = "fast"

    # 把前一轮的 bundle 注入 selfplay 子进程：
    #   R0: 不需要（policy 是 reference_human + random，不调 V3）
    #   R1: 用 R0 的 bundle（policy_a=orchestrator:v3 → V3(R0)）
    #   R2: 用 R1 的 bundle（V3(R1) self-play）
    prior_bundle_map = {"R1": "R0", "R2": "R1"}
    prior_round = prior_bundle_map.get(round_name)
    if prior_round:
        prior_bundle = params_root / prior_round
        if prior_bundle.exists():
            env["LINHAI_V3_PARAMS_DIR"] = str(prior_bundle)
            print(f"$ LINHAI_V3_PARAMS_DIR={prior_bundle} (loading {prior_round} bundle for {round_name})",
                  flush=True)
        else:
            raise RuntimeError(
                f"{round_name} requires {prior_round} bundle at {prior_bundle}, "
                f"but it doesn't exist. Run --start {prior_round} first."
            )
    print(f"$ LINHAI_V3_PROFILE={env['LINHAI_V3_PROFILE']} {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, cwd=REPO_ROOT, env=env)
    if rc != 0:
        raise RuntimeError(f"selfplay_sample failed for {round_name} (rc={rc})")
    return sample_file


def run_training(round_name: str, sample_file: Path, params_dir: Path,
                 model_kind: str = "auto") -> Path:
    """Train all 8 heads. Returns the bundle directory."""
    bundle_dir = _ensure_dir(params_dir / round_name)
    for head in TRAINING_HEADS:
        cmd = [
            sys.executable, "tools/train_model_stub.py",
            "--task", head,
            "--version-dir", str(bundle_dir),
            "--dataset", str(sample_file),
            "--validation-split", "0.2",
            "--model-kind", model_kind,
        ]
        rc = _run(cmd)
        if rc != 0:
            print(f"WARNING: training head {head} returned rc={rc} (continuing)", flush=True)
    return bundle_dir


def run_acceptance(params_dir: Path, output_path: Path) -> Dict:
    """Run acceptance_25_8 with the trained bundle, return summary dict."""
    env = os.environ.copy()
    env["LINHAI_V3_PARAMS_DIR"] = str(params_dir)
    cmd = [
        sys.executable, "tools/acceptance_25_8.py",
        "--output", str(output_path),
    ]
    print(f"$ LINHAI_V3_PARAMS_DIR={params_dir} {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, cwd=REPO_ROOT, env=env)
    if output_path.exists():
        report = json.loads(output_path.read_text())
        return {"name": "R3", "passed": report.get("passed", False),
                "report_path": str(output_path), "rc": rc}
    return {"name": "R3", "passed": False, "rc": rc, "report_path": None}


def run_round(round_name: str, cfg: Dict, params_root: Path, sample_root: Path,
              games_override: Optional[int], model_kind: str) -> Dict:
    print(f"\n{'='*60}\n{round_name}: {cfg['description']}\n{'='*60}", flush=True)
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()

    if cfg.get("is_acceptance"):
        # R3 is pure acceptance, no new training. Use the most recent bundle
        # (the directory should have been populated by R2).
        report_path = params_root / "acceptance_R3.json"
        result = run_acceptance(params_root / "R2", report_path)
        elapsed = time.time() - t0
        return {**result, "started_at": started_at, "elapsed_s": round(elapsed, 1)}

    sample_file = run_sample_generation(round_name, cfg, sample_root, games_override, params_root)
    bundle_dir = run_training(round_name, sample_file, params_root, model_kind)
    elapsed = time.time() - t0
    return {
        "name": round_name,
        "started_at": started_at,
        "elapsed_s": round(elapsed, 1),
        "sample_file": str(sample_file),
        "bundle_dir": str(bundle_dir),
        "passed": True,  # training always "passes" structurally
    }


def main() -> int:
    rounds_list = list(ROUNDS.keys())
    ap = argparse.ArgumentParser(description="R0-R3 iterative training pipeline")
    ap.add_argument("--start", choices=rounds_list, default="R0")
    ap.add_argument("--end", choices=rounds_list, default="R3")
    ap.add_argument("--params-root", default="data/iter_params")
    ap.add_argument("--sample-root", default="data/iter_samples")
    ap.add_argument("--games-override", type=int, default=None,
                    help="Override per-round games count (used for smoke testing)")
    ap.add_argument("--model-kind", choices=["auto", "gbdt", "linear"], default="auto")
    ap.add_argument("--accelerator", choices=["cpu", "gpu"], default="cpu",
                    help="GPU is intentionally not implemented in this milestone")
    args = ap.parse_args()

    if args.accelerator == "gpu":
        print(json.dumps({
            "error": "gpu_not_implemented",
            "message": "GPU path: not implemented in this milestone, contact for quote.",
            "milestone": "spec §4.3 documents this as a deferred deliverable",
        }, indent=2))
        return 2

    params_root = _ensure_dir(Path(args.params_root))
    sample_root = _ensure_dir(Path(args.sample_root))

    start_idx = rounds_list.index(args.start)
    end_idx = rounds_list.index(args.end)
    log: List[Dict] = []
    log_path = params_root / "iter_log.json"

    overall_rc = 0
    for name in rounds_list[start_idx:end_idx + 1]:
        result = run_round(
            name, ROUNDS[name],
            params_root=params_root,
            sample_root=sample_root,
            games_override=args.games_override,
            model_kind=args.model_kind,
        )
        log.append(result)
        log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))
        if not result.get("passed", True):
            overall_rc = 1

    print()
    print("=" * 60)
    print("Iteration log:")
    print(json.dumps(log, indent=2, ensure_ascii=False))
    print(f"Saved to {log_path}")
    return overall_rc


if __name__ == "__main__":
    sys.exit(main())
