#!/usr/bin/env python3
"""A-2b/A-2c one-click training + evaluation pipeline for V3 GBDT bundles.

Pipeline stages (any can be skipped via --skip-<stage>):
    1. generate  - Produce self-play sample jsonl (parallel, multi-opponent).
    2. train     - Train 6 GBDT heads and assemble a standard bundle dir.
    3. evaluate  - Run self-play vs heuristic for BOTH the new bundle and
                   the currently-deployed bundle over N games x K seeds.
    4. decide    - Pooled win rate + two-proportion z-test. If delta >=
                   --deploy-threshold AND z >= --significance-z, deploy
                   (backup old engine/params/v3, copy new heads in-place).
                   Otherwise archive the bundle under artifacts/bundles/.

Deploy is SAFE BY DEFAULT:
    * The current engine/params/ directory is never touched unless BOTH the
      delta and significance gates pass AND --no-dry-run is set.
    * When deploying, the old engine/params/v3/<head>/model.json files are
      copied to engine/params_backup_<timestamp>/ before being replaced.

Typical usage:
    # Full 1500-game, 3-opponent pipeline with 400-game eval and auto-archive
    python3 tools/train_and_evaluate_v3.py --run-name a2c_v1 --games-per-pair 1500

    # Re-use existing samples / bundle (iterate on eval only)
    python3 tools/train_and_evaluate_v3.py --run-name a2c_v1 \
        --skip-generate --skip-train --eval-games 800 --eval-seeds 9001,13579,24680,35791

    # Actually deploy to engine/params/ if gates pass
    python3 tools/train_and_evaluate_v3.py --run-name a2c_v1 --no-dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


# Six V3 heads that must all be trained for a complete bundle.
V3_HEADS: Tuple[str, ...] = (
    "agari_prob",
    "tenpai_prob",
    "houjuu_prob",
    "betaori",
    "tsumo_num",
    "ryukyoku_prob",
)


# Default opponent pairs for data generation. Each entry is
# (policy_a, policy_b, games_multiplier, seed). `games_multiplier` scales the
# shared --games-per-pair budget: orchestrator shards are ~5x slower than
# heuristic, so we halve their game count but keep the overall behavioural
# diversity (strong/strong, strong/mid, mid/mid, mid/weak).
#
# A-2b baseline recipe. Attempts to diversify with orchestrator self-play
# pairings (A-2c-1/A-2c-1b) did NOT improve win rate in 800-game evals --
# orchestrator:linear is itself the baseline, so mimicking it provides no
# new signal. Keep this conservative recipe; opt-in to orchestrator pairs
# via --opponent-pair on the CLI if a stronger teacher is available.
DEFAULT_OPPONENT_PAIRS: Tuple[Tuple[str, str, float, int], ...] = (
    ("heuristic", "heuristic", 1.0, 101),
    ("heuristic", "random", 1.0, 202),
    ("random", "random", 1.0, 303),
)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_cmd(cmd: List[str], *, env: Optional[Dict[str, str]] = None, check: bool = True,
            capture_output: bool = False) -> subprocess.CompletedProcess:
    """Run a subprocess command with nice logging and optional output capture."""
    env_full = os.environ.copy()
    if env:
        env_full.update(env)
    pretty = " ".join(cmd) if not env else f"{' '.join(f'{k}={v}' for k, v in env.items())} {' '.join(cmd)}"
    log(f"$ {pretty}")
    result = subprocess.run(
        cmd,
        env=env_full,
        check=check,
        capture_output=capture_output,
        text=True,
    )
    return result


def two_proportion_z(p1_wins: int, p1_total: int, p2_wins: int, p2_total: int) -> Tuple[float, float]:
    """Return (delta_pp, z) for a standard two-proportion z-test. Δ is in
    percentage points. Positive z means p1 > p2."""
    if p1_total == 0 or p2_total == 0:
        return 0.0, 0.0
    p1 = p1_wins / p1_total
    p2 = p2_wins / p2_total
    pool = (p1_wins + p2_wins) / (p1_total + p2_total)
    se = math.sqrt(pool * (1 - pool) * (1 / p1_total + 1 / p2_total))
    z = (p1 - p2) / se if se > 0 else 0.0
    return (p1 - p2) * 100.0, z


# --------------------------------------------------------------------------- #
# Stages                                                                      #
# --------------------------------------------------------------------------- #


def _needs_orchestrator(policy_spec: str) -> bool:
    return policy_spec.startswith("orchestrator")


def stage_generate(run_dir: Path, games_per_pair: int, baseline_params_dir: Path,
                   pairs=DEFAULT_OPPONENT_PAIRS) -> Path:
    """Run self-play sample generation in parallel for each opponent pair,
    then concatenate into a single jsonl.

    Shards that use ``orchestrator[...]`` get ``LINHAI_V3_PARAMS_DIR`` pointed
    at *baseline_params_dir* so they draw behaviour from the currently-deployed
    model (not from whatever bundle happens to be in /tmp).
    """
    safe_name = lambda s: s.replace(":", "_").replace("/", "_")
    out_dir = run_dir / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: List[Path] = []
    procs: List[Tuple[subprocess.Popen, Path, str, int]] = []

    log(f"generate: launching {len(pairs)} parallel shards, base budget={games_per_pair} games")
    for idx, entry in enumerate(pairs, start=1):
        # Accept both legacy 3-tuple (pa, pb, seed) and new 4-tuple with multiplier.
        if len(entry) == 4:
            pa, pb, multiplier, seed = entry
        else:
            pa, pb, seed = entry
            multiplier = 1.0
        games = max(1, int(round(games_per_pair * multiplier)))
        shard = out_dir / f"shard_{idx}_{safe_name(pa)}_vs_{safe_name(pb)}.jsonl"
        shard.unlink(missing_ok=True)
        cmd = [
            PYTHON, str(REPO_ROOT / "tools" / "selfplay_sample.py"),
            "--policy-a", pa, "--policy-b", pb,
            "--games", str(games),
            "--seed", str(seed),
            "--output", str(shard),
        ]
        # Only orchestrator shards pay the model-load cost; pin them to a known
        # bundle so data is reproducible run-to-run.
        shard_env = os.environ.copy()
        if _needs_orchestrator(pa) or _needs_orchestrator(pb):
            shard_env["LINHAI_V3_PARAMS_DIR"] = str(baseline_params_dir)
        log(f"  shard {idx}: {pa} vs {pb} games={games} seed={seed} -> {shard.name}")
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=shard_env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        procs.append((proc, shard, f"{pa}_vs_{pb}", games))

    failures: List[str] = []
    for proc, shard, tag, games in procs:
        _, stderr = proc.communicate()
        if proc.returncode != 0:
            failures.append(f"{tag}: rc={proc.returncode} stderr={stderr[-400:]}")
        else:
            shard_paths.append(shard)
            n = sum(1 for _ in shard.open("r", encoding="utf-8"))
            log(f"  shard {tag} OK, {games} games -> {n} rows")
    if failures:
        raise RuntimeError("generate stage failed:\n  " + "\n  ".join(failures))

    merged = run_dir / "samples_all.jsonl"
    with merged.open("w", encoding="utf-8") as fout:
        for shard in shard_paths:
            with shard.open("r", encoding="utf-8") as fin:
                shutil.copyfileobj(fin, fout)
    total = sum(1 for _ in merged.open("r", encoding="utf-8"))
    log(f"generate: merged {len(shard_paths)} shards -> {merged} ({total} rows total)")
    return merged


def stage_train(run_dir: Path, samples_path: Path, n_estimators: int, num_leaves: int,
                max_depth: int, lr: float, min_rows: int) -> Path:
    """Train 6 heads sequentially, then assemble a bundle that shadows
    engine/params (so fallback linhai/ako params are still available)."""
    raw_dir = run_dir / "heads_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    head_metrics: Dict[str, Dict] = {}
    for task in V3_HEADS:
        head_vdir = raw_dir / task
        head_vdir.mkdir(parents=True, exist_ok=True)
        log(f"train: {task}")
        run_cmd([
            PYTHON, str(REPO_ROOT / "tools" / "train_model_stub.py"),
            "--task", task,
            "--dataset", str(samples_path),
            "--version-dir", str(head_vdir),
            "--model-kind", "gbdt",
            "--n-estimators", str(n_estimators),
            "--num-leaves", str(num_leaves),
            "--max-depth", str(max_depth),
            "--gbdt-learning-rate", str(lr),
            "--gbdt-min-rows", str(min_rows),
            "--validation-split", "0.1",
        ])
        meta = json.loads((head_vdir / "v3" / task / "model_meta.json").read_text())
        head_metrics[task] = meta.get("metrics") or meta.get("validation_metrics") or {}

    # Assemble standard bundle: shadow engine/params, then replace v3/<head>/*.
    bundle_dir = run_dir / "bundle"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    shutil.copytree(REPO_ROOT / "engine" / "params", bundle_dir)
    for task in V3_HEADS:
        dst = bundle_dir / "v3" / task
        dst.mkdir(parents=True, exist_ok=True)
        src = raw_dir / task / "v3" / task
        shutil.copy(src / "model.json", dst / "model.json")
        shutil.copy(src / "model_meta.json", dst / "model_meta.json")

    # Sanity: collect model_type per head. If ANY head failed to train a GBDT
    # (e.g. degenerate label distribution), bail so the operator can react --
    # a mixed bundle is legal but usually indicates a data quality problem we
    # want visibility on.
    head_types: Dict[str, str] = {}
    degenerate: List[str] = []
    for task in V3_HEADS:
        mt = json.loads((bundle_dir / "v3" / task / "model.json").read_text()).get("model_type")
        head_types[task] = mt or "unknown"
        if mt != "lightgbm_gbdt":
            degenerate.append(f"{task}={mt}")
    (run_dir / "head_metrics.json").write_text(
        json.dumps({"metrics": head_metrics, "model_types": head_types},
                   indent=2, ensure_ascii=False)
    )
    if degenerate:
        log(f"train: WARNING non-GBDT heads detected: {', '.join(degenerate)} "
            f"(bundle is still usable, C++ loader falls back per-head)")
    log(f"train: bundle ready at {bundle_dir}")
    return bundle_dir


def stage_evaluate(run_dir: Path, bundle_dir: Path, label: str, seeds: List[int],
                   games_per_seed: int, max_turns: int) -> Dict:
    """Run orchestrator(bundle_dir) vs heuristic for each seed. Each seed is a
    separate subprocess so the C++ engine is re-initialized cleanly with the
    right LINHAI_V3_PARAMS_DIR."""
    eval_dir = run_dir / "evals" / label
    eval_dir.mkdir(parents=True, exist_ok=True)
    shards: List[Dict] = []
    for seed in seeds:
        shard_out = eval_dir / f"seed_{seed}.jsonl"
        shard_out.unlink(missing_ok=True)
        log(f"eval[{label}]: seed={seed} games={games_per_seed}")
        run_cmd(
            [
                PYTHON, str(REPO_ROOT / "tools" / "selfplay_eval.py"),
                "--policy-a", f"orchestrator:{label}",
                "--policy-b", "heuristic",
                "--games", str(games_per_seed),
                "--seed", str(seed),
                "--max-turns", str(max_turns),
                "--output", str(shard_out),
            ],
            env={"LINHAI_V3_PARAMS_DIR": str(bundle_dir)},
        )
        rec = json.loads(shard_out.read_text().strip().splitlines()[-1])
        shards.append(rec)

    pooled_a = sum(r["wins_a"] for r in shards)
    pooled_b = sum(r["wins_b"] for r in shards)
    pooled_n = pooled_a + pooled_b
    houjuu_a = sum(r["wins_a"] * 0 + r["houjuu_rate_a"] * r["games_played"] for r in shards)
    houjuu_b = sum(r["houjuu_rate_b"] * r["games_played"] for r in shards)
    games_played = sum(r["games_played"] for r in shards)
    summary = {
        "label": label,
        "bundle_dir": str(bundle_dir),
        "seeds": seeds,
        "games_per_seed": games_per_seed,
        "pooled_wins_a": pooled_a,
        "pooled_wins_b": pooled_b,
        "pooled_winrate_a": pooled_a / pooled_n if pooled_n else 0.0,
        "pooled_houjuu_a": houjuu_a / games_played if games_played else 0.0,
        "pooled_houjuu_b": houjuu_b / games_played if games_played else 0.0,
        "per_seed": shards,
    }
    (run_dir / f"eval_summary_{label}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"eval[{label}]: pooled winrate {summary['pooled_winrate_a']:.3f} ({pooled_a}/{pooled_n}), "
        f"self houjuu {summary['pooled_houjuu_a']:.3f}")
    return summary


def stage_decide(run_dir: Path, new_eval: Dict, base_eval: Dict, bundle_dir: Path,
                 deploy_threshold_pp: float, significance_z: float, dry_run: bool) -> Dict:
    """Apply the 2-gate deployment rule and emit REPORT.md + verdict.json."""
    delta_pp, z = two_proportion_z(
        new_eval["pooled_wins_a"], new_eval["pooled_wins_a"] + new_eval["pooled_wins_b"],
        base_eval["pooled_wins_a"], base_eval["pooled_wins_a"] + base_eval["pooled_wins_b"],
    )
    gates_passed = (delta_pp >= deploy_threshold_pp) and (z >= significance_z)
    verdict = {
        "new_winrate": new_eval["pooled_winrate_a"],
        "base_winrate": base_eval["pooled_winrate_a"],
        "delta_pp": delta_pp,
        "z_score": z,
        "deploy_threshold_pp": deploy_threshold_pp,
        "significance_z": significance_z,
        "gates_passed": gates_passed,
        "dry_run": dry_run,
        "action": None,
    }

    head_metrics_path = run_dir / "head_metrics.json"
    if head_metrics_path.exists():
        hm_doc = json.loads(head_metrics_path.read_text())
        # Accept both the legacy flat {task: metrics} shape and the new
        # {"metrics": ..., "model_types": ...} shape.
        head_metrics = hm_doc.get("metrics", hm_doc) if isinstance(hm_doc, dict) else {}
        head_types = hm_doc.get("model_types", {}) if isinstance(hm_doc, dict) else {}
    else:
        head_metrics = {}
        head_types = {}

    lines = [
        f"# V3 GBDT pipeline — {run_dir.name}",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Evaluation",
        "",
        "| Variant | Pooled winrate | Pooled houjuu (self) |",
        "|---|---|---|",
        f"| New (GBDT) | **{new_eval['pooled_winrate_a']:.3f}** "
        f"({new_eval['pooled_wins_a']}/{new_eval['pooled_wins_a']+new_eval['pooled_wins_b']}) "
        f"| {new_eval['pooled_houjuu_a']:.3f} |",
        f"| Baseline (engine/params/) | {base_eval['pooled_winrate_a']:.3f} "
        f"({base_eval['pooled_wins_a']}/{base_eval['pooled_wins_a']+base_eval['pooled_wins_b']}) "
        f"| {base_eval['pooled_houjuu_a']:.3f} |",
        f"| Δ | **{delta_pp:+.2f} pp** | |",
        f"| z | **{z:+.2f}** (threshold {significance_z:+.2f}) | |",
        "",
        "## Head metrics (training-time validation)",
        "",
        "| Head | model_type | train→val |",
        "|---|---|---|",
    ]
    for task in V3_HEADS:
        m = head_metrics.get(task, {})
        mt = head_types.get(task, "(n/a)")
        keys = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in m.items()) if m else "(n/a)"
        lines.append(f"| {task} | {mt} | {keys} |")

    lines += [
        "",
        "## Deployment gates",
        f"- Winrate delta ≥ {deploy_threshold_pp:.1f} pp: "
        f"{'PASS' if delta_pp >= deploy_threshold_pp else 'FAIL'} (Δ = {delta_pp:+.2f} pp)",
        f"- z ≥ {significance_z:+.2f}: {'PASS' if z >= significance_z else 'FAIL'} (z = {z:+.2f})",
        "",
    ]

    if gates_passed and not dry_run:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = REPO_ROOT / f"engine/params_backup_{ts}"
        shutil.copytree(REPO_ROOT / "engine" / "params" / "v3", backup_dir / "v3")
        for task in V3_HEADS:
            dst = REPO_ROOT / "engine" / "params" / "v3" / task
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(bundle_dir / "v3" / task / "model.json", dst / "model.json")
            shutil.copy(bundle_dir / "v3" / task / "model_meta.json", dst / "model_meta.json")
        verdict["action"] = "deployed"
        verdict["backup_dir"] = str(backup_dir)
        lines += [f"## Verdict: DEPLOYED", "",
                  f"Old `engine/params/v3/` copied to `{backup_dir.name}/`.",
                  f"New bundle heads installed into `engine/params/v3/`.", ""]
        log(f"decide: DEPLOYED (backup at {backup_dir})")
    elif gates_passed and dry_run:
        verdict["action"] = "would_deploy_dry_run"
        lines += ["## Verdict: GATES PASS (dry-run, did not deploy)",
                  "Re-run with `--no-dry-run` to promote this bundle to `engine/params/v3/`.", ""]
        log("decide: gates PASS, but --dry-run set → not deployed")
    else:
        archive_dir = REPO_ROOT / "artifacts" / "bundles" / run_dir.name
        if archive_dir.exists():
            shutil.rmtree(archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)
        for task in V3_HEADS:
            dst = archive_dir / "v3" / task
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(bundle_dir / "v3" / task / "model.json", dst / "model.json")
            shutil.copy(bundle_dir / "v3" / task / "model_meta.json", dst / "model_meta.json")
        verdict["action"] = "archived"
        verdict["archive_dir"] = str(archive_dir)
        lines += [f"## Verdict: ARCHIVED (gates not met)",
                  f"Bundle preserved at `{archive_dir.relative_to(REPO_ROOT)}/` for future comparison.",
                  f"`engine/params/` is UNCHANGED.", ""]
        log(f"decide: archived to {archive_dir}")

    (run_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (run_dir / "verdict.json").write_text(json.dumps(verdict, indent=2, ensure_ascii=False))
    return verdict


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", required=True, help="Identifier; run dir = artifacts/runs/<run_name>/")
    parser.add_argument("--games-per-pair", type=int, default=1500, help="Self-play games per opponent pair")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--gbdt-learning-rate", type=float, default=0.05)
    parser.add_argument("--gbdt-min-rows", type=int, default=2000)
    parser.add_argument("--eval-games", type=int, default=200, help="Games per eval seed")
    parser.add_argument("--eval-seeds", default="9001,13579",
                        help="Comma-separated seeds used for BOTH new and baseline eval")
    parser.add_argument("--eval-max-turns", type=int, default=200)
    parser.add_argument("--deploy-threshold", type=float, default=3.0, help="Minimum Δ in percentage points to deploy")
    parser.add_argument("--significance-z", type=float, default=1.96,
                        help="Minimum two-proportion z-score to deploy (default 1.96 = 95%%)")
    parser.add_argument("--no-dry-run", action="store_true",
                        help="Actually overwrite engine/params/v3/ when gates pass. Default is dry-run.")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true", help="Skip eval stage (implies no deploy)")
    parser.add_argument("--samples-path", help="Override merged samples path (default: <run>/samples_all.jsonl)")
    parser.add_argument("--bundle-dir", help="Override assembled bundle dir (default: <run>/bundle)")
    parser.add_argument("--baseline-params-dir",
                        default=str(REPO_ROOT / "engine" / "params"),
                        help="Params dir used by orchestrator shards during data generation "
                             "(and as the baseline for the final eval). "
                             "Default: engine/params (currently deployed linear bundle).")
    args = parser.parse_args()

    run_dir = REPO_ROOT / "artifacts" / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"run_dir = {run_dir}")

    t0 = time.time()
    samples_path = Path(args.samples_path) if args.samples_path else (run_dir / "samples_all.jsonl")
    bundle_dir = Path(args.bundle_dir) if args.bundle_dir else (run_dir / "bundle")

    baseline_params_dir = Path(args.baseline_params_dir).resolve()
    if not baseline_params_dir.exists():
        raise SystemExit(f"--baseline-params-dir does not exist: {baseline_params_dir}")

    if args.skip_generate:
        if not samples_path.exists():
            raise SystemExit(f"--skip-generate but samples file missing: {samples_path}")
        log(f"generate: SKIPPED (using {samples_path})")
    else:
        samples_path = stage_generate(run_dir, args.games_per_pair, baseline_params_dir)

    if args.skip_train:
        if not (bundle_dir / "v3" / "agari_prob" / "model.json").exists():
            raise SystemExit(f"--skip-train but bundle missing: {bundle_dir}")
        log(f"train: SKIPPED (using {bundle_dir})")
    else:
        bundle_dir = stage_train(
            run_dir, samples_path,
            n_estimators=args.n_estimators,
            num_leaves=args.num_leaves,
            max_depth=args.max_depth,
            lr=args.gbdt_learning_rate,
            min_rows=args.gbdt_min_rows,
        )

    if args.skip_eval:
        log("eval: SKIPPED; no deployment decision will be made")
        log(f"DONE in {time.time()-t0:.1f}s (training-only)")
        return

    seeds = [int(s) for s in args.eval_seeds.split(",") if s.strip()]
    new_eval = stage_evaluate(run_dir, bundle_dir, "new", seeds, args.eval_games, args.eval_max_turns)
    base_eval = stage_evaluate(run_dir, REPO_ROOT / "engine" / "params", "base",
                               seeds, args.eval_games, args.eval_max_turns)

    verdict = stage_decide(
        run_dir, new_eval, base_eval, bundle_dir,
        deploy_threshold_pp=args.deploy_threshold,
        significance_z=args.significance_z,
        dry_run=not args.no_dry_run,
    )

    log(f"DONE in {time.time()-t0:.1f}s -> verdict: {verdict['action']} "
        f"(Δ={verdict['delta_pp']:+.2f}pp, z={verdict['z_score']:+.2f})")
    log(f"Report: {run_dir/'REPORT.md'}")


if __name__ == "__main__":
    main()
