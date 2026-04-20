from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from tools.selfplay_eval import HeuristicPolicy, RandomPolicy, run_match  # noqa: E402
from tools.selfplay_sample import run_sample_generation  # noqa: E402
from tools.train_neural import train_neural_bundle  # noqa: E402


def _make_samples(tmp_path: Path, games: int = 40) -> Path:
    import random

    samples_path = tmp_path / "samples.jsonl"
    run_sample_generation(
        HeuristicPolicy(),
        RandomPolicy(random.Random(3)),
        games=games,
        seed=3,
        swap_sides=True,
        max_turns=80,
        output_path=samples_path,
    )
    return samples_path


def test_train_neural_bundle_writes_onnx_and_metadata(tmp_path: Path) -> None:
    samples = _make_samples(tmp_path, games=30)
    out_dir = tmp_path / "bundle"
    bundle = train_neural_bundle(
        dataset_path=samples,
        output_dir=out_dir,
        hidden=16,
        epochs=3,
        batch_size=64,
        lr=1e-2,
        val_split=0.2,
        seed=1,
        export_onnx=True,
    )
    assert (out_dir / "model.pt").is_file()
    assert (out_dir / "model.onnx").is_file()
    meta = json.loads((out_dir / "model_meta.json").read_text(encoding="utf-8"))
    assert meta["model_type"] == "mlp_multihead"
    assert set(meta["heads"]) == {"agari", "houjuu", "tsumo_num"}
    assert meta["feature_names"] and meta["feature_names"] == bundle.feature_names
    for name in meta["feature_names"]:
        assert name in meta["feature_means"]
        assert name in meta["feature_stds"]
    assert "train_agari_acc" in meta["metrics"]


def test_neural_policy_loads_bundle_and_plays(tmp_path: Path) -> None:
    # End-to-end: sample -> train -> export ONNX -> load as NeuralPolicy ->
    # play some games. The goal is to verify the pipeline doesn't crash; we
    # deliberately don't assert a win rate (tiny training set can't beat the
    # heuristic reliably, and that's fine -- the point is the plumbing).
    samples = _make_samples(tmp_path, games=40)
    bundle_dir = tmp_path / "bundle"
    train_neural_bundle(
        dataset_path=samples,
        output_dir=bundle_dir,
        hidden=16,
        epochs=3,
        batch_size=64,
        lr=1e-2,
        val_split=0.2,
        seed=0,
        export_onnx=True,
    )

    from tools.selfplay_eval import NeuralPolicy

    neural = NeuralPolicy(bundle_dir, name="neural_test")
    stats = run_match(
        neural,
        HeuristicPolicy(),
        games=4,
        seed=0,
        swap_sides=True,
        max_turns=80,
    )
    assert stats.games_played == 4
    summary = stats.as_dict("neural", "heuristic")
    assert 0.0 <= summary["winrate_a"] <= 1.0
    assert 0.0 <= summary["winrate_b"] <= 1.0


def test_policy_factory_accepts_neural_spec(tmp_path: Path) -> None:
    samples = _make_samples(tmp_path, games=15)
    bundle_dir = tmp_path / "bundle"
    train_neural_bundle(
        dataset_path=samples,
        output_dir=bundle_dir,
        hidden=16,
        epochs=2,
        batch_size=64,
        lr=1e-2,
        val_split=0.0,
        seed=0,
        export_onnx=True,
    )

    from tools.selfplay_eval import NeuralPolicy, policy_factory

    import random
    rng = random.Random(0)
    policy = policy_factory(f"neural:{bundle_dir}", rng)
    assert isinstance(policy, NeuralPolicy)


def test_cli_train_neural_then_eval_neural_vs_heuristic(tmp_path: Path) -> None:
    # Ensure the two tools wire up cleanly at the CLI level. This mirrors
    # what an operator would run on a prod box: generate samples, train, eval.
    samples = tmp_path / "samples.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(TOOLS_DIR / "selfplay_sample.py"),
            "--policy-a", "heuristic",
            "--policy-b", "random",
            "--games", "20",
            "--seed", "11",
            "--max-turns", "80",
            "--output", str(samples),
        ],
        check=True,
    )
    bundle_dir = tmp_path / "bundle"
    subprocess.run(
        [
            sys.executable,
            str(TOOLS_DIR / "train_neural.py"),
            "--dataset", str(samples),
            "--output-dir", str(bundle_dir),
            "--hidden", "16",
            "--epochs", "3",
            "--seed", "0",
        ],
        check=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            str(TOOLS_DIR / "selfplay_eval.py"),
            "--policy-a", f"neural:{bundle_dir}",
            "--policy-b", "heuristic",
            "--games", "4",
            "--seed", "1",
            "--max-turns", "80",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    summary = json.loads(result.stdout)
    assert summary["games_played"] == 4
    assert summary["policy_a"].startswith("neural")
    assert summary["policy_b"] == "heuristic"
