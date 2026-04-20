from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.selfplay_eval import HeuristicPolicy, RandomPolicy  # noqa: E402
from tools.selfplay_sample import run_sample_generation  # noqa: E402


def test_run_sample_generation_emits_per_step_records(tmp_path: Path) -> None:
    out = tmp_path / "samples.jsonl"
    stats = run_sample_generation(
        HeuristicPolicy(),
        HeuristicPolicy(),
        games=5,
        seed=7,
        swap_sides=True,
        max_turns=80,
        output_path=out,
    )
    assert stats.games == 5
    assert stats.steps > 0
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == stats.steps

    for record in lines:
        # Core metadata
        assert "record_id" in record
        assert record["action"] == "discard"
        assert record["discard_tile"] is not None
        assert record["seat_wind"] in {"east", "south"}
        assert record["wall_remaining"] >= 0

        # Features must be a populated dict of floats; this is what gets fed
        # back into train_model_stub.py.
        features = record["model_features"]
        assert isinstance(features, dict) and features
        for value in features.values():
            assert isinstance(value, (int, float))

        # Outcome labels must line up with the game that produced them.
        outcome = record["outcome"]
        assert outcome["result"] in {"win", "loss", "draw"}
        assert outcome["kind"] in {"tsumo", "ron", "draw"}
        assert outcome["steps_to_end"] >= 0

        # Task labels are what train_model_stub.py / eval_search.py consume.
        labels = record["task_labels"]
        assert isinstance(labels["can_win_label"], bool)
        assert labels["best_discard_tile"] == record["discard_tile"]
        source = record["source_meta"]
        assert source["houjuu_label"] in (0, 1)
        assert source["tsumo_num_label"] >= 0
        assert source["ryukyoku_label"] in (0, 1)


def test_emitted_jsonl_feeds_train_model_stub(tmp_path: Path) -> None:
    # Generate samples, then hand them straight to train_model_stub.py; this
    # is the canonical B-1 -> (future B-2) pipeline.
    dataset = tmp_path / "samples.jsonl"
    run_sample_generation(
        HeuristicPolicy(),
        RandomPolicy(random.Random(3)),
        games=25,
        seed=3,
        swap_sides=True,
        max_turns=80,
        output_path=dataset,
    )
    trained_dir = tmp_path / "params"
    result = subprocess.run(
        [
            sys.executable,
            str(TOOLS_DIR / "train_model_stub.py"),
            "--task",
            "agari_prob",
            "--version-dir",
            str(trained_dir),
            "--dataset",
            str(dataset),
            "--epochs",
            "20",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    status_line = json.loads(result.stdout.splitlines()[-1])
    assert status_line["status"] == "trained"
    model = json.loads((trained_dir / "v3" / "agari_prob" / "model.json").read_text(encoding="utf-8"))
    assert model["model_type"] in {"logistic_regression"}
    assert len(model["weights"]) == len(model["feature_names"])


def test_cli_emits_summary(tmp_path: Path) -> None:
    out = tmp_path / "samples.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(TOOLS_DIR / "selfplay_sample.py"),
            "--policy-a",
            "heuristic",
            "--policy-b",
            "random",
            "--games",
            "4",
            "--seed",
            "99",
            "--max-turns",
            "80",
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    summary = json.loads(result.stdout)
    assert summary["games_played"] == 4
    assert summary["steps_recorded"] > 0
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == summary["steps_recorded"]
