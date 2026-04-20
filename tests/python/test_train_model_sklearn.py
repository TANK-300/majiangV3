"""A-2a: sanity tests for the upgraded sklearn-backed trainer.

The trainer MUST stay compatible with the C++ V3LinearModel loader:
* model.json exposes `feature_names`, `feature_means`, `feature_stds`,
  `weights`, `intercept`, `model_type` in the exact schema it has today;
* length of `weights` == length of `feature_names`;
* model_type is one of {logistic_regression, linear_regression}.

These are guard-rails so that swapping in sklearn here doesn't silently
break the C++ inference path -- any change to the on-disk schema would have
to be paired with matching C++ changes.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"


pytest.importorskip("sklearn")


def _write_dataset(path: Path, rows: list) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def test_sklearn_trainer_produces_v3linearmodel_compatible_json(tmp_path: Path) -> None:
    dataset = tmp_path / "canonical.jsonl"
    rows = []
    for i in range(40):
        rows.append(
            {
                "record_id": f"pos-{i}",
                "model_features": {
                    "wall_remaining": float(60 - i),
                    "hand_t_white": 2.0 if i % 2 == 0 else 0.0,
                    "opp_disc_t_1w": float(i % 3),
                    "hand_size": 13.0,
                },
                "task_labels": {"can_win_label": bool(i % 2 == 0)},
            }
        )
    _write_dataset(dataset, rows)

    target = tmp_path / "params"
    subprocess.check_call(
        [
            sys.executable,
            str(TOOLS_DIR / "train_model_stub.py"),
            "--task",
            "agari_prob",
            "--version-dir",
            str(target),
            "--dataset",
            str(dataset),
            "--validation-split",
            "0.25",
        ]
    )

    model_path = target / "v3" / "agari_prob" / "model.json"
    meta_path = target / "v3" / "agari_prob" / "model_meta.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert set(
        {"task", "label_key", "model_type", "feature_names", "feature_means",
         "feature_stds", "intercept", "weights", "training_examples", "metrics"}
    ).issubset(model.keys()), "A-2a: on-disk schema must remain a superset of what C++ V3LinearModel loads"

    assert model["model_type"] == "logistic_regression"
    assert isinstance(model["feature_names"], list) and model["feature_names"]
    assert len(model["weights"]) == len(model["feature_names"])
    assert len(model["feature_means"]) == len(model["feature_names"])
    assert len(model["feature_stds"]) == len(model["feature_names"])
    for name in model["feature_names"]:
        assert name in model["feature_means"]
        assert name in model["feature_stds"]

    # A-2a: fitter and metrics
    assert model.get("fitter") == "sklearn"
    assert "train_accuracy" in model["metrics"]
    assert "val_accuracy" in model["metrics"]

    assert meta["status"] == "trained"
    assert meta["trained_examples"] == len(rows)


def test_sklearn_trainer_handles_regression_target(tmp_path: Path) -> None:
    dataset = tmp_path / "reg.jsonl"
    rows = []
    for i in range(40):
        rows.append(
            {
                "record_id": f"tsn-{i}",
                "model_features": {
                    "wall_remaining": float(60 - i),
                    "hand_t_white": float(i % 4),
                    "hand_size": 13.0,
                },
                "source_meta": {"tsumo_num_label": float(i % 5)},
            }
        )
    _write_dataset(dataset, rows)

    target = tmp_path / "params"
    subprocess.check_call(
        [
            sys.executable,
            str(TOOLS_DIR / "train_model_stub.py"),
            "--task",
            "tsumo_num",
            "--version-dir",
            str(target),
            "--dataset",
            str(dataset),
            "--validation-split",
            "0.25",
        ]
    )

    model = json.loads((target / "v3" / "tsumo_num" / "model.json").read_text(encoding="utf-8"))
    assert model["model_type"] == "linear_regression"
    assert model["fitter"] == "sklearn"
    assert len(model["weights"]) == len(model["feature_names"])
    assert "train_r2" in model["metrics"]
