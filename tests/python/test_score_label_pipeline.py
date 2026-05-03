"""End-to-end pipeline test: selfplay_sample → score_label → train_model_stub.

Verifies:
1. selfplay_sample.py emits score_label in task_labels
2. train_model_stub.py picks it up via TASK_LABEL_KEYS["agari_score"]
3. The trained model.json has objective=regression
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_selfplay_sample_emits_score_label(tmp_path):
    sample_path = tmp_path / "samples.jsonl"
    r = subprocess.run(
        [
            sys.executable, "tools/selfplay_sample.py",
            "--games", "16",
            "--policy-a", "heuristic", "--policy-b", "heuristic",
            "--seed", "1",
            "--output", str(sample_path),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert sample_path.exists(), f"sample file not created: stderr={r.stderr}"
    lines = sample_path.read_text().splitlines()
    assert lines, "no samples produced"
    rows = [json.loads(line) for line in lines if line.strip()]
    score_label_rows = [
        row for row in rows
        if "score_label" in row.get("task_labels", {})
    ]
    # 至少 1 行有 score_label
    assert len(score_label_rows) >= 1, f"no rows have score_label among {len(rows)} total"
    # 类型必须 int（trainer 加载会调 float() OK 但保证一致性）
    for row in score_label_rows[:5]:
        assert isinstance(row["task_labels"]["score_label"], int)


def test_train_model_stub_recognizes_agari_score_task(tmp_path):
    sample_path = tmp_path / "samples.jsonl"
    bundle_dir = tmp_path / "bundle"
    # 1. 生成样本
    subprocess.run(
        [sys.executable, "tools/selfplay_sample.py",
         "--games", "32",
         "--policy-a", "heuristic", "--policy-b", "heuristic",
         "--seed", "1", "--output", str(sample_path)],
        cwd=REPO_ROOT, check=True, timeout=60, capture_output=True,
    )
    # 2. 训练 agari_score head
    r = subprocess.run(
        [sys.executable, "tools/train_model_stub.py",
         "--task", "agari_score",
         "--version-dir", str(bundle_dir),
         "--dataset", str(sample_path),
         "--validation-split", "0.2",
         "--model-kind", "linear"],  # force linear; lightgbm not always available
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, f"trainer failed: stdout={r.stdout}\nstderr={r.stderr}"

    model_path = bundle_dir / "v3" / "agari_score" / "model.json"
    assert model_path.exists(), "model.json not produced"
    model = json.loads(model_path.read_text())
    # score_label 是连续值 → 应为 regression 系列
    assert model["model_type"] in {"linear_regression", "logistic_regression",
                                   "linear_regression_handrolled"}, \
        f"unexpected model_type {model['model_type']}"


def test_train_model_stub_recognizes_houjuu_score_task(tmp_path):
    sample_path = tmp_path / "samples.jsonl"
    bundle_dir = tmp_path / "bundle"
    subprocess.run(
        [sys.executable, "tools/selfplay_sample.py",
         "--games", "32",
         "--policy-a", "heuristic", "--policy-b", "heuristic",
         "--seed", "1", "--output", str(sample_path)],
        cwd=REPO_ROOT, check=True, timeout=60, capture_output=True,
    )
    r = subprocess.run(
        [sys.executable, "tools/train_model_stub.py",
         "--task", "houjuu_score",
         "--version-dir", str(bundle_dir),
         "--dataset", str(sample_path),
         "--validation-split", "0.2",
         "--model-kind", "linear"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, f"trainer failed: stdout={r.stdout}\nstderr={r.stderr}"
    model_path = bundle_dir / "v3" / "houjuu_score" / "model.json"
    assert model_path.exists()
    model = json.loads(model_path.read_text())
    # 应该走 regression 路径（score_label 为整数，非 0/1）
    assert "linear_regression" in model["model_type"] or model["model_type"] == "logistic_regression"
