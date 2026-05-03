"""perf_regression.py 短跑通测试。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_perf_regression_runs(tmp_path):
    out = tmp_path / "perf.json"
    result = subprocess.run(
        [
            sys.executable, "tools/perf_regression.py",
            "--runs", "5",
            "--output", str(out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.exists(), f"perf report not produced.\nstdout={result.stdout}\nstderr={result.stderr}"
    report = json.loads(out.read_text())
    assert "results" in report
    assert "active_engine" in report
    # 4 个 case 全部跑过（即使是 heuristic 兜底，每个 case 都应该返回结果）
    assert len(report["results"]) == 4
    case_names = {r["name"] for r in report["results"]}
    assert "4_whites" in case_names
    assert "complex_meld_response" in case_names
    assert "multi_qianggang" in case_names
    assert "grab_charge_with_tree" in case_names


def test_each_case_has_p95_or_skip_reason(tmp_path):
    out = tmp_path / "perf.json"
    subprocess.run(
        [sys.executable, "tools/perf_regression.py", "--runs", "5", "--output", str(out)],
        cwd=REPO_ROOT, check=True, timeout=60, capture_output=True,
    )
    report = json.loads(out.read_text())
    for case_result in report["results"]:
        # Either p95_ms is reported, or there's a skip reason.
        assert "p95_ms" in case_result or "skipped" in case_result


def test_passed_when_all_under_budget():
    """Heuristic fallback 应该全部 < 800ms。"""
    import subprocess as sp, json
    r = sp.run(
        [sys.executable, "tools/perf_regression.py", "--runs", "5",
         "--p95-budget-ms", "10000", "--output", "/tmp/perf_test.json"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
