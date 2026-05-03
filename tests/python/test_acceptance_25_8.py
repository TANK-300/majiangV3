"""acceptance_25_8.py 端到端冒烟测试。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_acceptance_smoke_runs_to_completion(tmp_path):
    """16 局 / 1 seed 跑完不抛异常，输出 JSON 含必要字段。"""
    out = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable, "tools/acceptance_25_8.py",
            "--games-per-seed", "16",
            "--seeds", "1",
            "--policy-a", "heuristic",
            "--policy-b", "heuristic",
            "--output", str(out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    # passed=False is OK on smoke; we only require the script ran.
    assert out.exists(), f"Output file not produced.\nstdout={result.stdout}\nstderr={result.stderr}"
    report = json.loads(out.read_text())
    assert "passed" in report
    assert "per_seed" in report
    assert len(report["per_seed"]) == 1
    assert report["per_seed"][0]["seed"] == 1
    assert "windows" in report["per_seed"][0]


def test_settle_windows_correctly_drops_partial():
    """8-game windowing should drop partial trailing window."""
    from tools.acceptance_25_8 import settle_windows
    chongs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # 11 games
    windows = settle_windows(chongs, window_size=8)
    assert len(windows) == 1
    assert windows[0] == sum(range(1, 9))


def test_settle_windows_exact_multiple():
    from tools.acceptance_25_8 import settle_windows
    chongs = [1] * 24  # 3 windows of 8
    windows = settle_windows(chongs, window_size=8)
    assert windows == [8, 8, 8]


def test_evaluate_pass_rejects_negative_window():
    from tools.acceptance_25_8 import evaluate_pass
    per_seed = [{
        "seed": 1, "games": 16,
        "windows": [10, -3, 5],
        "all_positive": False, "min_window": -3,
        "avg_chong_per_game": 0.75, "houjuu_rate": 0.10, "winrate": 0.6,
    }]
    summary = evaluate_pass(per_seed, min_window_chong=2,
                            min_avg_chong=0.5, max_houjuu_rate=0.12)
    assert summary["passed"] is False
    assert summary["min_window_chong"] == -3


def test_evaluate_pass_accepts_strong_result():
    from tools.acceptance_25_8 import evaluate_pass
    per_seed = [{
        "seed": s, "games": 200,
        "windows": [5] * 25,
        "all_positive": True, "min_window": 5,
        "avg_chong_per_game": 0.625, "houjuu_rate": 0.10, "winrate": 0.65,
    } for s in range(1, 6)]
    summary = evaluate_pass(per_seed, min_window_chong=2,
                            min_avg_chong=0.5, max_houjuu_rate=0.12)
    assert summary["passed"] is True
    assert summary["all_positive_window_count"] == 125
    assert summary["total_window_count"] == 125
