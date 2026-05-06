"""Integration smoke test: run.py --synthetic completes < 60s and meets G3 threshold."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_smoke_synthetic_run_completes_and_passes_threshold():
    artifacts = ROOT / "artifacts"
    if (artifacts / "metrics.json").exists():
        (artifacts / "metrics.json").unlink()
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "--synthetic"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    elapsed = time.perf_counter() - t0
    assert proc.returncode == 0, f"run.py exit={proc.returncode}\nSTDERR:\n{proc.stderr}"
    assert elapsed < 60, f"smoke run too slow: {elapsed:.1f}s (G2 budget 60s)"

    metrics_path = artifacts / "metrics.json"
    assert metrics_path.exists()
    with open(metrics_path) as f:
        metrics = json.load(f)
    assert set(metrics.keys()) >= {"ridge", "random_forest", "xgboost", "mlp"}
    best = max(m["mean_r2_core"] for m in metrics.values())
    assert best > 0.4, f"best mean_r2_core={best:.3f} below G3 threshold 0.4"


def test_smoke_real_dry_run_handles_missing_dirs(tmp_path):
    proc = subprocess.run(
        [
            sys.executable, str(ROOT / "run.py"),
            "--real", "--dry-run",
            "--hirise", str(tmp_path / "no_such"),
            "--mastcamz", str(tmp_path / "no_such2"),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"dry-run exit={proc.returncode}\nSTDERR:\n{proc.stderr}"
    assert "Missing input dirs" in (proc.stdout + proc.stderr)
