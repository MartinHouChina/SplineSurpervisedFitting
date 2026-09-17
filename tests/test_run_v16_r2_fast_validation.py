from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v16_r2_fast_validation.sh"


def test_r2_fast_validation_is_isolated_and_keeps_historical_contract():
    source = SCRIPT.read_text(encoding="utf-8")
    required = (
        "LEGACY_COMMIT=48d8b4e",
        "outputs/rollback_v16_r2_source",
        "git -C \"$REPO_ROOT\" worktree add --detach",
        "EPOCHS=13",
        "PROPOSAL_EPOCHS=1",
        "TRAIN_SIZE=600",
        "VAL_SIZE=160",
        "REAL_VAL_SIZE=20",
        "BATCH_SIZE=128",
        "--min-control-points 8 --max-control-points 60 --candidate-knots 56",
        "--synthetic-boundary-val-size 16",
        "--mse-tolerance 1e-4",
        "--certified-minimal-source",
        "--synthetic-count-role upper_bound",
        "--synthetic-geometry-oracle-teacher",
        "--one-shot-selection-policy mass_topk",
        "--init-checkpoint \"$WARMSTART\"",
        "quick_compare_v16_checkpoints.py",
        "--min-source-k 4 --max-source-k 56 --samples-per-k 1",
        "refusing to overwrite existing run artifact",
    )
    for fragment in required:
        assert fragment in source
    assert "git reset" not in source
    assert "git checkout" not in source
    assert "--include-verified-ours" not in source


def test_r2_fast_validation_bash_syntax():
    bash = shutil.which("bash") if os.name != "nt" else Path(
        r"C:\Program Files\Git\bin\bash.exe"
    )
    if not bash or not Path(bash).is_file():
        pytest.skip("native Bash is unavailable")
    result = subprocess.run(
        [str(bash), "-n", str(SCRIPT)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
