from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v16_mse1e-4_3090.sh"


def test_linux_runner_encodes_current_training_and_evaluation_contract() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    required = (
        "#!/usr/bin/env bash",
        "set -Eeuo pipefail",
        'RUN_NAME="candidate_selection_v16_mse1e-4_k56_ordered_highk_softcost_linux"',
        'SIMPLIFICATION_CONTRACT="ranked_prefix_ordered_proposal_high_k_soft_subset_cost_v3"',
        "Checkpoint selection: mean_per_curve_subset_cost_v1; aggregate pass is reporting only.",
        "EPOCHS=104",
        "PROPOSAL_EPOCHS=40",
        "--min-control-points 8",
        "--max-control-points 60",
        "--candidate-knots 56",
        "--knot-min-span 0.01",
        "--synthetic-boundary-val-size 32",
        "--proposal-high-k-fraction \"$PROPOSAL_HIGH_K_FRACTION\"",
        "--proposal-high-k-min-knots \"$PROPOSAL_HIGH_K_MIN_KNOTS\"",
        "--proposal-knot-assignment-weight 1.0",
        "--prepare-real-data",
        "--mse-tolerance 1e-4",
        "--initial-keep-fraction 0.5357142857142857",
        "--min-knot-count 4 --max-knot-count 56",
        "--max-internal-knots 56",
        "--paper-initial-knots 56",
        "--liang-dense-knots 56",
        "--method-set published",
        "scripts/inspect_v16_checkpoint.py",
        "scripts/plot_v16_method_comparison.py",
        "scripts/visualize_v16_real_deployments.py",
    )
    for fragment in required:
        assert fragment in source

    assert source.index("scripts/train_v16.py") < source.index(
        "scripts/inspect_v16_checkpoint.py"
    )
    assert source.index("scripts/inspect_v16_checkpoint.py") < source.index(
        "scripts/benchmark_v16_datasets.py"
    )
    assert source.index("scripts/benchmark_v16_datasets.py") < source.index(
        "scripts/plot_v16_method_comparison.py"
    )
    assert source.index("scripts/plot_v16_method_comparison.py") < source.index(
        "scripts/visualize_v16_real_deployments.py"
    )
    assert "--resume" not in source
    assert "--overwrite" not in source
    assert "--proposal-pass-target" not in source
    assert "--deployment-pass-target" not in source
    assert "--complexity-pass-margin" not in source
    assert "--allow-infeasible-proposals" not in source
    assert "--required-pass-rate" not in source


def test_linux_runner_refuses_unknown_options_and_documents_help() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "unknown option" in source
    assert "--no-init-checkpoint" in source
    assert "rerun with --prepare-real-data" in source
    assert "--diagnostic" in source
    assert "--dry-run" in source
    assert "--python PATH" in source
    assert "--proposal-high-k-fraction X" in source
    assert "--proposal-high-k-min-knots N" in source


def test_linux_runner_has_valid_bash_syntax_when_bash_is_available() -> None:
    bash = shutil.which("bash")
    if bash is None or not Path(bash).as_posix().startswith("/"):
        pytest.skip("native /bin/bash is unavailable on this platform")
    completed = subprocess.run(
        [bash, "-n", str(SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
