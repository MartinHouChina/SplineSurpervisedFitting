"""CPU-only launcher checks: quoting, safe output, logs, and failure propagation."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_v16_short_ablation_linux.sh"


def _bash() -> str:
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    executable = str(git_bash) if git_bash.is_file() else shutil.which("bash")
    if executable is None:
        pytest.skip("Bash is not available")
    return executable


def _bash_path(path: Path) -> str:
    value = path.resolve().as_posix()
    if re.match(r"^[A-Za-z]:/", value):
        return f"/{value[0].lower()}/{value[3:]}"
    return value


def _run(*options: str, python: str = "/nonexistent-python-for-dry-run"):
    return subprocess.run(
        [_bash(), _bash_path(SCRIPT), *options],
        cwd=ROOT,
        env={**os.environ, "SPLINE_PYTHON": python},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _command(result) -> list[str]:
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(line for line in result.stdout.splitlines() if line.startswith("[short_ablation] "))
    return shlex.split(line.removeprefix("[short_ablation] "))


def test_dry_run_forwards_python_arguments_without_splitting_or_side_effects(tmp_path):
    output = tmp_path / "results with spaces"
    options = [
        "--checkpoint", "checkpoints/model with spaces.pt", "--data-root", "data with spaces",
        "--device", "cpu", "--epochs", "1", "--train-size", "4", "--val-size", "2",
        "--test-size", "2", "--modes", "fixed", "legacy", "geometry", "--skip-real-data",
        "--learning-rate", "3e-5", "--decoder-lr", "1e-5", "--teacher-max-deletions", "2",
        "--max-squared-error-tolerance", "5e-4",
    ]
    command = _command(_run("--dry-run", "--output-dir", _bash_path(output), *options))
    assert command[:3] == ["/nonexistent-python-for-dry-run", "-u", "scripts/run_v16_short_ablation.py"]
    assert command[3:] == ["--output-dir", _bash_path(output), *options]
    assert not list(tmp_path.iterdir())


def test_run_name_maps_to_new_ablation_directory():
    command = _command(_run("--dry-run", "--run-name", "pytest_short_k32_r1"))
    assert command[command.index("--output-dir") + 1] == "outputs/ablations/pytest_short_k32_r1"
    assert "--run-name" not in command
    assert "train_v16.py" not in command


@pytest.mark.parametrize("name", ["../existing", "/", "..", "has spaces", "", "bad/name"])
def test_invalid_run_names_are_rejected(name):
    result = _run("--dry-run", f"--run-name={name}")
    assert result.returncode != 0
    assert "--run-name" in result.stderr


def test_ambiguous_destination_and_missing_option_values_are_rejected(tmp_path):
    for options in [
        ["--run-name", "one", "--output-dir", _bash_path(tmp_path / "two")],
        ["--output-dir"], ["--run-name", "--epochs", "1"], ["--output-dir="],
    ]:
        result = _run("--dry-run", *options)
        assert result.returncode != 0
    assert not list(tmp_path.iterdir())


def test_help_does_not_require_python():
    result = _run("--help")
    assert result.returncode == 0
    assert "--checkpoint" in result.stdout and "--resume" in result.stdout


def test_existing_output_or_log_is_not_overwritten(tmp_path):
    output = tmp_path / "existing run"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("original", encoding="utf-8")
    result = _run("--output-dir", _bash_path(output))
    assert result.returncode != 0 and "already exists" in result.stderr
    assert marker.read_text(encoding="utf-8") == "original"
    log = tmp_path / "other run.run.log"
    log.write_text("old log", encoding="utf-8")
    result = _run("--output-dir", _bash_path(tmp_path / "other run"))
    assert result.returncode != 0 and "already exists" in result.stderr
    assert log.read_text(encoding="utf-8") == "old log"


def test_unbuffered_tee_keeps_python_failure_and_resume_appends(tmp_path):
    fake_python = tmp_path / "fake python.sh"
    fake_python.write_text(
        '#!/usr/bin/env bash\n'
        'printf "unbuffered=%s\\n" "$PYTHONUNBUFFERED"\n'
        'printf "argument=<%s>\\n" "$@"\n'
        'printf "intentional failure\\n" >&2\n'
        'exit 7\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    output = tmp_path / "new results"
    args = ["--output-dir", _bash_path(output), "--checkpoint", "path with spaces.pt"]
    result = _run(*args, python=_bash_path(fake_python))
    assert result.returncode == 7, result.stdout + result.stderr
    log = tmp_path / "new results.run.log"
    content = log.read_text(encoding="utf-8")
    assert "unbuffered=1" in content and "argument=<-u>" in content
    assert "argument=<path with spaces.pt>" in content and "intentional failure" in content
    assert not output.exists(), "the launcher must not pre-create the Python run directory"
    resumed = _run(*args, "--resume", python=_bash_path(fake_python))
    assert resumed.returncode == 7
    updated = log.read_text(encoding="utf-8")
    assert updated.startswith(content)
    assert updated.count("intentional failure") == 2
    assert "argument=<--resume>" in updated
