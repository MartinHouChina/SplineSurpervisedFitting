import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("paper_render", Path(__file__).resolve().parents[1] / "scripts/render_postprune_paper.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_command_chain_uses_one_report_and_checkpoint():
    commands = module.commands(Path("report.json"), Path("model.pt"), Path("figures"), 300, 17)
    assert len(commands) == 4
    assert len({cmd[cmd.index("--report") + 1] for cmd in commands}) == 1
    assert len({cmd[cmd.index("--output-dir") + 1] for cmd in commands}) == 1
    assert commands[1][commands[1].index("--seed") + 1] == "17"
    assert all("train" not in Path(cmd[1]).name for cmd in commands)


def test_render_checks_hash_preserves_existing_and_calls_children(tmp_path, monkeypatch):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"fixture")
    report = tmp_path / "comparison.json"
    report.write_text(json.dumps({"metadata": {"checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(b"fixture").hexdigest()}}), encoding="utf-8")
    output = tmp_path / "rendered"
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    assert module.main(["--report", str(report), "--output-dir", str(output)]) == 0
    assert len(calls) == 4
    assert all(kwargs["check"] and kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1" for _, kwargs in calls)
    artifact = output / "existing.png"
    artifact.write_bytes(b"preserve")
    with pytest.raises(SystemExit):
        module.main(["--report", str(report), "--output-dir", str(output)])
    assert artifact.read_bytes() == b"preserve"
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(SystemExit):
        module.main(["--report", str(report), "--output-dir", str(tmp_path / "next")])
