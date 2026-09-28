"""Audited parameter export must not relabel a different trained checkpoint."""
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("postprune_parameters", ROOT / "scripts/export_postprune_parameters.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def tiny_report():
    return {"metadata": {"methods": ["ours", "other"], "cases": [{"dataset": "Synthetic", "sample_id": "a"}],
                         "ours_post_pruning": {"enabled": True, "applies_to": "ours_only"}},
            "measurements": [{"dataset": "Synthetic", "sample_id": "a", "method": "ours", "status": "ok"},
                             {"dataset": "Synthetic", "sample_id": "a", "method": "other", "status": "failed"}]}


def test_failed_rows_count_as_completed():
    assert module.validate_completed_report(tiny_report())["methods"] == ["ours", "other"]


def test_incomplete_comparison_rejected():
    report = tiny_report()
    report["measurements"].pop()
    with pytest.raises(ValueError, match="exactly one"):
        module.validate_completed_report(report)


def test_duplicate_results_rejected():
    report = tiny_report()
    report["measurements"].append(report["measurements"][0].copy())
    with pytest.raises(ValueError, match="exactly one"):
        module.validate_completed_report(report)


def test_unpruned_report_rejected():
    report = tiny_report()
    report["metadata"]["ours_post_pruning"]["enabled"] = False
    with pytest.raises(ValueError, match="Ours-only"):
        module.validate_completed_report(report)


def test_checkpoint_hash_mismatch_prevents_audit(tmp_path, monkeypatch):
    report = tiny_report()
    report["metadata"]["checkpoint_sha256"] = "expected_hash"
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(module, "file_hash", lambda _: "wrong_hash")
    monkeypatch.setattr(module, "audit_checkpoint", lambda _: pytest.fail("must not audit a mismatched file"))
    with pytest.raises(ValueError, match="SHA256 does not match"):
        module.create_payload(path, tmp_path / "wrong.pt")


def test_missing_parameter_is_not_defaulted():
    with pytest.raises(ValueError, match="not recorded"):
        module._required({}, "epochs")


def test_parameter_table_render(tmp_path):
    rows = [{"section": "Model", "parameter": "Internal / full capacity", "setting": "32 / 40", "evidence": "fixture"},
            {"section": "Training", "parameter": "Configured Proposal + Joint", "setting": "12 + 20", "evidence": "fixture"}]
    target = tmp_path / "table.png"
    module.render_table(target, rows)
    assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
