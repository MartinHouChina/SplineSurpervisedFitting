from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spline_fitting.checkpointing import assess_v16_checkpoint
from test_inspect_v16_checkpoint import _checkpoint


def test_mixed_pretraining_cannot_be_reported_as_synthetic_only():
    payload = _checkpoint()
    payload["initialization_provenance"] = {
        "mode": "full_model", "source_checkpoint": "historical.pt",
        "synthetic_only_model_lineage": False,
        "source_training_real_fraction": 0.5,
    }
    result = assess_v16_checkpoint(payload)
    assert result["formal_reporting_eligible"] is False
    assert result["synthetic_only_model_lineage"] is False
    assert any("pretraining exposure" in reason for reason in result["reasons"])


def test_full_model_finetuning_can_skip_completed_proposal_stage():
    payload = _checkpoint()
    payload["initialization_provenance"] = {
        "mode": "full_model", "source_checkpoint": "synthetic_only.pt",
        "synthetic_only_model_lineage": True,
    }
    payload["training_config"]["proposal_epochs"] = 0
    result = assess_v16_checkpoint(payload)
    assert not any("proposal_epochs is missing or invalid" in reason for reason in result["reasons"])
    missing = deepcopy(payload)
    missing.pop("initialization_provenance")
    result = assess_v16_checkpoint(missing)
    assert any("proposal_epochs is missing or invalid" in reason for reason in result["reasons"])
