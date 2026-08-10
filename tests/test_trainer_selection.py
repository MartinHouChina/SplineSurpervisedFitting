from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.training.trainer import Trainer


def _metrics(
    loss: float,
    existence_f1: float,
    knot_match_f1: float,
    knot_match_precision: float = 0.5,
    knot_matched_mae: float = 0.02,
) -> dict[str, float]:
    return {
        "loss": loss,
        "fit_loss": loss,
        "existence_loss": 0.5,
        "knot_position_loss": 0.1,
        "expected_active_count": 2.0,
        "hard_active_count": 2.0,
        "candidate_knot_count": 3.0,
        "gate_nonzero_count": 2.0,
        "existence_f1": existence_f1,
        "knot_match_f1": knot_match_f1,
        "knot_match_precision": knot_match_precision,
        "knot_matched_mae": knot_matched_mae,
    }


class TrainerSelectionTests(unittest.TestCase):
    def test_mean_metrics_uses_global_knot_match_counts(self) -> None:
        metrics = Trainer._mean_metrics(
            {
                "knot_match_count": 3.0,
                "knot_predicted_count": 5.0,
                "knot_target_count": 4.0,
                "knot_match_error_sum": 0.06,
            },
            samples=1,
        )

        self.assertAlmostEqual(metrics["knot_match_precision"], 0.6)
        self.assertAlmostEqual(metrics["knot_match_recall"], 0.75)
        self.assertAlmostEqual(metrics["knot_match_f1"], 2 * 0.6 * 0.75 / 1.35)
        self.assertAlmostEqual(metrics["knot_matched_mae"], 0.02)

    def test_checkpoint_prioritizes_geometric_knot_f1_over_other_metrics(self) -> None:
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = Trainer(model, torch.nn.Identity(), optimizer, torch.device("cpu"))
        loader = DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1)
        epoch_metrics = [
            _metrics(0.2, 0.7, 0.4),
            _metrics(0.2, 0.7, 0.4),
            _metrics(0.4, 0.6, 0.5),
            _metrics(0.4, 0.6, 0.5),
            _metrics(0.1, 0.9, 0.49),
            _metrics(0.1, 0.9, 0.49),
        ]

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "best.pt"
            with patch.object(trainer, "_run_epoch", side_effect=epoch_metrics):
                trainer.fit(
                    loader,
                    loader,
                    epochs=3,
                    checkpoint_path=checkpoint_path,
                )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )

        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(
            checkpoint["selection_metric"],
            "knot_match_f1_then_precision_mae_loss",
        )
        self.assertAlmostEqual(checkpoint["selection_value"], 0.5)
        self.assertAlmostEqual(checkpoint["best_val"], 0.4)


if __name__ == "__main__":
    unittest.main()
