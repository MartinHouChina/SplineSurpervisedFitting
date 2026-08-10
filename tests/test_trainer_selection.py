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


def _metrics(loss: float, existence_f1: float) -> dict[str, float]:
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
    }


class TrainerSelectionTests(unittest.TestCase):
    def test_checkpoint_prioritizes_existence_f1_over_lower_loss(self) -> None:
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = Trainer(model, torch.nn.Identity(), optimizer, torch.device("cpu"))
        loader = DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1)
        epoch_metrics = [
            _metrics(0.2, 0.5),
            _metrics(0.2, 0.5),
            _metrics(0.4, 0.6),
            _metrics(0.4, 0.6),
            _metrics(0.1, 0.59),
            _metrics(0.1, 0.59),
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
        self.assertEqual(checkpoint["selection_metric"], "existence_f1_then_loss")
        self.assertAlmostEqual(checkpoint["selection_value"], 0.6)
        self.assertAlmostEqual(checkpoint["best_val"], 0.4)


if __name__ == "__main__":
    unittest.main()
