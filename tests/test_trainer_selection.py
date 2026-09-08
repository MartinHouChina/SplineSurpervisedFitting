from __future__ import annotations

from contextlib import redirect_stdout
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.training.trainer import Trainer  # noqa: E402


def _metrics(
    loss: float,
    existence_f1: float,
    knot_match_f1: float,
    knot_match_precision: float = 0.5,
    knot_matched_mae: float = 0.02,
    count_accuracy: float = 0.0,
    count_mae: float = 0.0,
    deployment_pass: float = 0.0,
    deployment_rms: float = 1.0,
    deployment_rms_p95: float | None = None,
    retained_count: float = 2.0,
) -> dict[str, float]:
    return {
        "loss": loss,
        "fit_loss": loss,
        "existence_loss": 0.5,
        "knot_position_loss": 0.1,
        "expected_active_count": 2.0,
        "hard_active_count": retained_count,
        "candidate_knot_count": 3.0,
        "gate_nonzero_count": 2.0,
        "existence_f1": existence_f1,
        "count_loss": 0.5,
        "count_accuracy": count_accuracy,
        "count_absolute_error": count_mae,
        "knot_match_f1": knot_match_f1,
        "knot_match_precision": knot_match_precision,
        "knot_matched_mae": knot_matched_mae,
        "candidate_coverage_loss": 0.1,
        "candidate_recall": 0.9,
        "candidate_nearest_mae": 0.01,
        "safe_action_top1": 0.0,
        "false_stop_rate": 0.0,
        "unsafe_delete_rate": 0.0,
        "teacher_mask_accuracy": 0.8,
        "adaptive_keep_threshold_mean": 0.1,
        "deployment_threshold_satisfied_rate": deployment_pass,
        "deployment_bspline_rms": deployment_rms,
        "deployment_bspline_rms_p95": (
            deployment_rms if deployment_rms_p95 is None else deployment_rms_p95
        ),
        "deployment_retained_knot_count": retained_count,
    }


class TrainerSelectionTests(unittest.TestCase):
    def test_progress_line_reports_phase_percentage_rate_and_eta(self) -> None:
        line = Trainer._progress_line(
            phase="validation",
            completed=25,
            total=100,
            loss=0.125,
            elapsed=5.0,
        )

        self.assertIn("validation", line)
        self.assertIn("25.00%", line)
        self.assertIn("loss=0.125000", line)
        self.assertIn("5.00 batch/s", line)
        self.assertIn("ETA 00:15", line)

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

    def test_mean_metrics_derives_multiscale_candidate_recall(self) -> None:
        metrics = Trainer._mean_metrics(
            {
                "candidate_match_count": 3.0,
                "candidate_match_count_at_0p005": 1.0,
                "candidate_match_count_at_0p01": 2.0,
                "candidate_match_count_at_0p02": 3.0,
                "candidate_target_count": 4.0,
                "candidate_nearest_error_sum": 0.08,
            },
            samples=1,
        )

        self.assertAlmostEqual(metrics["candidate_recall_at_0p005"], 0.25)
        self.assertAlmostEqual(metrics["candidate_recall_at_0p01"], 0.5)
        self.assertAlmostEqual(metrics["candidate_recall_at_0p02"], 0.75)

    def test_geometric_metric_targets_are_warped_to_output_parameterization(
        self,
    ) -> None:
        true_params = torch.tensor([[0.0, 0.25, 1.0]], dtype=torch.float64)
        output_params = torch.tensor([[0.0, 0.50, 1.0]], dtype=torch.float64)
        true_knots = torch.tensor([[0.25, 0.75]], dtype=torch.float64)

        aligned = Trainer._true_knots_in_output_parameterization(
            true_knots,
            true_params,
            output_params,
        )

        torch.testing.assert_close(
            aligned,
            torch.tensor([[0.50, 5.0 / 6.0]], dtype=torch.float64),
        )

    def test_parameter_feedback_stage_logs_its_own_metric_schema(self) -> None:
        model = torch.nn.Linear(1, 1)
        model.structure_mode = "candidate_pruning_one_shot"
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = Trainer(model, torch.nn.Identity(), optimizer, torch.device("cpu"))
        loader = DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1)
        feedback_metrics = {
            "loss": 0.4,
            "fit_loss": 2.5e-5,
            "true_parameter_loss": 4.0e-4,
            "threshold_satisfied_rate": 0.75,
            "parameter_feedback_chord_blend_weight": 0.6,
            "parameter_feedback_gap_logit_shift_mean_abs": 0.025,
            "hard_active_count": 8.0,
            "candidate_knot_count": 28.0,
            "deployment_threshold_satisfied_rate": 0.8,
            "deployment_bspline_rms": 0.0045,
            "deployment_bspline_rms_p95": 0.006,
            "deployment_retained_knot_count": 8.0,
            "knot_match_f1": 0.5,
            "knot_match_precision": 0.5,
            "knot_matched_mae": 0.01,
        }

        stream = io.StringIO()
        with (
            patch.object(
                trainer,
                "_run_epoch",
                side_effect=[feedback_metrics, feedback_metrics],
            ),
            redirect_stdout(stream),
        ):
            trainer.fit(
                loader,
                loader,
                epochs=1,
                stage_name="parameter_feedback_calibration",
                deployment_validation=True,
            )

        report = stream.getvalue()
        self.assertIn("deploy_pass=0.800", report)
        self.assertIn("deploy_RMS=0.00450", report)
        self.assertIn("parameter_RMSE=0.02000", report)
        self.assertIn("chord_blend=0.600", report)
        self.assertIn("gap_shift=0.0250", report)
        self.assertIn("parameter_MSE=0.000400", report)
        self.assertNotIn("coverage=", report)
        self.assertNotIn("knot_pos=", report)

    def test_candidate_pretrain_checkpoint_prioritizes_full_candidate_fit(self) -> None:
        model = torch.nn.Linear(1, 1)
        model.structure_mode = "candidate_pruning"
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = Trainer(model, torch.nn.Identity(), optimizer, torch.device("cpu"))
        loader = DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1)

        broad_but_imprecise = _metrics(0.1, 0.0, 0.7)
        broad_but_imprecise.update(
            {
                "candidate_recall": 0.99,
                "candidate_nearest_mae": 0.001,
                "candidate_recall_at_0p005": 0.60,
                "candidate_recall_at_0p01": 0.95,
                "candidate_recall_at_0p02": 0.99,
                "deployment_threshold_satisfied_rate": 0.50,
                "deployment_bspline_rms": 0.02,
                "deployment_bspline_rms_p95": 0.03,
            }
        )
        strict_but_broader_metrics_lower = _metrics(0.2, 0.0, 0.6)
        strict_but_broader_metrics_lower.update(
            {
                "candidate_recall": 0.90,
                "candidate_nearest_mae": 0.02,
                "candidate_recall_at_0p005": 0.70,
                "candidate_recall_at_0p01": 0.90,
                "candidate_recall_at_0p02": 0.90,
                "deployment_threshold_satisfied_rate": 0.95,
                "deployment_bspline_rms": 0.004,
                "deployment_bspline_rms_p95": 0.006,
            }
        )
        epoch_metrics = [
            broad_but_imprecise,
            broad_but_imprecise,
            strict_but_broader_metrics_lower,
            strict_but_broader_metrics_lower,
        ]

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "proposal.pt"
            with patch.object(trainer, "_run_epoch", side_effect=epoch_metrics):
                trainer.fit(
                    loader,
                    loader,
                    epochs=2,
                    checkpoint_path=checkpoint_path,
                    stage_name="candidate_pretrain",
                )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )

        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(
            checkpoint["selection_metric"],
            "all_candidate_standard_bspline_feasible_then_"
            "candidate_recall_0p005_0p01_0p02_mae_rms_p95_loss",
        )
        self.assertAlmostEqual(checkpoint["selection_value"], 0.95)

    def test_proposal_parameter_warmup_selects_complete_weighted_loss(self) -> None:
        model = torch.nn.Linear(1, 1)
        model.structure_mode = "candidate_pruning"
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = Trainer(model, torch.nn.Identity(), optimizer, torch.device("cpu"))
        loader = DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1)
        lower_coordinate_mse = _metrics(0.1, 0.0, 0.0)
        lower_coordinate_mse["raw_proposal_parameter_loss"] = 0.01
        lower_coordinate_mse["proposal_parameter_log_gap_mae"] = 0.1
        lower_weighted_objective = _metrics(0.05, 0.0, 0.0)
        lower_weighted_objective["raw_proposal_parameter_loss"] = 0.02
        lower_weighted_objective["proposal_parameter_log_gap_mae"] = 0.

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "parameter_warmup.pt"
            with patch.object(
                trainer,
                "_run_epoch",
                side_effect=[
                    lower_coordinate_mse,
                    lower_coordinate_mse,
                    lower_weighted_objective,
                    lower_weighted_objective,
                ],
            ):
                trainer.fit(
                    loader,
                    loader,
                    epochs=2,
                    checkpoint_path=checkpoint_path,
                    epoch_offset=3,
                    stage_name="proposal_parameter_warmup",
                )
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )

        self.assertEqual(checkpoint["epoch"], 5)
        self.assertEqual(checkpoint["stage"], "proposal_parameter_warmup")
        self.assertEqual(
            checkpoint["selection_metric"],
            "parameter_warmup_val_loss_then_log_gap_mae_coordinate_mse",
        )
        self.assertAlmostEqual(checkpoint["selection_value"], 0.05)

    def test_proposal_parameter_warmup_uses_gap_then_coordinate_ties(self) -> None:
        model = torch.nn.Linear(1, 1)
        model.structure_mode = "candidate_pruning"
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = Trainer(model, torch.nn.Identity(), optimizer, torch.device("cpu"))
        loader = DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1)

        metrics = []
        for gap_mae, coordinate_mse in ((0.2, 0.01), (0.1, 0.03), (0.1, 0.02)):
            epoch = _metrics(0.05, 0.0, 0.0)
            epoch["proposal_parameter_log_gap_mae"] = gap_mae
            epoch["raw_proposal_parameter_loss"] = coordinate_mse
            metrics.extend((epoch, epoch))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "parameter_warmup.pt"
            with patch.object(trainer, "_run_epoch", side_effect=metrics):
                trainer.fit(
                    loader,
                    loader,
                    epochs=3,
                    checkpoint_path=checkpoint_path,
                    stage_name="proposal_parameter_warmup",
                )
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )

        self.assertEqual(checkpoint["epoch"], 3)
        self.assertEqual(checkpoint["selection_rank"], [-0.05, -0.1, -0.02])

    def test_candidate_pretrain_prioritizes_recall_after_fit_is_feasible(self) -> None:
        model = torch.nn.Linear(1, 1)
        model.structure_mode = "candidate_pruning"
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = Trainer(
            model,
            torch.nn.Identity(),
            optimizer,
            torch.device("cpu"),
            deployment_pass_rate_target=0.97,
        )
        loader = DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1)

        lower_rms_lower_recall = _metrics(0.1, 0.0, 0.5)
        lower_rms_lower_recall.update(
            {
                "candidate_recall": 0.95,
                "candidate_nearest_mae": 0.004,
                "candidate_recall_at_0p005": 0.70,
                "candidate_recall_at_0p01": 0.85,
                "candidate_recall_at_0p02": 0.95,
                "deployment_threshold_satisfied_rate": 0.99,
                "deployment_bspline_rms": 0.002,
                "deployment_bspline_rms_p95": 0.003,
            }
        )
        higher_rms_higher_recall = _metrics(0.2, 0.0, 0.7)
        higher_rms_higher_recall.update(
            {
                "candidate_recall": 0.99,
                "candidate_nearest_mae": 0.002,
                "candidate_recall_at_0p005": 0.85,
                "candidate_recall_at_0p01": 0.95,
                "candidate_recall_at_0p02": 0.99,
                "deployment_threshold_satisfied_rate": 0.97,
                "deployment_bspline_rms": 0.004,
                "deployment_bspline_rms_p95": 0.0048,
            }
        )
        epoch_metrics = [
            lower_rms_lower_recall,
            lower_rms_lower_recall,
            higher_rms_higher_recall,
            higher_rms_higher_recall,
        ]

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "proposal.pt"
            with patch.object(trainer, "_run_epoch", side_effect=epoch_metrics):
                trainer.fit(
                    loader,
                    loader,
                    epochs=2,
                    checkpoint_path=checkpoint_path,
                    stage_name="candidate_pretrain",
                )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )

        self.assertEqual(checkpoint["epoch"], 2)
        self.assertAlmostEqual(checkpoint["selection_value"], 0.85)

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

    def test_structured_checkpoint_prioritizes_count_mae_after_warmup(self) -> None:
        model = torch.nn.Linear(1, 1)
        model.structure_mode = "interactive_dynamic"
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = Trainer(model, torch.nn.Identity(), optimizer, torch.device("cpu"))
        loader = DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1)
        epoch_metrics = [
            _metrics(0.1, 0.0, 0.9, count_accuracy=0.8, count_mae=0.5),
            _metrics(0.1, 0.0, 0.9, count_accuracy=0.8, count_mae=0.5),
            _metrics(0.2, 0.0, 0.3, count_accuracy=0.4, count_mae=1.0),
            _metrics(0.2, 0.0, 0.3, count_accuracy=0.4, count_mae=1.0),
            _metrics(0.3, 0.0, 0.8, count_accuracy=0.6, count_mae=1.5),
            _metrics(0.3, 0.0, 0.8, count_accuracy=0.6, count_mae=1.5),
        ]

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "best.pt"
            with patch.object(trainer, "_run_epoch", side_effect=epoch_metrics):
                trainer.fit(
                    loader,
                    loader,
                    epochs=3,
                    checkpoint_selection_start_epoch=1,
                    checkpoint_path=checkpoint_path,
                )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )

        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(
            checkpoint["selection_metric"],
            "count_mae_then_accuracy_knot_f1_precision_mae_loss",
        )
        self.assertAlmostEqual(checkpoint["selection_value"], 1.0)
        self.assertAlmostEqual(checkpoint["best_count_mae"], 1.0)

    def test_one_shot_checkpoint_minimizes_knots_after_pass_constraint(self) -> None:
        model = torch.nn.Linear(1, 1)
        model.structure_mode = "candidate_pruning_one_shot"
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = Trainer(model, torch.nn.Identity(), optimizer, torch.device("cpu"))
        loader = DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1)
        epoch_metrics = [
            _metrics(
                0.1,
                0.95,
                0.8,
                deployment_pass=0.99,
                deployment_rms=0.003,
                retained_count=28.0,
            ),
            _metrics(
                0.1,
                0.95,
                0.8,
                deployment_pass=0.99,
                deployment_rms=0.003,
                retained_count=28.0,
            ),
            _metrics(
                0.3,
                0.70,
                0.6,
                deployment_pass=0.97,
                deployment_rms=0.004,
                retained_count=10.0,
            ),
            _metrics(
                0.3,
                0.70,
                0.6,
                deployment_pass=0.97,
                deployment_rms=0.004,
                retained_count=10.0,
            ),
            _metrics(
                0.2,
                0.90,
                0.7,
                deployment_pass=0.96,
                deployment_rms=0.0045,
                retained_count=5.0,
            ),
            _metrics(
                0.2,
                0.90,
                0.7,
                deployment_pass=0.96,
                deployment_rms=0.0045,
                retained_count=5.0,
            ),
        ]

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "best.pt"
            with patch.object(trainer, "_run_epoch", side_effect=epoch_metrics):
                trainer.fit(
                    loader,
                    loader,
                    epochs=3,
                    checkpoint_path=checkpoint_path,
                    deployment_validation=True,
                )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )

        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(
            checkpoint["selection_metric"],
            "constrained_min_knots_at_target_pass_then_mask_count_rms",
        )
        self.assertAlmostEqual(checkpoint["selection_value"], 10.0)
        self.assertAlmostEqual(checkpoint["best_deployment_bspline_rms"], 0.004)
        self.assertAlmostEqual(checkpoint["best_deployment_retained_knot_count"], 10.0)
        self.assertTrue(checkpoint["best_deployment_pass_constraint_satisfied"])

    def test_one_shot_checkpoint_maximizes_pass_before_feasible(self) -> None:
        model = torch.nn.Linear(1, 1)
        model.structure_mode = "candidate_pruning_one_shot"
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = Trainer(
            model,
            torch.nn.Identity(),
            optimizer,
            torch.device("cpu"),
            deployment_pass_rate_target=0.97,
        )
        loader = DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1)
        epoch_metrics = [
            _metrics(0.1, 0.8, 0.8, deployment_pass=0.8, retained_count=8.0),
            _metrics(0.1, 0.8, 0.8, deployment_pass=0.8, retained_count=8.0),
            _metrics(0.2, 0.7, 0.7, deployment_pass=0.9, retained_count=20.0),
            _metrics(0.2, 0.7, 0.7, deployment_pass=0.9, retained_count=20.0),
        ]

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "best.pt"
            with patch.object(trainer, "_run_epoch", side_effect=epoch_metrics):
                trainer.fit(
                    loader,
                    loader,
                    epochs=2,
                    checkpoint_path=checkpoint_path,
                    deployment_validation=True,
                )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )

        self.assertEqual(checkpoint["epoch"], 2)
        self.assertAlmostEqual(checkpoint["selection_value"], 0.9)
        self.assertFalse(checkpoint["best_deployment_pass_constraint_satisfied"])

    def test_v9_checkpoint_minimizes_complexity_after_real_fit_constraint(self) -> None:
        model = torch.nn.Linear(1, 1)
        model.structure_mode = "candidate_pruning_one_shot"
        model.pruning_head = torch.nn.Module()
        model.pruning_head.one_shot_fixed_proposal_geometry = True
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = Trainer(model, torch.nn.Identity(), optimizer, torch.device("cpu"))
        loader = DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1)
        epoch_metrics = [
            _metrics(
                0.2,
                0.7,
                0.5,
                deployment_pass=0.98,
                deployment_rms=0.003,
                deployment_rms_p95=0.004,
                retained_count=14.0,
            ),
            _metrics(
                0.2,
                0.7,
                0.5,
                deployment_pass=0.98,
                deployment_rms=0.003,
                deployment_rms_p95=0.004,
                retained_count=14.0,
            ),
            _metrics(
                0.1,
                0.8,
                0.7,
                deployment_pass=0.97,
                deployment_rms=0.004,
                deployment_rms_p95=0.0048,
                retained_count=8.0,
            ),
            _metrics(
                0.1,
                0.8,
                0.7,
                deployment_pass=0.97,
                deployment_rms=0.004,
                deployment_rms_p95=0.0048,
                retained_count=8.0,
            ),
        ]

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "best.pt"
            with patch.object(trainer, "_run_epoch", side_effect=epoch_metrics):
                trainer.fit(
                    loader,
                    loader,
                    epochs=2,
                    checkpoint_path=checkpoint_path,
                    deployment_validation=True,
                )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )

        # Both epochs satisfy the real deployment constraint, so v9 selects
        # the lower-complexity mask despite its slightly worse RMS/pass rate.
        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(
            checkpoint["selection_metric"],
            "v9_constrained_standard_bspline_min_knots_rms_p95_recall",
        )
        self.assertAlmostEqual(checkpoint["selection_value"], 8.0)
        self.assertAlmostEqual(checkpoint["best_deployment_bspline_rms_p95"], 0.0048)


if __name__ == "__main__":
    unittest.main()
