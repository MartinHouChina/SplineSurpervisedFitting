from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from ..evaluation.knot_diagnostics import activity_statistics, match_internal_knots


class Trainer:
    """Trainer with gate warm-up, expected-L0 scheduling and temperature annealing."""

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        grad_clip: float | None = 5.0,
        activity_threshold: float = 0.5,
        knot_match_tolerance: float = 0.05,
    ) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.device = device
        self.grad_clip = grad_clip
        self.activity_threshold = activity_threshold
        if knot_match_tolerance < 0.0:
            raise ValueError("knot_match_tolerance must be non-negative")
        self.knot_match_tolerance = float(knot_match_tolerance)

    @staticmethod
    def _mean_metrics(accumulator: dict[str, float], samples: int) -> dict[str, float]:
        metrics = {key: value / max(samples, 1) for key, value in accumulator.items()}
        required = {
            "existence_true_positive_count",
            "existence_predicted_count",
            "existence_target_count",
        }
        if required.issubset(metrics):
            true_positive = metrics["existence_true_positive_count"]
            predicted = metrics["existence_predicted_count"]
            target = metrics["existence_target_count"]
            precision = true_positive / predicted if predicted > 0.0 else 0.0
            recall = true_positive / target if target > 0.0 else 0.0
            metrics["existence_precision"] = precision
            metrics["existence_recall"] = recall
            metrics["existence_f1"] = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall > 0.0
                else 0.0
            )
        knot_required = {
            "knot_match_count",
            "knot_predicted_count",
            "knot_target_count",
            "knot_match_error_sum",
        }
        if knot_required.issubset(metrics):
            matched = metrics["knot_match_count"]
            predicted = metrics["knot_predicted_count"]
            target = metrics["knot_target_count"]
            precision = matched / predicted if predicted > 0.0 else 0.0
            recall = matched / target if target > 0.0 else 0.0
            metrics["knot_match_precision"] = precision
            metrics["knot_match_recall"] = recall
            metrics["knot_match_f1"] = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall > 0.0
                else 0.0
            )
            metrics["knot_matched_mae"] = (
                metrics["knot_match_error_sum"] / matched
                if matched > 0.0
                else float("nan")
            )
        return metrics

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool,
        l0_scale: float,
        activity_scale: float,
        binary_scale: float,
    ) -> dict[str, float]:
        self.model.train(training)
        totals: dict[str, float] = defaultdict(float)

        for batch in loader:
            points = batch["points"].to(self.device)
            chord_params = batch["chord_params"].to(self.device)
            true_params = batch.get("true_params")
            if true_params is not None:
                true_params = true_params.to(self.device)
            true_internal_knots = batch.get("true_internal_knots")
            true_internal_knot_mask = batch.get("true_internal_knot_mask")
            if true_internal_knots is not None:
                true_internal_knots = true_internal_knots.to(self.device)
            if true_internal_knot_mask is not None:
                true_internal_knot_mask = true_internal_knot_mask.to(self.device)
            true_internal_knot_count = (
                true_internal_knot_mask.to(torch.long).sum(dim=-1)
                if true_internal_knot_mask is not None
                else None
            )

            with torch.set_grad_enabled(training):
                teacher_count = (
                    true_internal_knot_count
                    if training
                    and getattr(self.model, "structure_mode", None)
                    == "count_conditioned"
                    else None
                )
                output = self.model(
                    points,
                    true_internal_knot_count=teacher_count,
                )
                losses = self.loss_fn(
                    output,
                    points,
                    chord_params=chord_params,
                    true_params=true_params,
                    true_internal_knots=true_internal_knots,
                    true_internal_knot_mask=true_internal_knot_mask,
                    l0_scale=l0_scale,
                    activity_scale=activity_scale,
                    binary_scale=binary_scale,
                    activity_threshold=self.activity_threshold,
                )
                if "knot_mask" in output:
                    knot_mask_float = output["knot_mask"].to(points.dtype)
                    knot_metrics = {
                        "activity_mass": output["expected_knot_count"],
                        "hard_active_count": knot_mask_float.sum(dim=-1),
                        "candidate_knot_count": torch.full(
                            (points.shape[0],),
                            float(output["knot_mask"].shape[-1]),
                            device=points.device,
                            dtype=points.dtype,
                        ),
                    }
                else:
                    knot_metrics = activity_statistics(
                        output["activity"], self.activity_threshold
                    )

                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    losses["loss"].backward()
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                    self.optimizer.step()

            batch_size = points.shape[0]
            geometric_metrics: dict[str, torch.Tensor] = {}
            if true_internal_knots is not None and true_internal_knot_mask is not None:
                matched_count = 0
                predicted_count = 0
                target_count = 0
                matched_error_sum = 0.0
                retained_mask = output.get("knot_mask")
                if retained_mask is None:
                    retained_mask = output["activity"] >= self.activity_threshold
                for sample_index in range(batch_size):
                    match = match_internal_knots(
                        output["internal_knots"][sample_index, retained_mask[sample_index]],
                        true_internal_knots[
                            sample_index, true_internal_knot_mask[sample_index]
                        ],
                        tolerance=self.knot_match_tolerance,
                    )
                    matched_count += match.matched_count
                    predicted_count += match.predicted_count
                    target_count += match.true_count
                    if match.matched_count:
                        matched_error_sum += match.matched_mae * match.matched_count
                geometric_metrics = {
                    "knot_match_count": points.new_tensor(matched_count / batch_size),
                    "knot_predicted_count": points.new_tensor(
                        predicted_count / batch_size
                    ),
                    "knot_target_count": points.new_tensor(target_count / batch_size),
                    "knot_match_error_sum": points.new_tensor(
                        matched_error_sum / batch_size
                    ),
                }

            actual_gate = output.get("knot_mask")
            if actual_gate is None:
                actual_gate = output.get("activity_gate", output["activity"])
            actual_gate = actual_gate.to(points.dtype)
            metrics = {
                **losses,
                **geometric_metrics,
                "activity_mass": knot_metrics["activity_mass"].mean(),
                "hard_active_count": knot_metrics["hard_active_count"].mean(),
                "gate_mass": actual_gate.sum(dim=-1).mean(),
                "gate_nonzero_count": (actual_gate > 0.0)
                .to(actual_gate.dtype)
                .sum(dim=-1)
                .mean(),
                "candidate_knot_count": knot_metrics["candidate_knot_count"].mean(),
            }
            for key, value in metrics.items():
                totals[key] += float(value.detach().cpu()) * batch_size

        return self._mean_metrics(totals, len(loader.dataset))

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        epochs: int,
        l0_schedule: Callable[[int], float] | None = None,
        activity_schedule: Callable[[int], float] | None = None,
        binary_schedule: Callable[[int], float] | None = None,
        gate_temperature_schedule: Callable[[int], float] | None = None,
        gate_warmup_epochs: int = 0,
        checkpoint_selection_start_epoch: int = 0,
        checkpoint_path: str | Path | None = None,
        epoch_offset: int = 0,
        stage_name: str = "joint",
    ) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        best_val = float("inf")
        best_knot_match_f1 = float("-inf")
        best_knot_match_precision = float("-inf")
        best_knot_matched_mae = float("inf")

        for epoch in range(epochs):
            l0_scale = l0_schedule(epoch) if l0_schedule else 1.0
            activity_scale = activity_schedule(epoch) if activity_schedule else 1.0
            binary_scale = binary_schedule(epoch) if binary_schedule else 1.0
            gate_temperature = (
                gate_temperature_schedule(epoch)
                if gate_temperature_schedule is not None
                else None
            )
            if gate_temperature is not None and hasattr(
                self.model, "set_gate_temperature"
            ):
                self.model.set_gate_temperature(gate_temperature)
            if hasattr(self.model, "set_force_open_gates"):
                self.model.set_force_open_gates(epoch < gate_warmup_epochs)
            train_metrics = self._run_epoch(
                train_loader, True, l0_scale, activity_scale, binary_scale
            )

            record = {f"train/{k}": v for k, v in train_metrics.items()}
            if val_loader is not None:
                val_metrics = self._run_epoch(val_loader, False, l0_scale, 1.0, 1.0)
                record.update({f"val/{k}": v for k, v in val_metrics.items()})
                current_val = val_metrics["loss"]
                selection_metrics = val_metrics
            else:
                current_val = train_metrics["loss"]
                selection_metrics = train_metrics

            current_knot_match_f1 = selection_metrics.get("knot_match_f1", 0.0)
            current_knot_match_precision = selection_metrics.get(
                "knot_match_precision", 0.0
            )
            current_knot_matched_mae = selection_metrics.get(
                "knot_matched_mae", float("inf")
            )
            if current_knot_matched_mae != current_knot_matched_mae:
                current_knot_matched_mae = float("inf")

            history.append(record)
            displayed_metrics = val_metrics if val_loader is not None else train_metrics
            if getattr(self.model, "structure_mode", None) == "count_conditioned":
                structure_report = (
                    f"count_loss={displayed_metrics['count_loss']:.4f} | "
                    f"count_acc={displayed_metrics.get('count_accuracy', 0.0):.3f} | "
                    f"count_MAE={displayed_metrics.get('count_absolute_error', 0.0):.3f} | "
                    f"K={displayed_metrics['hard_active_count']:.2f}/"
                    f"{displayed_metrics['candidate_knot_count']:.0f}"
                )
            else:
                structure_report = (
                    f"exist={displayed_metrics['existence_loss']:.4f} | "
                    f"exist_F1={displayed_metrics.get('existence_f1', 0.0):.3f} | "
                    f"E[K]={displayed_metrics['expected_active_count']:.2f} | "
                    f"active@{self.activity_threshold:.2f}="
                    f"{displayed_metrics['hard_active_count']:.2f}/"
                    f"{displayed_metrics['candidate_knot_count']:.0f} | "
                    f"gate_nonzero={displayed_metrics['gate_nonzero_count']:.2f}"
                )
            print(
                f"Stage {stage_name} | "
                f"Epoch {epoch_offset + epoch + 1:04d} "
                f"(stage {epoch + 1:04d}/{epochs:04d}) | "
                f"train={train_metrics['loss']:.6f} | val={current_val:.6f} | "
                f"fit={displayed_metrics['fit_loss']:.6f} | "
                f"{structure_report} | "
                f"knot_F1@{self.knot_match_tolerance:.3f}="
                f"{displayed_metrics.get('knot_match_f1', 0.0):.3f} | "
                f"knot_pos={displayed_metrics['knot_position_loss']:.4f} | "
                f"l0_scale={l0_scale:.3f} | "
                f"temperature={gate_temperature if gate_temperature is not None else float('nan'):.3f}"
            )

            current_rank = (
                current_knot_match_f1,
                current_knot_match_precision,
                -current_knot_matched_mae,
                -current_val,
            )
            best_rank = (
                best_knot_match_f1,
                best_knot_match_precision,
                -best_knot_matched_mae,
                -best_val,
            )
            if (
                checkpoint_path is not None
                and epoch >= checkpoint_selection_start_epoch
                and current_rank > best_rank
            ):
                best_val = current_val
                best_knot_match_f1 = current_knot_match_f1
                best_knot_match_precision = current_knot_match_precision
                best_knot_matched_mae = current_knot_matched_mae
                path = Path(checkpoint_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "epoch": epoch_offset + epoch + 1,
                        "stage": stage_name,
                        "best_val": current_val,
                        "selection_metric": (
                            "knot_match_f1_then_precision_mae_loss"
                        ),
                        "selection_value": current_knot_match_f1,
                        "best_knot_match_f1": best_knot_match_f1,
                        "best_knot_match_precision": best_knot_match_precision,
                        "best_knot_matched_mae": best_knot_matched_mae,
                        "knot_match_tolerance": self.knot_match_tolerance,
                        "best_existence_f1": selection_metrics.get(
                            "existence_f1", 0.0
                        ),
                        "best_count_accuracy": selection_metrics.get(
                            "count_accuracy", 0.0
                        ),
                        "best_count_mae": selection_metrics.get(
                            "count_absolute_error", float("nan")
                        ),
                        "metrics": record,
                        "history": list(history),
                        "activity_threshold": self.activity_threshold,
                        "gate_temperature": gate_temperature,
                        "gate_warmup_epochs": gate_warmup_epochs,
                        "checkpoint_selection_start_epoch": (
                            checkpoint_selection_start_epoch
                        ),
                    },
                    path,
                )

        return history
