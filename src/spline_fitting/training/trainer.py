from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys
import time
from typing import Callable

import torch
from torch.utils.data import DataLoader

from ..evaluation.bspline_inference import refit_hard_gated_bspline_batch
from ..evaluation.knot_diagnostics import (
    activity_statistics,
    match_internal_knots,
    warp_internal_knots_to_parameterization,
)


class Trainer:
    """Train and validate current or checkpoint-compatible spline models."""

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        grad_clip: float | None = 5.0,
        activity_threshold: float = 0.5,
        knot_match_tolerance: float = 0.05,
        log_every_batches: int = 25,
        deployment_pass_rate_target: float = 0.97,
        show_progress: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.device = device
        self.grad_clip = grad_clip
        self.activity_threshold = activity_threshold
        if log_every_batches < 0:
            raise ValueError("log_every_batches must be non-negative")
        self.log_every_batches = int(log_every_batches)
        if knot_match_tolerance < 0.0:
            raise ValueError("knot_match_tolerance must be non-negative")
        self.knot_match_tolerance = float(knot_match_tolerance)
        if not 0.0 <= deployment_pass_rate_target <= 1.0:
            raise ValueError("deployment_pass_rate_target must lie in [0, 1]")
        self.deployment_pass_rate_target = float(deployment_pass_rate_target)
        self.show_progress = bool(show_progress)

    @staticmethod
    def _true_knots_in_output_parameterization(
        true_internal_knots: torch.Tensor,
        true_params: torch.Tensor | None,
        output_params: torch.Tensor,
    ) -> torch.Tensor:
        """Transport labelled knots to the domain used by predicted knots."""

        if true_params is None:
            return true_internal_knots
        if true_params.shape != output_params.shape:
            raise ValueError("true and output parameters must share shape [B,M]")
        if (
            true_internal_knots.ndim != 2
            or true_internal_knots.shape[0] != true_params.shape[0]
        ):
            raise ValueError("true internal knots must have shape [B,K]")
        return torch.stack(
            [
                warp_internal_knots_to_parameterization(
                    knot_row,
                    true_row,
                    output_row,
                )
                for knot_row, true_row, output_row in zip(
                    true_internal_knots,
                    true_params,
                    output_params,
                    strict=True,
                )
            ],
            dim=0,
        )

    @staticmethod
    def _format_eta(seconds: float) -> str:
        if seconds == float("inf") or seconds < 0.0:
            return "--:--"
        total_seconds = int(round(seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return (
            f"{hours:d}:{minutes:02d}:{secs:02d}"
            if hours
            else f"{minutes:02d}:{secs:02d}"
        )

    @classmethod
    def _progress_line(
        cls,
        *,
        phase: str,
        completed: int,
        total: int,
        loss: float,
        elapsed: float,
        width: int = 28,
    ) -> str:
        fraction = completed / max(total, 1)
        filled = min(width, max(0, int(round(width * fraction))))
        rate = completed / elapsed if elapsed > 0.0 else 0.0
        eta = (total - completed) / rate if rate > 0.0 else float("inf")
        bar = "#" * filled + "-" * (width - filled)
        return (
            f"  {phase:<10} [{bar}] {completed:>4}/{total:<4} "
            f"{100.0 * fraction:6.2f}% | loss={loss:.6f} | "
            f"{rate:5.2f} batch/s | ETA {cls._format_eta(eta)}"
        )

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
        candidate_required = {
            "candidate_match_count",
            "candidate_target_count",
            "candidate_nearest_error_sum",
        }
        if candidate_required.issubset(metrics):
            matched = metrics["candidate_match_count"]
            target = metrics["candidate_target_count"]
            metrics["candidate_recall"] = matched / target if target > 0.0 else 1.0
            metrics["candidate_nearest_mae"] = (
                metrics["candidate_nearest_error_sum"] / target if target > 0.0 else 0.0
            )
            for key, value in tuple(metrics.items()):
                prefix = "candidate_match_count_at_"
                if key.startswith(prefix):
                    suffix = key[len(prefix) :]
                    metrics[f"candidate_recall_at_{suffix}"] = (
                        value / target if target > 0.0 else 1.0
                    )
        return metrics

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool,
        l0_scale: float,
        activity_scale: float,
        binary_scale: float,
        teacher_forcing_ratio: float = 1.0,
        compute_deployment_metrics: bool = False,
        deployment_use_all_candidates: bool = False,
    ) -> dict[str, float]:
        self.model.train(training)
        totals: dict[str, float] = defaultdict(float)
        deployment_rms_values: list[torch.Tensor] = []

        phase = "train" if training else "validation"
        total_batches = len(loader)
        progress_started_at = time.perf_counter()
        interactive_progress = self.show_progress and sys.stdout.isatty()
        for batch_index, batch in enumerate(loader, start=1):
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
            teacher_kwargs = {
                key: value.to(self.device)
                for key, value in batch.items()
                if key.startswith("teacher_") and isinstance(value, torch.Tensor)
            }

            with torch.set_grad_enabled(training):
                teacher_count = (
                    true_internal_knot_count
                    if training
                    and getattr(self.model, "structure_mode", None)
                    in {"count_conditioned", "interactive_dynamic"}
                    else None
                )
                output = self.model(
                    points,
                    true_internal_knot_count=teacher_count,
                    teacher_forcing_ratio=(teacher_forcing_ratio if training else 0.0),
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
                    **teacher_kwargs,
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
            deployment_metrics: dict[str, torch.Tensor] = {}
            if compute_deployment_metrics:
                if training:
                    raise ValueError("deployment metrics are validation-only")
                if getattr(self.model, "structure_mode", None) != (
                    "candidate_pruning_one_shot"
                ):
                    raise ValueError(
                        "deployment metrics require candidate_pruning_one_shot"
                    )
                if deployment_use_all_candidates:
                    deployment_parameters = output["params"]
                    proposal_parameters = output.get(
                        "proposal_params", deployment_parameters
                    )
                    deployment_knots = output.get(
                        "proposal_internal_knots", output["internal_knots"]
                    )
                    warp = getattr(self.model, "_warp_parameter_coordinates", None)
                    if warp is not None:
                        deployment_knots = warp(
                            deployment_knots,
                            proposal_parameters,
                            deployment_parameters,
                        )
                    hard_mask = torch.ones_like(deployment_knots, dtype=torch.bool)
                else:
                    deployment_parameters = output["params"]
                    deployment_knots = output["internal_knots"]
                    hard_mask = output.get(
                        "learned_keep_mask", output.get("knot_mask")
                    )
                    if hard_mask is None:
                        raise KeyError("one-shot output is missing learned_keep_mask")
                deployed = refit_hard_gated_bspline_batch(
                    parameters=deployment_parameters,
                    candidate_knots=deployment_knots,
                    hard_gates=hard_mask,
                    points=points,
                    degree=int(getattr(self.model, "degree", 3)),
                    smoothness_weight=float(
                        getattr(self.loss_fn, "deletion_smoothness_weight", 1e-6)
                    ),
                    control_ridge=float(
                        getattr(self.loss_fn, "deletion_control_ridge", 0.0)
                    ),
                    interpolate_endpoints=True,
                )
                deployment_rms = torch.stack(
                    [item.fit_rmse.to(points) for item in deployed]
                )
                deployment_rms_values.append(deployment_rms.detach().cpu())
                fit_tolerance = float(getattr(self.loss_fn, "fit_tolerance", 5e-3))
                deployment_metrics = {
                    "deployment_bspline_rms": deployment_rms.mean(),
                    "deployment_retained_knot_count": hard_mask.to(points.dtype)
                    .sum(dim=-1)
                    .mean(),
                    "deployment_threshold_satisfied_rate": (
                        deployment_rms <= fit_tolerance
                    )
                    .to(points.dtype)
                    .mean(),
                }
            geometric_metrics: dict[str, torch.Tensor] = {}
            if true_internal_knots is not None and true_internal_knot_mask is not None:
                metric_true_internal_knots = (
                    self._true_knots_in_output_parameterization(
                        true_internal_knots,
                        true_params,
                        output["params"],
                    )
                )
                matched_count = 0
                predicted_count = 0
                target_count = 0
                matched_error_sum = 0.0
                retained_mask = output.get("knot_mask")
                if retained_mask is None:
                    retained_mask = output["activity"] >= self.activity_threshold
                for sample_index in range(batch_size):
                    match = match_internal_knots(
                        output["internal_knots"][
                            sample_index, retained_mask[sample_index]
                        ],
                        metric_true_internal_knots[
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
                **deployment_metrics,
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

            if interactive_progress:
                progress = self._progress_line(
                    phase=phase,
                    completed=batch_index,
                    total=total_batches,
                    loss=float(losses["loss"].detach().cpu()),
                    elapsed=max(time.perf_counter() - progress_started_at, 1e-9),
                )
                # Padding erases stale characters when ETA or throughput shrinks.
                print(f"\r{progress:<120}", end="", flush=True)
                if batch_index == total_batches:
                    print(flush=True)
            elif self.log_every_batches and (
                batch_index % self.log_every_batches == 0
                or batch_index == total_batches
            ):
                print(
                    f"  {phase} batches: {batch_index}/{total_batches} | "
                    f"loss={float(losses['loss'].detach().cpu()):.6f}",
                    flush=True,
                )

        mean_metrics = self._mean_metrics(totals, len(loader.dataset))
        if deployment_rms_values:
            all_deployment_rms = torch.cat(deployment_rms_values)
            mean_metrics["deployment_bspline_rms_p95"] = float(
                torch.quantile(all_deployment_rms, 0.95)
            )
        return mean_metrics

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
        teacher_forcing_schedule: Callable[[int], float] | None = None,
        checkpoint_selection_start_epoch: int = 0,
        checkpoint_path: str | Path | None = None,
        epoch_offset: int = 0,
        stage_name: str = "joint",
        deployment_validation: bool = False,
    ) -> list[dict[str, float]]:
        if checkpoint_selection_start_epoch < 0:
            raise ValueError("checkpoint_selection_start_epoch must be non-negative")
        history: list[dict[str, float]] = []
        best_knot_match_f1 = float("-inf")
        best_knot_match_precision = float("-inf")
        best_knot_matched_mae = float("inf")
        best_rank: tuple[float, ...] | None = None

        for epoch in range(epochs):
            dataset = train_loader.dataset
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch_offset + epoch)
            l0_scale = l0_schedule(epoch) if l0_schedule else 1.0
            activity_scale = activity_schedule(epoch) if activity_schedule else 1.0
            binary_scale = binary_schedule(epoch) if binary_schedule else 1.0
            gate_temperature = (
                gate_temperature_schedule(epoch)
                if gate_temperature_schedule is not None
                else None
            )
            teacher_forcing_ratio = (
                teacher_forcing_schedule(epoch)
                if teacher_forcing_schedule is not None
                else 1.0
            )
            if not 0.0 <= teacher_forcing_ratio <= 1.0:
                raise ValueError(
                    "teacher forcing schedule must return a value in [0, 1]"
                )
            if gate_temperature is not None and hasattr(
                self.model, "set_gate_temperature"
            ):
                self.model.set_gate_temperature(gate_temperature)
            if hasattr(self.model, "set_force_open_gates"):
                self.model.set_force_open_gates(epoch < gate_warmup_epochs)
            train_metrics = self._run_epoch(
                train_loader,
                True,
                l0_scale,
                activity_scale,
                binary_scale,
                teacher_forcing_ratio,
            )

            record = {
                **{f"train/{k}": v for k, v in train_metrics.items()},
                "train/teacher_forcing_ratio": teacher_forcing_ratio,
            }
            if val_loader is not None:
                val_metrics = self._run_epoch(
                    val_loader,
                    False,
                    l0_scale,
                    1.0,
                    1.0,
                    compute_deployment_metrics=deployment_validation,
                    deployment_use_all_candidates=(stage_name == "candidate_pretrain"),
                )
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
            structure_mode = getattr(self.model, "structure_mode", None)
            parameter_feedback_stage = (
                stage_name == "parameter_feedback_calibration"
                or "parameter_feedback_chord_blend_weight" in displayed_metrics
            )
            if parameter_feedback_stage:
                # Parameter feedback freezes proposal/selection/relocation, so its
                # loss intentionally has no candidate-coverage or knot-position
                # terms.  Report the quantities this stage actually optimizes
                # instead of pretending the missing structural losses are zero.
                if "deployment_threshold_satisfied_rate" in displayed_metrics:
                    fit_report = (
                        f"deploy_pass="
                        f"{displayed_metrics['deployment_threshold_satisfied_rate']:.3f}"
                        f" | deploy_RMS="
                        f"{displayed_metrics['deployment_bspline_rms']:.5f}"
                    )
                else:
                    fit_report = (
                        f"surrogate_pass="
                        f"{displayed_metrics.get('threshold_satisfied_rate', 0.0):.3f}"
                    )
                has_parameter_supervision = (
                    displayed_metrics.get(
                        "true_parameter_supervision_fraction",
                        1.0,
                    )
                    > 0.0
                )
                parameter_report = (
                    f"{max(displayed_metrics.get('true_parameter_loss', 0.0), 0.0) ** 0.5:.5f}"
                    if has_parameter_supervision
                    else "n/a"
                )
                structure_report = (
                    f"{fit_report} | "
                    f"keep={displayed_metrics['hard_active_count']:.2f}/"
                    f"{displayed_metrics['candidate_knot_count']:.0f} | "
                    f"parameter_RMSE={parameter_report} | "
                    f"chord_blend="
                    f"{displayed_metrics.get('parameter_feedback_chord_blend_weight', 0.0):.3f} | "
                    f"gap_shift="
                    f"{displayed_metrics.get('parameter_feedback_gap_logit_shift_mean_abs', 0.0):.4f}"
                )
                if "deployment_fit_loss" in displayed_metrics:
                    structure_report += (
                        f" | exact_refit_MSE="
                        f"{displayed_metrics['deployment_fit_loss']:.6e}"
                        f" | count_score="
                        f"{displayed_metrics.get('joint_requested_count_mean', 0.0):.2f}/"
                        f"{displayed_metrics.get('joint_target_policy_score_mean', 0.0):.2f}"
                        f"(K={displayed_metrics.get('joint_target_count_mean', 0.0):.2f})"
                    )
            elif structure_mode in {
                "candidate_pruning",
                "candidate_pruning_one_shot",
            }:
                structure_report = (
                    f"coverage={displayed_metrics['candidate_coverage_loss']:.4f} | "
                    f"candidate_R@{self.knot_match_tolerance:.3f}="
                    f"{displayed_metrics.get('candidate_recall', 0.0):.3f} | "
                    f"nearest_MAE={displayed_metrics.get('candidate_nearest_mae', 0.0):.4f} | "
                    f"safe_action={displayed_metrics.get('safe_action_top1', 0.0):.3f} | "
                    f"false_STOP={displayed_metrics.get('false_stop_rate', 0.0):.3f} | "
                    f"keep={displayed_metrics['hard_active_count']:.2f}/"
                    f"{displayed_metrics['candidate_knot_count']:.0f}"
                )
                if structure_mode == "candidate_pruning_one_shot":
                    if "deployment_threshold_satisfied_rate" in displayed_metrics:
                        structure_report += (
                            f" | deploy_pass="
                            f"{displayed_metrics['deployment_threshold_satisfied_rate']:.3f}"
                            f" | deploy_RMS="
                            f"{displayed_metrics['deployment_bspline_rms']:.5f}"
                            f" | P95="
                            f"{displayed_metrics.get('deployment_bspline_rms_p95', displayed_metrics['deployment_bspline_rms']):.5f}"
                        )
                    else:
                        structure_report += (
                            f" | surrogate_pass="
                            f"{displayed_metrics.get('surrogate_threshold_satisfied_rate', 0.0):.3f}"
                        )
                    structure_report += (
                        f" | mask_acc={displayed_metrics.get('teacher_mask_accuracy', 0.0):.3f}"
                        f" | beta={displayed_metrics.get('adaptive_keep_threshold_mean', 0.0):.3f}"
                    )
                    if "predicted_relocation_mean_abs" in displayed_metrics:
                        structure_report += (
                            f" | reloc={displayed_metrics['predicted_relocation_mean_abs']:.4f}"
                            f" | gap={displayed_metrics.get('teacher_survivor_spacing_loss', 0.0):.4f}"
                        )
            elif structure_mode in {
                "count_conditioned",
                "interactive_dynamic",
            }:
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
            if "knot_position_loss" in displayed_metrics:
                position_report = (
                    f"knot_pos={displayed_metrics['knot_position_loss']:.4f}"
                )
            elif (
                "true_parameter_loss" in displayed_metrics
                and displayed_metrics.get(
                    "true_parameter_supervision_fraction",
                    1.0,
                )
                > 0.0
            ):
                position_report = (
                    f"parameter_MSE={displayed_metrics['true_parameter_loss']:.6f}"
                )
            else:
                position_report = "position_loss=n/a"
            print(
                f"Stage {stage_name} | "
                f"Epoch {epoch_offset + epoch + 1:04d} "
                f"(stage {epoch + 1:04d}/{epochs:04d}) | "
                f"train={train_metrics['loss']:.6f} | val={current_val:.6f} | "
                f"fit={displayed_metrics['fit_loss']:.6f} | "
                f"{structure_report} | "
                f"knot_F1@{self.knot_match_tolerance:.3f}="
                f"{displayed_metrics.get('knot_match_f1', 0.0):.3f} | "
                f"{position_report} | "
                f"teacher={teacher_forcing_ratio:.3f} | "
                f"l0_scale={l0_scale:.3f} | "
                f"temperature={gate_temperature if gate_temperature is not None else float('nan'):.3f}"
            )

            structure_mode = getattr(self.model, "structure_mode", None)
            structured_count_model = structure_mode in {
                "count_conditioned",
                "interactive_dynamic",
            }
            if stage_name == "proposal_parameter_warmup":
                # The warm-up caller gives the loss only its coordinate-MSE
                # and log-gap terms.  Select by that complete weighted
                # objective first: coordinate MSE alone can hide large local
                # interval errors that cancel after cumulative integration.
                current_gap_mae = selection_metrics.get(
                    "proposal_parameter_log_gap_mae",
                    float("inf"),
                )
                current_parameter_loss = selection_metrics.get(
                    "raw_proposal_parameter_loss",
                    float("inf"),
                )
                if current_gap_mae != current_gap_mae:
                    current_gap_mae = float("inf")
                if current_parameter_loss != current_parameter_loss:
                    current_parameter_loss = float("inf")
                current_rank = (
                    -current_val,
                    -current_gap_mae,
                    -current_parameter_loss,
                )
                selection_metric_name = (
                    "parameter_warmup_val_loss_then_log_gap_mae_coordinate_mse"
                )
                selection_value = current_val
            elif structure_mode in {
                "candidate_pruning",
                "candidate_pruning_one_shot",
            }:
                current_candidate_recall = selection_metrics.get(
                    "candidate_recall", 0.0
                )
                current_candidate_mae = selection_metrics.get(
                    "candidate_nearest_mae", float("inf")
                )
                if current_candidate_recall != current_candidate_recall:
                    current_candidate_recall = 0.0
                if current_candidate_mae != current_candidate_mae:
                    current_candidate_mae = float("inf")
                current_safe_action = selection_metrics.get("safe_action_top1", 0.0)
                current_unsafe_action = selection_metrics.get("unsafe_delete_rate", 1.0)
                current_false_stop = selection_metrics.get("false_stop_rate", 1.0)
                if stage_name == "candidate_pretrain":
                    current_pass_rate = selection_metrics.get(
                        "deployment_threshold_satisfied_rate", 0.0
                    )
                    current_deployment_rms = selection_metrics.get(
                        "deployment_bspline_rms", float("inf")
                    )
                    current_deployment_rms_p95 = selection_metrics.get(
                        "deployment_bspline_rms_p95", float("inf")
                    )
                    strict_recall = selection_metrics.get(
                        "candidate_recall_at_0p005", current_candidate_recall
                    )
                    medium_recall = selection_metrics.get(
                        "candidate_recall_at_0p01", current_candidate_recall
                    )
                    broad_recall = selection_metrics.get(
                        "candidate_recall_at_0p02", current_candidate_recall
                    )
                    proposal_feasible = (
                        current_pass_rate >= self.deployment_pass_rate_target
                    )
                    if proposal_feasible:
                        # Once every-candidate deployment is sufficiently
                        # feasible, preserve the reason for this stage: a
                        # high-recall proposal set.  Ranking feasible epochs by
                        # ever-smaller RMS first previously selected candidates
                        # that fit globally but missed labelled local knots.
                        current_rank = (
                            1.0,
                            strict_recall,
                            medium_recall,
                            broad_recall,
                            -current_candidate_mae,
                            -current_deployment_rms,
                            -current_deployment_rms_p95,
                            current_knot_match_f1,
                            -current_val,
                        )
                    else:
                        # A high-recall but infeasible proposal cannot seed a
                        # valid Hard-RMS teacher.  Reach the exact deployment
                        # constraint before optimizing localization quality.
                        current_rank = (
                            0.0,
                            current_pass_rate,
                            -current_deployment_rms,
                            -current_deployment_rms_p95,
                            strict_recall,
                            medium_recall,
                            broad_recall,
                            -current_candidate_mae,
                            current_knot_match_f1,
                            -current_val,
                        )
                    selection_metric_name = (
                        "all_candidate_standard_bspline_feasible_then_"
                        "candidate_recall_0p005_0p01_0p02_mae_rms_p95_loss"
                    )
                    selection_value = (
                        strict_recall if proposal_feasible else current_pass_rate
                    )
                elif structure_mode == "candidate_pruning_one_shot":
                    current_pass_rate = selection_metrics.get(
                        "deployment_threshold_satisfied_rate", 0.0
                    )
                    current_deployment_rms = selection_metrics.get(
                        "deployment_bspline_rms", float("inf")
                    )
                    current_deployment_rms_p95 = selection_metrics.get(
                        "deployment_bspline_rms_p95", float("inf")
                    )
                    current_mask_accuracy = selection_metrics.get(
                        "teacher_mask_accuracy", 0.0
                    )
                    current_mask_f1 = selection_metrics.get("existence_f1", 0.0)
                    current_count_mae = selection_metrics.get(
                        "count_absolute_error", float("inf")
                    )
                    current_retained_count = selection_metrics.get(
                        "deployment_retained_knot_count",
                        selection_metrics.get("hard_active_count", float("inf")),
                    )
                    pass_constraint_satisfied = (
                        current_pass_rate >= self.deployment_pass_rate_target
                    )
                    fixed_proposal = bool(
                        getattr(
                            getattr(self.model, "pruning_head", None),
                            "one_shot_fixed_proposal_geometry",
                            False,
                        )
                    )
                    if fixed_proposal:
                        # Fixed-proposal one-shot models select only with real
                        # standard-B-spline
                        # deployment quantities.  Before feasibility it
                        # improves pass/RMS/P95.  Once the target pass rate is
                        # reached, the original minimum-complexity objective
                        # takes over so an all-keep checkpoint cannot win just
                        # by over-satisfying the fit constraint.
                        if pass_constraint_satisfied:
                            current_rank = (
                                1.0,
                                -current_retained_count,
                                current_pass_rate,
                                -current_deployment_rms,
                                -current_deployment_rms_p95,
                                current_knot_match_f1,
                                current_knot_match_precision,
                                -current_knot_matched_mae,
                                current_mask_f1,
                                -current_count_mae,
                                -current_val,
                            )
                        else:
                            current_rank = (
                                0.0,
                                current_pass_rate,
                                -current_deployment_rms,
                                -current_deployment_rms_p95,
                                -current_retained_count,
                                current_knot_match_f1,
                                current_knot_match_precision,
                                -current_knot_matched_mae,
                                current_mask_f1,
                                -current_count_mae,
                                -current_val,
                            )
                        structured_policy = getattr(
                            getattr(self.model, "pruning_head", None),
                            "one_shot_selection_policy",
                            "threshold",
                        )
                        selection_metric_name = (
                            "one_shot_structured_feasible_standard_bspline_"
                            "min_knots_rms_p95_recall"
                            if structured_policy == "mass_topk"
                            else "v9_constrained_standard_bspline_"
                            "min_knots_rms_p95_recall"
                        )
                        selection_value = (
                            current_retained_count
                            if pass_constraint_satisfied
                            else current_pass_rate
                        )
                    elif pass_constraint_satisfied:
                        # Constrained deployment objective: once the required
                        # standard-B-spline pass rate is met, fewer knots are
                        # always preferred.  This prevents the former
                        # all-candidates solution from winning merely by
                        # increasing its pass rate by a few tenths of a percent.
                        current_rank = (
                            1.0,
                            -current_retained_count,
                            current_mask_f1,
                            -current_count_mae,
                            -current_deployment_rms,
                            current_mask_accuracy,
                            current_candidate_recall,
                            -current_candidate_mae,
                            current_knot_match_f1,
                            -current_val,
                        )
                    else:
                        # Until feasibility is reached, improve the real
                        # deployment pass rate first.  Complexity becomes the
                        # next tie-breaker, never a substitute for feasibility.
                        current_rank = (
                            0.0,
                            current_pass_rate,
                            -current_deployment_rms,
                            -current_retained_count,
                            current_mask_f1,
                            -current_count_mae,
                            current_mask_accuracy,
                            current_candidate_recall,
                            -current_candidate_mae,
                            current_knot_match_f1,
                            -current_val,
                        )
                    if not fixed_proposal:
                        selection_metric_name = (
                            "constrained_min_knots_at_target_pass_then_mask_count_rms"
                        )
                        selection_value = (
                            current_retained_count
                            if pass_constraint_satisfied
                            else current_pass_rate
                        )
                else:
                    current_rank = (
                        current_candidate_recall,
                        -current_candidate_mae,
                        current_safe_action,
                        -current_unsafe_action,
                        -current_false_stop,
                        current_knot_match_f1,
                        current_knot_match_precision,
                        -current_knot_matched_mae,
                        -current_val,
                    )
                    selection_metric_name = (
                        "candidate_recall_mae_then_safe_action_knot_f1_loss"
                    )
                    selection_value = current_candidate_recall
            elif structured_count_model:
                current_count_mae = selection_metrics.get(
                    "count_absolute_error", float("inf")
                )
                current_count_accuracy = selection_metrics.get("count_accuracy", 0.0)
                if current_count_mae != current_count_mae:
                    current_count_mae = float("inf")
                if current_count_accuracy != current_count_accuracy:
                    current_count_accuracy = 0.0
                current_rank = (
                    -current_count_mae,
                    current_count_accuracy,
                    current_knot_match_f1,
                    current_knot_match_precision,
                    -current_knot_matched_mae,
                    -current_val,
                )
                selection_metric_name = (
                    "count_mae_then_accuracy_knot_f1_precision_mae_loss"
                )
                selection_value = current_count_mae
            else:
                current_rank = (
                    current_knot_match_f1,
                    current_knot_match_precision,
                    -current_knot_matched_mae,
                    -current_val,
                )
                selection_metric_name = "knot_match_f1_then_precision_mae_loss"
                selection_value = current_knot_match_f1
            if (
                checkpoint_path is not None
                and epoch >= checkpoint_selection_start_epoch
                and (best_rank is None or current_rank > best_rank)
            ):
                best_rank = current_rank
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
                        "selection_metric": selection_metric_name,
                        "selection_value": selection_value,
                        "selection_rank": list(current_rank),
                        "best_knot_match_f1": best_knot_match_f1,
                        "best_knot_match_precision": best_knot_match_precision,
                        "best_knot_matched_mae": best_knot_matched_mae,
                        "knot_match_tolerance": self.knot_match_tolerance,
                        "best_existence_f1": selection_metrics.get("existence_f1", 0.0),
                        "best_count_accuracy": selection_metrics.get(
                            "count_accuracy", 0.0
                        ),
                        "best_count_mae": selection_metrics.get(
                            "count_absolute_error", float("nan")
                        ),
                        "best_candidate_recall": selection_metrics.get(
                            "candidate_recall", float("nan")
                        ),
                        "best_candidate_nearest_mae": selection_metrics.get(
                            "candidate_nearest_mae", float("nan")
                        ),
                        "best_safe_action_top1": selection_metrics.get(
                            "safe_action_top1", float("nan")
                        ),
                        "best_false_stop_rate": selection_metrics.get(
                            "false_stop_rate", float("nan")
                        ),
                        "best_threshold_satisfied_rate": selection_metrics.get(
                            "deployment_threshold_satisfied_rate", float("nan")
                        ),
                        "best_deployment_bspline_rms": selection_metrics.get(
                            "deployment_bspline_rms", float("nan")
                        ),
                        "best_deployment_bspline_rms_p95": selection_metrics.get(
                            "deployment_bspline_rms_p95", float("nan")
                        ),
                        "best_deployment_retained_knot_count": (
                            selection_metrics.get(
                                "deployment_retained_knot_count", float("nan")
                            )
                        ),
                        "deployment_pass_rate_target": (
                            self.deployment_pass_rate_target
                        ),
                        "best_deployment_pass_constraint_satisfied": (
                            selection_metrics.get(
                                "deployment_threshold_satisfied_rate", 0.0
                            )
                            >= self.deployment_pass_rate_target
                        ),
                        "best_teacher_mask_accuracy": selection_metrics.get(
                            "teacher_mask_accuracy", float("nan")
                        ),
                        "best_teacher_mask_f1": selection_metrics.get(
                            "existence_f1", float("nan")
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
