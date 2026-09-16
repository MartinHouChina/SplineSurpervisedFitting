"""Offline feasible-subset labels for the frozen v16 proposal frame.

The proposal encoder, parameter head and candidate head must stay frozen
throughout joint training.  This cache searches hard, standard B-spline refits
on *their* predicted parameters/candidates, rather than treating a synthetic
source knot count as a feasible cardinality in the predicted frame.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..evaluation.bspline_inference import refit_bspline_control_points
from ..evaluation.knot_diagnostics import warp_internal_knots_to_parameterization
from ..spline.bspline_deletion_teacher import single_knot_deletion_rmse_batch
from .one_shot_teacher import (
    OneShotTeacherBatch,
    OneShotTeacherConfig,
    build_one_shot_teacher_batch,
    load_one_shot_teacher_cache,
    save_one_shot_teacher_cache,
)


_PROPOSAL_PREFIXES = ("encoder.", "parameter_head.", "candidate_head.")
_ANCHOR_STRATEGY_VERSION = "synthetic_anchor_first_v2_residual_addback_safe_cleanup"
_TEACHER_STRATEGIES = {"full_greedy", "synthetic_anchor_first"}


def _devices_equivalent(actual: torch.device, requested: torch.device) -> bool:
    """Compare effective devices, including bare `cuda` = current CUDA index."""
    if actual.type != requested.type:
        return False
    if actual.type != "cuda":
        return actual == requested
    actual_index = (
        torch.cuda.current_device() if actual.index is None else actual.index
    )
    requested_index = (
        torch.cuda.current_device() if requested.index is None else requested.index
    )
    return actual_index == requested_index


def _proposal_fingerprint(model: torch.nn.Module) -> str:
    """Hash only weights that determine proposal parameters and knots.

    Selector and subset-decoder weights are deliberately excluded: they can
    change during Joint without invalidating labels from a frozen Proposal.
    """
    state = model.state_dict()
    proposal_names = [
        name for name in state if name.startswith(_PROPOSAL_PREFIXES)
    ]
    if not proposal_names:
        # A minimal test double may have no v16 module names.  Do not allow an
        # empty fingerprint to silently reuse a cache after its output changes.
        proposal_names = list(state)
    if not proposal_names:
        raise ValueError("proposal model has no tensor state to fingerprint")
    digest = hashlib.sha256()
    for name in sorted(proposal_names):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    for name in (
        "point_dim", "degree", "max_internal_knots", "min_parameter_gap",
        "min_knot_gap",
    ):
        if hasattr(model, name):
            digest.update(f"{name}={getattr(model, name)!r}".encode("utf-8"))
    return digest.hexdigest()


def _fixed_dataset_points(
    dataset: Dataset, *, teacher_strategy: str = "full_greedy",
    anchor_match_tolerance: float = 0.02,
    anchor_fallback_to_greedy: bool = True,
) -> tuple[torch.Tensor, str, list[tuple[torch.Tensor, torch.Tensor]] | None]:
    """Materialize and fingerprint the exact ordered points in index order."""
    if bool(getattr(dataset, "resample_each_epoch", False)):
        raise ValueError("offline teacher requires a fixed, non-resampling dataset")
    if len(dataset) < 1:
        raise ValueError("offline teacher requires at least one sample")
    digest = hashlib.sha256()
    points: list[torch.Tensor] = []
    anchors: list[tuple[torch.Tensor, torch.Tensor]] = []
    if teacher_strategy == "synthetic_anchor_first":
        digest.update(_ANCHOR_STRATEGY_VERSION.encode("ascii"))
        digest.update(repr(anchor_match_tolerance).encode("ascii"))
        digest.update(repr(anchor_fallback_to_greedy).encode("ascii"))
    expected_shape: tuple[int, int] | None = None
    for index in range(len(dataset)):
        sample = dataset[index]
        if not isinstance(sample, Mapping):
            raise TypeError("offline teacher dataset rows must be mappings")
        if sample.get("source") != "Synthetic":
            raise ValueError(
                "v16 feasible teacher training is synthetic-only; "
                f"sample {index} came from {sample.get('source')!r}"
            )
        value = sample.get("points")
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise ValueError(f"sample {index} points must be a [M,D] tensor")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"sample {index} points must be finite floating values")
        point = value.detach().cpu().contiguous()
        shape = tuple(point.shape)
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError("all fixed teacher samples must share [M,D]")
        digest.update(index.to_bytes(8, "little", signed=False))
        digest.update(str(point.dtype).encode("ascii"))
        digest.update(str(shape).encode("ascii"))
        digest.update(point.numpy().tobytes())
        points.append(point)
        if teacher_strategy == "synthetic_anchor_first":
            if not bool(torch.as_tensor(sample.get("target_geometry_valid", False))):
                raise ValueError(
                    "anchor-first teacher requires certified synthetic geometry "
                    f"labels at sample {index}"
                )
            parameter = sample.get("target_params")
            knots = sample.get("target_internal_knots")
            mask = sample.get("target_internal_knot_mask")
            if (
                not isinstance(parameter, torch.Tensor)
                or parameter.shape != (point.shape[0],)
                or not isinstance(knots, torch.Tensor)
                or knots.ndim != 1
                or not isinstance(mask, torch.Tensor)
                or mask.shape != knots.shape
                or mask.dtype != torch.bool
            ):
                raise ValueError(f"sample {index} has invalid certified anchor labels")
            if (
                not parameter.is_floating_point()
                or not knots.is_floating_point()
                or not torch.isfinite(parameter).all()
                or not torch.isfinite(knots[mask]).all()
                or bool((parameter[1:] <= parameter[:-1]).any())
            ):
                raise ValueError(f"sample {index} has invalid anchor parameterization")
            true_knots = knots[mask].detach().cpu().contiguous()
            if true_knots.numel() and (
                bool((true_knots[1:] <= true_knots[:-1]).any())
                or bool((true_knots <= parameter[0]).any())
                or bool((true_knots >= parameter[-1]).any())
            ):
                raise ValueError(f"sample {index} anchors must be ordered and internal")
            parameter_cpu = parameter.detach().cpu().contiguous()
            for name, value in (("target_params", parameter_cpu), ("anchors", true_knots)):
                digest.update(name.encode("ascii"))
                digest.update(str(value.dtype).encode("ascii"))
                digest.update(str(tuple(value.shape)).encode("ascii"))
                digest.update(value.numpy().tobytes())
            anchors.append((parameter_cpu, true_knots))
    return torch.stack(points), digest.hexdigest(), (
        anchors if teacher_strategy == "synthetic_anchor_first" else None
    )


def _ordered_anchor_slots(
    candidates: torch.Tensor,
    anchors: torch.Tensor,
    *,
    max_distance: float,
) -> list[int] | None:
    """Minimum-distance order-preserving, one-to-one source-to-proposal match."""
    candidate_count = int(candidates.numel())
    anchor_count = int(anchors.numel())
    if anchor_count > candidate_count:
        return None
    if anchor_count == 0:
        return []
    # Dynamic programming prevents two source anchors from claiming the same
    # proposal slot, a common failure of independent nearest-neighbour labels.
    values = candidates.detach().cpu().double().tolist()
    targets = anchors.detach().cpu().double().tolist()
    costs = [[math.inf] * (candidate_count + 1) for _ in range(anchor_count + 1)]
    takes = [[False] * (candidate_count + 1) for _ in range(anchor_count + 1)]
    for candidate_index in range(candidate_count + 1):
        costs[0][candidate_index] = 0.0
    for anchor_index in range(1, anchor_count + 1):
        for candidate_index in range(1, candidate_count + 1):
            skip_cost = costs[anchor_index][candidate_index - 1]
            distance = abs(targets[anchor_index - 1] - values[candidate_index - 1])
            take_cost = (
                costs[anchor_index - 1][candidate_index - 1] + distance
                if distance <= max_distance else math.inf
            )
            if take_cost < skip_cost:
                costs[anchor_index][candidate_index] = take_cost
                takes[anchor_index][candidate_index] = True
            else:
                costs[anchor_index][candidate_index] = skip_cost
    if not math.isfinite(costs[anchor_count][candidate_count]):
        return None
    slots: list[int] = []
    anchor_index = anchor_count
    candidate_index = candidate_count
    while anchor_index:
        if takes[anchor_index][candidate_index]:
            slots.append(candidate_index - 1)
            anchor_index -= 1
        candidate_index -= 1
    return list(reversed(slots))


def _next_residual_candidate(
    candidates: torch.Tensor,
    selected: set[int],
    parameters: torch.Tensor,
    points: torch.Tensor,
    reconstructed: torch.Tensor,
) -> int:
    """Choose one slot near current residual, penalizing near-duplicate knots.

    This uses no trial solves: only a [K, M] local Gaussian reduction and
    then one refit for the chosen slot. The distance term discourages repeatedly
    inserting tightly clustered candidates around an existing survivor.
    """
    residual_squared = (reconstructed - points).square().sum(-1)
    distance = (candidates[:, None] - parameters[None, :]).abs()
    bandwidth = 0.03
    weight = torch.exp(-0.5 * (distance / bandwidth).square())
    local_error = (weight @ residual_squared) / weight.sum(-1).clamp_min(1e-12)
    if selected:
        selected_knots = candidates[sorted(selected)]
        spacing = (candidates[:, None] - selected_knots[None, :]).abs().amin(-1)
        score = local_error * (spacing / 0.025).clamp(min=0.1, max=1.0)
    else:
        score = local_error
    score[list(selected)] = -math.inf
    if bool(torch.isfinite(score).any()) and float(score.max()) > 0:
        return int(score.argmax())
    # Degenerate zero-residual tie: deterministic largest uncovered gap.
    positions = candidates.detach().cpu().double().tolist()
    occupied = [positions[index] for index in sorted(selected)] + [0.0, 1.0]
    return max(
        (index for index in range(len(positions)) if index not in selected),
        key=lambda index: (
            min(abs(positions[index] - position) for position in occupied),
            -index,
        ),
    )


@torch.no_grad()
def _anchor_first_teacher_batch(
    parameters: torch.Tensor,
    points: torch.Tensor,
    candidate_knots: torch.Tensor,
    anchor_labels: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    sample_indices: torch.Tensor,
    config: OneShotTeacherConfig,
    anchor_match_tolerance: float,
    anchor_fallback_to_greedy: bool,
) -> OneShotTeacherBatch:
    """Fit source-aligned slots, then add slots until the hard RMS bound holds.

    This is an opt-in synthetic pilot.  It certifies the *selected proposal
    subset* numerically, but does not prove global minimum cardinality.
    """
    if len(anchor_labels) != parameters.shape[0]:
        raise ValueError("anchor label rows must match the proposal batch")
    batch_rows: dict[str, list[torch.Tensor]] = {}
    candidate_count = int(candidate_knots.shape[1])
    for row_index, (source_params_cpu, source_knots_cpu) in enumerate(anchor_labels):
        row_params = parameters[row_index]
        row_points = points[row_index]
        row_candidates = candidate_knots[row_index]
        source_params = source_params_cpu.to(row_params)
        source_knots = source_knots_cpu.to(row_params)
        warped = warp_internal_knots_to_parameterization(
            source_knots, source_params, row_params,
        )
        matched_slots = _ordered_anchor_slots(
            row_candidates, warped, max_distance=anchor_match_tolerance,
        )
        if matched_slots is None:
            if not anchor_fallback_to_greedy:
                raise ValueError(
                    f"sample {int(sample_indices[row_index])} cannot uniquely "
                    "match certified anchors within the requested tolerance"
                )
            fallback = build_one_shot_teacher_batch(
                row_params.unsqueeze(0),
                row_points.unsqueeze(0),
                row_candidates.unsqueeze(0),
                sample_indices=sample_indices[row_index : row_index + 1],
                config=config,
            )
            row_labels = fallback.as_loss_kwargs()
        else:
            selected = set(matched_slots)
            min_count = config.min_internal_knots
            fit = None
            while True:
                current_slots = sorted(selected)
                current_knots = row_candidates[current_slots]
                fit = refit_bspline_control_points(
                    row_params, row_points, current_knots,
                    degree=config.degree,
                    smoothness_weight=config.smoothness_weight,
                    control_ridge=config.control_ridge,
                    interpolate_endpoints=True,
                    rcond=config.rcond,
                )
                if (
                    len(selected) >= min_count
                    and float(fit.fit_rmse) <= config.error_tolerance
                ) or len(selected) == candidate_count:
                    break
                selected.add(_next_residual_candidate(
                    row_candidates, selected, row_params, row_points,
                    fit.reconstructed_points,
                ))
            assert fit is not None
            # A single low-risk deletion sweep trims additions made before
            # the residual profile changed. Each accepted deletion is checked
            # by the actual endpoint-constrained spline refit; there is no
            # combinatorial Proposal-wide greedy search here.
            if (
                len(selected) > min_count
                and float(fit.fit_rmse) <= config.error_tolerance
            ):
                cleanup_slots = sorted(selected)
                cleanup_rms = single_knot_deletion_rmse_batch(
                    row_params.unsqueeze(0),
                    row_points.unsqueeze(0),
                    row_candidates[cleanup_slots].unsqueeze(0),
                    degree=config.degree,
                    smoothness_weight=config.smoothness_weight,
                    control_ridge=config.control_ridge,
                    interpolate_endpoints=True,
                    rcond=config.rcond,
                )[0]
                removal_order = sorted(
                    range(len(cleanup_slots)),
                    key=lambda index: (float(cleanup_rms[index]), cleanup_slots[index]),
                )
                for local_index in removal_order:
                    if len(selected) <= min_count:
                        break
                    slot = cleanup_slots[local_index]
                    if slot not in selected:
                        continue
                    # A deletion safe in the original subset may not remain
                    # safe after earlier deletions, so always refit the live set.
                    trial_slots = sorted(selected - {slot})
                    trial_fit = refit_bspline_control_points(
                        row_params, row_points, row_candidates[trial_slots],
                        degree=config.degree,
                        smoothness_weight=config.smoothness_weight,
                        control_ridge=config.control_ridge,
                        interpolate_endpoints=True,
                        rcond=config.rcond,
                    )
                    if float(trial_fit.fit_rmse) <= config.error_tolerance:
                        selected.remove(slot)
                        fit = trial_fit
            current_slots = sorted(selected)
            count = len(current_slots)
            retained = torch.zeros_like(row_candidates, dtype=torch.bool)
            retained[current_slots] = True
            packed = torch.zeros_like(row_candidates)
            packed[:count] = row_candidates[current_slots]
            packed_mask = torch.zeros_like(retained)
            packed_mask[:count] = True
            risk = torch.zeros_like(row_candidates)
            if count:
                if count <= min_count:
                    keep_risk = row_candidates.new_ones(count)
                else:
                    leave_one_out = single_knot_deletion_rmse_batch(
                        row_params.unsqueeze(0),
                        row_points.unsqueeze(0),
                        row_candidates[current_slots].unsqueeze(0),
                        degree=config.degree,
                        smoothness_weight=config.smoothness_weight,
                        control_ridge=config.control_ridge,
                        interpolate_endpoints=True,
                        rcond=config.rcond,
                    )[0]
                    margin = leave_one_out / config.error_tolerance - 1.0
                    keep_risk = torch.sigmoid(margin / config.temperature).clamp_min(0.5)
                risk[current_slots] = keep_risk
            # Initial all-candidate leave-one-out costs and deletion order are
            # not needed by the v16 loss; zeros/-1 explicitly indicate this
            # anchor path did not run a full proposal-wide greedy search.
            row_labels = {
                "teacher_retained_mask": retained.unsqueeze(0),
                "teacher_soft_keep_risk": risk.unsqueeze(0),
                "teacher_internal_knots": packed.unsqueeze(0),
                "teacher_internal_knot_mask": packed_mask.unsqueeze(0),
                "teacher_count": torch.tensor([count], dtype=torch.long, device=row_candidates.device),
                "teacher_fit_rms": fit.fit_rmse.reshape(1),
                "teacher_fit_mse": fit.fit_mse.reshape(1),
                "teacher_threshold_satisfied": torch.tensor(
                    [float(fit.fit_rmse) <= config.error_tolerance],
                    dtype=torch.bool, device=row_points.device,
                ),
                "teacher_deletion_order": torch.full(
                    (1, candidate_count), -1, dtype=torch.long, device=row_candidates.device,
                ),
                "teacher_single_deletion_rms": row_candidates.new_zeros((1, candidate_count)),
                "teacher_greedy_count": torch.tensor([count], dtype=torch.long, device=row_candidates.device),
                "teacher_greedy_fit_rms": fit.fit_rmse.reshape(1),
                "teacher_relocation_mean_abs": row_points.new_zeros(1),
                "teacher_relocation_max_abs": row_points.new_zeros(1),
                "teacher_extra_deleted_after_relocation": torch.zeros(
                    1, dtype=torch.long, device=row_candidates.device,
                ),
            }
        for name, value in row_labels.items():
            batch_rows.setdefault(name, []).append(value)
    labels = {name: torch.cat(rows, dim=0) for name, rows in batch_rows.items()}
    return OneShotTeacherBatch(
        sample_indices=sample_indices,
        input_shapes={
            "parameters": tuple(parameters.shape),
            "points": tuple(points.shape),
            "candidate_knots": tuple(candidate_knots.shape),
        },
        config=config,
        **labels,
    )


@dataclass(frozen=True)
class V16FeasibleTeacherCache:
    """Indexed, CPU-resident teacher labels for a fixed Joint population."""

    batch: OneShotTeacherBatch
    path: Path
    loaded: bool
    teacher_strategy: str = "full_greedy"

    @property
    def labels(self) -> dict[str, torch.Tensor]:
        return self.batch.as_loss_kwargs()

    @property
    def sample_count(self) -> int:
        return int(self.batch.sample_indices.numel())

    @property
    def candidate_count(self) -> int:
        return int(self.batch.input_shapes["candidate_knots"][1])

    @property
    def feasible_fraction(self) -> float:
        return float(self.batch.teacher_threshold_satisfied.float().mean())

    def assert_proposal_unchanged(self, model: torch.nn.Module) -> None:
        """Fail fast if Joint has moved the weights behind cached slot labels."""
        if _proposal_fingerprint(model) != self.batch.config.proposal_fingerprint:
            raise RuntimeError(
                "frozen Proposal changed after feasible teacher generation; "
                "cached candidate-slot labels are no longer aligned"
            )

    def labels_for_indices(
        self,
        indices: torch.Tensor | Sequence[int],
        *,
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Gather rows by dataset position, preserving requested batch order."""
        index = torch.as_tensor(indices, dtype=torch.long, device="cpu")
        if index.ndim != 1:
            raise ValueError("feasible teacher indices must be one-dimensional")
        if bool(((index < 0) | (index >= self.sample_count)).any()):
            raise IndexError("feasible teacher index lies outside cached dataset")
        if not torch.equal(
            self.batch.sample_indices,
            torch.arange(self.sample_count, dtype=torch.long),
        ):
            raise RuntimeError("feasible teacher cache has lost positional alignment")
        return {
            name: value[index].to(device)
            for name, value in self.batch.as_loss_kwargs().items()
        }


@torch.no_grad()
def build_or_load_v16_feasible_teacher_cache(
    model: torch.nn.Module,
    dataset: Dataset,
    cache_path: str | Path,
    *,
    mse_tolerance: float,
    device: torch.device | str,
    batch_size: int = 16,
    smoothness_weight: float = 0.0,
    control_ridge: float = 0.0,
    teacher_strategy: str = "full_greedy",
    anchor_match_tolerance: float = 0.02,
    anchor_fallback_to_greedy: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> V16FeasibleTeacherCache:
    """Build once or validate/reuse exact fixed-Proposal, fixed-data labels.

    The hard teacher uses ``sqrt(mse_tolerance)`` as its Euclidean RMS bound.
    Search/refits run in float64 to match the v16 subset loss's numerical
    refit, with endpoint interpolation and zero regularization by default.
    An existing but mismatched cache is rejected, never overwritten.
    """
    if not math.isfinite(mse_tolerance) or mse_tolerance <= 0:
        raise ValueError("mse_tolerance must be finite and positive")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if teacher_strategy not in _TEACHER_STRATEGIES:
        raise ValueError(f"teacher_strategy must be one of {sorted(_TEACHER_STRATEGIES)}")
    if not math.isfinite(anchor_match_tolerance) or anchor_match_tolerance < 0:
        raise ValueError("anchor_match_tolerance must be finite and non-negative")
    destination = Path(cache_path)
    if destination.suffix.lower() != ".pt":
        raise ValueError("v16 feasible teacher cache must end with .pt")
    all_points, dataset_fingerprint, all_anchor_labels = _fixed_dataset_points(
        dataset,
        teacher_strategy=teacher_strategy,
        anchor_match_tolerance=anchor_match_tolerance,
        anchor_fallback_to_greedy=anchor_fallback_to_greedy,
    )
    proposal_fingerprint = _proposal_fingerprint(model)
    candidate_count = int(getattr(model, "max_internal_knots"))
    degree = int(getattr(model, "degree"))
    config = OneShotTeacherConfig(
        error_tolerance=math.sqrt(mse_tolerance),
        min_internal_knots=int(getattr(model, "min_selected_knots", 0)),
        degree=degree,
        smoothness_weight=smoothness_weight,
        control_ridge=control_ridge,
        proposal_fingerprint=proposal_fingerprint,
        dataset_fingerprint=dataset_fingerprint,
        relocation_strategy="none",
    )
    sample_count, point_count, point_dim = all_points.shape
    shapes = {
        "parameters": (sample_count, point_count),
        "points": (sample_count, point_count, point_dim),
        "candidate_knots": (sample_count, candidate_count),
    }
    sample_indices = torch.arange(sample_count, dtype=torch.long)
    if destination.exists():
        try:
            cached = load_one_shot_teacher_cache(
                destination,
                expected_config=config,
                expected_sample_indices=sample_indices,
                expected_input_shapes=shapes,
            )
        except (ValueError, RuntimeError) as exc:
            raise ValueError(
                f"existing feasible teacher cache does not match frozen "
                f"Proposal/data/numerical contract: {destination}; "
                "choose a new cache path"
            ) from exc
        return V16FeasibleTeacherCache(
            cached, destination, loaded=True, teacher_strategy=teacher_strategy,
        )

    requested_device = torch.device(device)
    model_device = next(model.parameters()).device
    if not _devices_equivalent(model_device, requested_device):
        raise ValueError(
            f"proposal model is on {model_device}, expected {requested_device}"
        )
    target_device = (
        torch.device("cuda", torch.cuda.current_device())
        if requested_device.type == "cuda" and requested_device.index is None
        else requested_device
    )
    model_dtype = next(model.parameters()).dtype
    was_training = model.training
    model.eval()
    rows: dict[str, list[torch.Tensor]] = {}
    try:
        for start in range(0, sample_count, batch_size):
            stop = min(start + batch_size, sample_count)
            points = all_points[start:stop].to(
                device=target_device, dtype=model_dtype
            )
            context = model.encode_candidates(points, mse_tolerance=mse_tolerance)
            parameters = context["proposal_params"]
            candidates = context["proposal_internal_knots"]
            if parameters.shape != (stop - start, point_count):
                raise ValueError("proposal parameter output has an invalid shape")
            if candidates.shape != (stop - start, candidate_count):
                raise ValueError("proposal candidate output has an invalid shape")
            if teacher_strategy == "synthetic_anchor_first":
                assert all_anchor_labels is not None
                teacher = _anchor_first_teacher_batch(
                    parameters.double(),
                    points.double(),
                    candidates.double(),
                    all_anchor_labels[start:stop],
                    sample_indices=sample_indices[start:stop],
                    config=config,
                    anchor_match_tolerance=anchor_match_tolerance,
                    anchor_fallback_to_greedy=anchor_fallback_to_greedy,
                )
            else:
                teacher = build_one_shot_teacher_batch(
                    parameters.double(),
                    points.double(),
                    candidates.double(),
                    sample_indices=sample_indices[start:stop],
                    config=config,
                )
            for name, value in teacher.as_loss_kwargs().items():
                rows.setdefault(name, []).append(value.detach().cpu())
            if progress is not None:
                progress(stop, sample_count)
    finally:
        model.train(was_training)
    labels = {name: torch.cat(values, dim=0) for name, values in rows.items()}
    full_batch = OneShotTeacherBatch(
        sample_indices=sample_indices,
        input_shapes=shapes,
        config=config,
        **labels,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f"{destination.stem}.tmp.{os.getpid()}.pt"
    )
    if temporary.exists():
        raise FileExistsError(f"temporary feasible teacher file exists: {temporary}")
    try:
        save_one_shot_teacher_cache(temporary, full_batch)
        # Atomic exclusive publication.  A second process racing for the same
        # cache name must fail, not replace the first process's labels.
        if os.name == "nt":
            # Windows rename is exclusive; hard links may be disallowed by
            # the workspace filesystem's security policy.
            os.rename(temporary, destination)
        else:
            os.link(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return V16FeasibleTeacherCache(
        full_batch, destination, loaded=False, teacher_strategy=teacher_strategy,
    )
