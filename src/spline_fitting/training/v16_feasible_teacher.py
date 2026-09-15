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

from .one_shot_teacher import (
    OneShotTeacherBatch,
    OneShotTeacherConfig,
    build_one_shot_teacher_batch,
    load_one_shot_teacher_cache,
    save_one_shot_teacher_cache,
)


_PROPOSAL_PREFIXES = ("encoder.", "parameter_head.", "candidate_head.")


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


def _fixed_dataset_points(dataset: Dataset) -> tuple[torch.Tensor, str]:
    """Materialize and fingerprint the exact ordered points in index order."""
    if bool(getattr(dataset, "resample_each_epoch", False)):
        raise ValueError("offline teacher requires a fixed, non-resampling dataset")
    if len(dataset) < 1:
        raise ValueError("offline teacher requires at least one sample")
    digest = hashlib.sha256()
    points: list[torch.Tensor] = []
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
    return torch.stack(points), digest.hexdigest()


@dataclass(frozen=True)
class V16FeasibleTeacherCache:
    """Indexed, CPU-resident teacher labels for a fixed Joint population."""

    batch: OneShotTeacherBatch
    path: Path
    loaded: bool

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
    destination = Path(cache_path)
    if destination.suffix.lower() != ".pt":
        raise ValueError("v16 feasible teacher cache must end with .pt")
    all_points, dataset_fingerprint = _fixed_dataset_points(dataset)
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
        return V16FeasibleTeacherCache(cached, destination, loaded=True)

    target_device = torch.device(device)
    model_device = next(model.parameters()).device
    if model_device != target_device:
        raise ValueError(
            f"proposal model is on {model_device}, expected {target_device}"
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
    return V16FeasibleTeacherCache(full_batch, destination, loaded=False)
