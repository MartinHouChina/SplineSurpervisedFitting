from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import matplotlib.figure
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "visualize_batch_comparison.py"


def _load_script():
    module_name = "test_visualize_batch_comparison_module"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve postponed annotations through ``sys.modules`` while
    # the module body is executing.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


batch_comparison = _load_script()


def test_mean_squared_euclidean_error_is_pointwise_vector_mse() -> None:
    predicted = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    observed = torch.zeros_like(predicted)

    # Squared Euclidean errors are 1^2+2^2=5 and 3^2+4^2=25.
    actual = batch_comparison.mean_squared_euclidean_error(predicted, observed)

    assert float(actual) == pytest.approx((5.0 + 25.0) / 2.0)
    assert float(actual) == pytest.approx(15.0)


def test_checkpoint_rms_tolerance_is_squared_into_mse() -> None:
    checkpoint = {"deployment_config": {"error_tolerance": 0.005}}

    actual = batch_comparison.resolve_mse_tolerance(checkpoint, None)

    assert actual == pytest.approx(2.5e-5)


def test_stratified_sampling_is_deterministic_balanced_unique_and_rng_local() -> None:
    index_to_count = {
        0: 4,
        1: 4,
        2: 4,
        3: 5,
        4: 5,
        5: 5,
        6: 6,
        7: 6,
        8: 6,
    }
    python_rng_before = random.getstate()
    torch_rng_before = torch.random.get_rng_state().clone()

    first = batch_comparison.stratified_random_indices(
        index_to_count,
        6,
        seed=90210,
        allowed_counts=(4, 5, 6),
    )
    second = batch_comparison.stratified_random_indices(
        index_to_count,
        6,
        seed=90210,
        allowed_counts=(4, 5, 6),
    )

    assert first == second
    assert len(first) == len(set(first)) == 6
    selected_counts = [index_to_count[index] for index in first]
    assert set(selected_counts) == {4, 5, 6}
    assert {count: selected_counts.count(count) for count in (4, 5, 6)} == {
        4: 2,
        5: 2,
        6: 2,
    }
    assert random.getstate() == python_rng_before
    torch.testing.assert_close(torch.random.get_rng_state(), torch_rng_before)


def test_deployment_geometry_keeps_proposal_separate_from_v11_positions() -> None:
    proposal = torch.tensor([[0.10, 0.30, 0.60, 0.85]])
    deployment = torch.tensor([[0.12, 0.27, 0.64, 0.81]])
    learned_mask = torch.tensor([[False, True, True, False]])
    output = {
        # Historical ``internal_knots`` is deployment geometry in v11.
        "internal_knots": deployment.clone(),
        "proposal_internal_knots": proposal.clone(),
        "deployment_internal_knots": deployment.clone(),
        "keep_probability": torch.tensor([[0.1, 0.8, 0.7, 0.2]]),
        "learned_keep_mask": learned_mask.clone(),
    }

    actual_proposal, actual_deployment, actual_mask = (
        batch_comparison.deployment_geometries(output)
    )

    torch.testing.assert_close(actual_proposal, proposal[0])
    torch.testing.assert_close(actual_deployment, deployment[0])
    torch.testing.assert_close(actual_mask, learned_mask[0])
    assert not torch.equal(actual_proposal, actual_deployment)
    torch.testing.assert_close(
        actual_deployment[actual_mask],
        torch.tensor([0.27, 0.64]),
    )
    # Reading the deployment path must not overwrite immutable Hard-pruning
    # proposal geometry.
    torch.testing.assert_close(output["proposal_internal_knots"], proposal)


def test_greedy_mse_pruning_respects_threshold_and_retained_structure() -> None:
    parameters = torch.linspace(0.0, 1.0, 24, dtype=torch.float64)
    # A planar cubic polynomial is exactly representable with no internal
    # knots, so every redundant candidate can safely be deleted.
    points = torch.stack(
        [parameters, 0.2 + 0.4 * parameters - 0.3 * parameters**2 + parameters**3],
        dim=-1,
    )
    candidates = torch.tensor([0.18, 0.37, 0.63, 0.82], dtype=torch.float64)
    tolerance = 1e-16

    result = batch_comparison.greedy_prune_to_mse_tolerance(
        parameters,
        points,
        candidates,
        mse_tolerance=tolerance,
        smoothness_weight=0.0,
        control_ridge=0.0,
    )

    assert result.initial_count == 4
    assert result.final_count == 0
    assert result.accepted_deletions == 4
    assert len(result.steps) == 4
    assert len(result.mse_trajectory) == 5
    assert result.threshold_satisfied
    assert float(result.final_fit.fit_mse) <= tolerance
    assert result.retained_internal_knots.numel() == 0
    assert result.retained_mask.dtype == torch.bool
    assert result.retained_mask.shape == candidates.shape
    assert not bool(result.retained_mask.any())
    assert all(step.accepted for step in result.steps)


def test_render_small_four_axis_png_uses_mse_titles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = torch.linspace(0.0, 1.0, 12, dtype=torch.float64)
    observed = torch.stack([parameters, parameters**3], dim=-1)
    fit = batch_comparison.refit_bspline_control_points(
        parameters,
        observed,
        torch.empty(0, dtype=torch.float64),
        degree=3,
        smoothness_weight=0.0,
        control_ridge=0.0,
        interpolate_endpoints=True,
    )
    dense_parameters = torch.linspace(0.0, 1.0, 32, dtype=torch.float64)
    panels = tuple(
        batch_comparison.fit_panel(
            fit,
            dense_parameters,
            title=title,
            curve_label=label,
            timing_text="total=0.01 ms",
            curve_color=color,
        )
        for title, label, color in (
            ("(a) Original source data", "source", "#355c9a"),
            ("(b) Redundant proposal", "proposal", "#6f6f6f"),
            ("(c) Learned deployment", "learned", "#198a77"),
            ("(d) Greedy hard pruning", "hard", "#c94b4b"),
        )
    )

    captured: dict[str, object] = {}
    original_savefig = matplotlib.figure.Figure.savefig

    def capture_savefig(figure, *args, **kwargs):
        captured["axis_count"] = len(figure.axes)
        captured["text"] = "\n".join(
            [text.get_text() for text in figure.texts]
            + [axis.get_title() for axis in figure.axes]
            + [text.get_text() for axis in figure.axes for text in axis.texts]
        )
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture_savefig)
    output_path = tmp_path / "comparison.png"

    batch_comparison.render_comparison_figure(
        panels,
        observed,
        output_path,
        degree=3,
        sample_index=7,
        source_count=0,
        canonical_count=0,
        mse_tolerance=2.5e-5,
        dpi=50,
    )

    assert captured["axis_count"] == 4
    figure_text = str(captured["text"])
    assert figure_text.count("MSE=") == 4
    assert "RMS" not in figure_text.upper()
    assert output_path.is_file()
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
