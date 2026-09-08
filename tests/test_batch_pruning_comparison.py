from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_fixed_samples_per_knot_count_covers_every_requested_stratum() -> None:
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

    first = batch_comparison.fixed_samples_per_knot_count(
        index_to_count,
        (4, 5, 6),
        samples_per_count=2,
        seed=777,
    )
    second = batch_comparison.fixed_samples_per_knot_count(
        index_to_count,
        (4, 5, 6),
        samples_per_count=2,
        seed=777,
    )

    assert first == second
    assert len(first) == len(set(first)) == 6
    assert [index_to_count[index] for index in first] == [4, 4, 5, 5, 6, 6]
    assert random.getstate() == python_rng_before
    torch.testing.assert_close(torch.random.get_rng_state(), torch_rng_before)


def test_fixed_samples_per_knot_count_reports_sparse_strata() -> None:
    with pytest.raises(ValueError, match="increase --scan-size"):
        batch_comparison.fixed_samples_per_knot_count(
            {0: 4, 1: 5},
            (4, 5),
            samples_per_count=2,
            seed=1,
        )


def test_explicit_sample_indices_support_single_and_batch_selection() -> None:
    assert batch_comparison.validate_explicit_sample_indices([17]) == [17]
    assert batch_comparison.validate_explicit_sample_indices([17, 3, 99]) == [17, 3, 99]

    with pytest.raises(ValueError, match="unique"):
        batch_comparison.validate_explicit_sample_indices([4, 4])
    with pytest.raises(ValueError, match="non-negative"):
        batch_comparison.validate_explicit_sample_indices([-1])


def test_timed_call_excludes_requested_warmups() -> None:
    calls: list[int] = []

    def operation() -> int:
        calls.append(len(calls) + 1)
        return calls[-1]

    result, duration_ms = batch_comparison.timed_call(
        operation,
        repeats=3,
        warmup_repeats=2,
    )

    assert result == 5
    assert len(calls) == 5
    assert duration_ms >= 0.0


def test_hard_timing_replay_uses_tolerant_non_blocking_diagnostics() -> None:
    authoritative = SimpleNamespace(
        final_count=2,
        retained_mask=torch.tensor([True, False, True]),
        retained_internal_knots=torch.tensor([0.2, 0.8], dtype=torch.float64),
        final_fit=SimpleNamespace(fit_mse=torch.tensor(2.0e-5)),
        threshold_satisfied=True,
    )
    tiny_roundoff = SimpleNamespace(
        final_count=2,
        retained_mask=torch.tensor([True, False, True]),
        retained_internal_knots=torch.tensor(
            [0.2 + 2.0e-7, 0.8 - 2.0e-7], dtype=torch.float64
        ),
        final_fit=SimpleNamespace(fit_mse=torch.tensor(2.0e-5 + 1.0e-10)),
        threshold_satisfied=True,
    )

    close = batch_comparison.compare_hard_timing_reproduction(
        authoritative,
        tiny_roundoff,
    )

    assert close.structurally_consistent
    assert close.count_equal
    assert close.retained_mask_equal
    assert close.retained_knots_allclose
    assert close.retained_knots_max_abs_difference == pytest.approx(2.0e-7)

    discrete_change = SimpleNamespace(
        final_count=1,
        retained_mask=torch.tensor([True, False, False]),
        retained_internal_knots=torch.tensor([0.2], dtype=torch.float64),
        final_fit=SimpleNamespace(fit_mse=torch.tensor(2.4e-5)),
        threshold_satisfied=True,
    )
    changed = batch_comparison.compare_hard_timing_reproduction(
        authoritative,
        discrete_change,
    )

    # A real discrete replay mismatch is reported, not raised. The caller can
    # continue rendering with the authoritative geometry.
    assert not changed.structurally_consistent
    assert not changed.count_equal
    assert not changed.retained_mask_equal
    assert not changed.retained_knots_allclose
    assert changed.retained_knots_max_abs_difference is None
    assert changed.mse_abs_difference == pytest.approx(4.0e-6)


@pytest.mark.parametrize("certified", [False, True])
def test_fast_source_count_scan_matches_materialized_dataset(certified: bool) -> None:
    config = {
        "num_points": 48,
        "point_dim": 2,
        "min_control_points": 8,
        "max_control_points": 24,
        "noise_std": 0.001,
        "canonical_knot_tolerance": 0.005,
        "certified_minimal_source": certified,
        "minimality_audit_points": 64,
    }
    dataset = batch_comparison.SyntheticCubicBSplineDataset(
        size=12,
        seed=2468,
        cache_samples=False,
        **config,
    )

    for sample_index in (0, 3, 11):
        expected = int(dataset[sample_index]["source_internal_knot_count"])
        actual = batch_comparison.source_knot_count_from_seed(
            config,
            dataset_seed=2468,
            sample_index=sample_index,
        )
        assert actual == expected


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


def test_full_chord_comparison_maps_shared_proposal_geometry() -> None:
    parameters = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    chord_parameters = torch.tensor([0.0, 0.2, 1.0], dtype=torch.float64)
    proposal = torch.tensor([0.25, 0.75], dtype=torch.float64)

    actual_parameters, actual_proposal, shared = (
        batch_comparison.comparison_parameterization_geometry(
            parameters,
            proposal,
            chord_parameters,
            ours_deployment="verified",
            verified_parameterization="chord",
        )
    )

    assert shared
    torch.testing.assert_close(actual_parameters, chord_parameters)
    torch.testing.assert_close(
        actual_proposal,
        torch.tensor([0.1, 0.6], dtype=torch.float64),
    )

    network_parameters, network_proposal, shared = (
        batch_comparison.comparison_parameterization_geometry(
            parameters,
            proposal,
            chord_parameters,
            ours_deployment="verified",
            verified_parameterization="chord-fallback",
        )
    )
    assert not shared
    assert network_parameters is parameters
    assert network_proposal is proposal


def test_verified_visualization_deployment_returns_final_chord_fit() -> None:
    chord_parameters = torch.linspace(0.0, 1.0, 24, dtype=torch.float64)
    network_parameters = chord_parameters.pow(1.2)
    points = torch.stack(
        [chord_parameters, 0.2 + chord_parameters - 0.3 * chord_parameters**3],
        dim=-1,
    )
    proposal = torch.tensor([0.18, 0.38, 0.62, 0.82], dtype=torch.float64)

    class DummyModel(torch.nn.Module):
        degree = 3

        def forward(self, batched_points: torch.Tensor) -> dict[str, torch.Tensor]:
            del batched_points
            return {
                "params": network_parameters.unsqueeze(0),
                "internal_knots": proposal.unsqueeze(0),
                "proposal_internal_knots": proposal.unsqueeze(0),
                "deployment_internal_knots": proposal.unsqueeze(0),
                "learned_keep_mask": torch.ones(
                    (1, proposal.numel()), dtype=torch.bool
                ),
                "keep_probability": torch.full(
                    (1, proposal.numel()), 0.8, dtype=torch.float64
                ),
            }

    result = batch_comparison.run_verified_visualization_deployment(
        DummyModel(),
        points.unsqueeze(0),
        points,
        chord_parameters,
        mse_tolerance=1e-12,
        min_internal_knots=0,
        smoothness_weight=0.0,
        control_ridge=0.0,
        compact=True,
        hard_fallback=True,
        residual_fallback=True,
        max_residual_insertions=4,
        residual_min_gap=1e-4,
        parameterization_policy="chord",
        refit_device="cpu",
    )

    assert result.repair.final_parameterization == "chord_length"
    assert result.repair.threshold_satisfied
    torch.testing.assert_close(result.repair.final_parameters, chord_parameters)
    assert float(result.repair.final_fit.fit_mse) <= 1e-12


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
        parameterization_note=(
            "Redundant, Ours verified, and Greedy share chord-length parameters."
        ),
        timing_scope_note=(
            "Timing scopes are symmetric end-to-end for this comparison."
        ),
    )

    assert captured["axis_count"] == 4
    figure_text = str(captured["text"])
    assert figure_text.count("MSE=") == 4
    assert "RMS" not in figure_text.upper()
    assert "share chord-length parameters" in figure_text
    assert "symmetric end-to-end" in figure_text
    assert output_path.is_file()
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_batch_cli_exposes_learned_and_verified_deployment_controls() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert '"--ours-deployment"' in source
    assert 'choices=("learned", "verified")' in source
    assert '"--verified-parameterization"' in source
    assert 'choices=("network", "chord-fallback", "chord")' in source
    assert '"--verified-refit-device"' in source
    assert '"--sample-indices"' in source
    assert '"--timing-scope"' in source
    assert '"--timing-warmups"' in source
    assert '("historical-asymmetric", "end-to-end")' in source
