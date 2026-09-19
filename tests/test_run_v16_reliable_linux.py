"""Reliable orchestration keeps real validation and numerical repair explicit."""
import pytest

from test_run_v16_1070_overnight_linux import _commands, _parsed, _run


def test_reliable_profile_is_synthetic_train_with_real_validation(tmp_path):
    commands = _commands(_run(tmp_path, "--reliable-selection"))
    train = _parsed(commands["train_fresh"])
    assert (train.epochs, train.proposal_epochs) == (32, 4)
    assert train.parameter_trust_enabled and train.parameter_trust_initial == .25
    assert train.real_fraction == 0 and train.real_val_size == 32
    assert len(train.real_manifest) == 3
    assert train.candidate_knots == 64 and train.max_control_points == 28
    assert train.teacher_geometry_candidates == 4
    assert train.parameter_counterfactual_weight == .25
    assert train.local_fit_weight == .1 and train.proposal_ordered_weight == .5
    assert train.allow_infeasible_proposals
    assert "--validate-real-splits" in commands["check_data_and_provenance"]
    for name in ("benchmark_six_methods", "plot_ours_cases", "plot_six_method_real_cases"):
        assert _parsed(commands[name]).published_feasibility_safeguard
    assert not list(tmp_path.iterdir())


def test_native_baseline_optout_reaches_benchmark_and_case_figures(tmp_path):
    commands = _commands(_run(tmp_path, "--reliable-selection", "--native-baselines",
                              "--real-val-size", "7"))
    assert _parsed(commands["train_fresh"]).real_val_size == 7
    for name in ("benchmark_six_methods", "plot_ours_cases", "plot_six_method_real_cases"):
        assert not _parsed(commands[name]).published_feasibility_safeguard


@pytest.mark.parametrize("options", [
    ("--reliable-selection", "--enhanced-selection"),
    ("--reliable-selection", "--real-val-size", "0"),
])
def test_incompatible_profiles_or_invalid_validation_size_fail_early(tmp_path, options):
    assert _run(tmp_path, *options).returncode != 0
    assert not list(tmp_path.iterdir())
