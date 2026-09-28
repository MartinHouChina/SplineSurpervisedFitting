import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_v16_saved_models_evaluation as runner
from benchmark_v15_datasets import select_real_test_indices


def test_paired_fraction_is_reproducible_and_rounds_up():
    records = [{} for _ in range(201)]
    a = select_real_test_indices(records, 100, 20260922, fraction=.1)
    assert len(a) == len(set(a)) == 21
    assert a == select_real_test_indices(records, 1, 20260922, fraction=.1)
    assert a != select_real_test_indices(records, 1, 123, fraction=.1)
    assert select_real_test_indices([{}], 100, 42, fraction=.1) == [0]


@pytest.mark.parametrize('fraction', [0, -1, 1.1, float('nan')])
def test_invalid_fractions(fraction):
    with pytest.raises(ValueError):
        select_real_test_indices([{}], 1, 42, fraction=fraction)


def test_no_training_and_correct_checkpoint_names():
    commands = runner.build_commands(runner.parser().parse_args([]))
    assert len(commands) == 6
    assert 'paper_coupled_clean_3090_r2_m16.pt' in ' '.join(commands[0])
    assert 'paper_coupled_clean_3090_r3_m32.pt' in ' '.join(commands[3])
    for index in (0, 3):
        cmd = commands[index]
        assert cmd[cmd.index('--real-test-fraction') + 1] == '0.1'
        assert '--no-published-feasibility-safeguard' in cmd
        assert not any('train' in arg for arg in cmd[:2])
    assert commands[2][commands[2].index('--max-cases-per-dataset') + 1] == '0'
    assert len(runner.build_commands(runner.parser().parse_args(['--capacities', '32']))) == 3


def test_conflicting_selection_rejected():
    with pytest.raises(ValueError):
        select_real_test_indices([{}], 1, 42, fraction=.1, all_test=True)


def test_generated_commands_parse_in_actual_entrypoints():
    from benchmark_v15_datasets import parser as benchmark_parser
    from visualize_v16_six_methods import parser as figure_parser
    from run_v16_paper_training import parser as training_parser, build_plan
    commands = runner.build_commands(runner.parser().parse_args([]))
    for index in (0, 3):
        args = benchmark_parser().parse_args(commands[index][2:])
        assert args.real_test_fraction == .1
        assert args.selection_seed == 20260922
        assert args.method_set == 'published'
    for index in (2, 5):
        assert figure_parser().parse_args(commands[index][2:]).max_cases_per_dataset == 0
    plan = build_plan(training_parser().parse_args([
        '--run-prefix', 'unused_dry', '--capacities', '32', '--real-test-fraction', '.1']))
    assert '--real-test-fraction' in plan['runs'][0]['command']


def test_fraction_reaches_real_case_preparation(tmp_path, monkeypatch):
    import torch
    import benchmark_v15_datasets as benchmark
    class Dataset:
        def __init__(self, *args, **kwargs):
            self.records = [dict(group_id=f'g{i}') for i in range(21)]
        def __len__(self):
            return len(self.records)
        def __getitem__(self, i):
            return dict(points=torch.tensor([[0., 0.], [1., 1.]]),
                        center=torch.zeros(2), scale=torch.tensor(1.), curve_id=f'c{i}',
                        group_id=f'g{i}', source_dataset='test')
        def load_reference_points(self, i):
            return self[i]['points']
    monkeypatch.setattr(benchmark, '_dataset_config_from_checkpoint', lambda *a: dict(num_points=2, point_dim=2))
    monkeypatch.setattr(benchmark, 'RealWorldCurveDataset', Dataset)
    monkeypatch.setattr(benchmark, 'read_curve_manifest', lambda p: [])
    monkeypatch.setattr(benchmark, 'validate_source_records', lambda *a, **k: None)
    monkeypatch.setattr(benchmark, 'source_description', lambda *a: {})
    monkeypatch.setattr(benchmark, 'resolve_points_path', lambda *a: tmp_path / 'points')
    monkeypatch.setattr(benchmark, 'sha256_file', lambda *a: 'testhash')
    args = benchmark.parser().parse_args([
        '--checkpoint', 'unused.pt', '--output-dir', str(tmp_path), '--skip-synthetic',
        '--manifest', f'Test={tmp_path / "manifest"}', '--real-test-fraction', '.1'])
    cases, provenance = benchmark.prepare_cases(args, {}, {})
    assert len(cases) == 3
    assert provenance[0]['sampling'] == 'uniform_without_replacement'
    assert provenance[0]['test_fraction'] == .1
    assert provenance[0]['selected_count'] == 3
