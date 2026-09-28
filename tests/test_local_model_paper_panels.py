from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from plot_local_model_paper_panels import choose_comparison_cases, choose_showcase_cases, summarize


def row(i, mse=1e-6, status='ok', group=None):
    return dict(dataset='UJI', sample_id=str(i), method='ours', mse=mse,
                status=status, fit_pass=status == 'ok' and mse <= 5e-5,
                group_id=group or str(i), has_reference=False,
                max_squared_error=mse * 10, final_k=10, total_ms=1.)


def test_random_comparison_is_not_selected_by_error():
    grouped = {('UJI', str(i)): {'ours': row(i)} for i in range(10)}
    before = choose_comparison_cases(grouped, 42)
    for i, values in enumerate(grouped.values()):
        values['ours']['mse'] = 999 - i
        values['ours']['fit_pass'] = False
    assert choose_comparison_cases(grouped, 42) == before


def test_showcase_preference_and_group_diversity():
    grouped = {('UJI', str(i)): {'ours': row(i, (i + 1) * 1e-6, group='shared' if i < 2 else str(i))}
               for i in range(7)}
    selected = choose_showcase_cases(grouped, 5e-5)['UJI']
    assert len(selected) == 5
    assert selected[0] == ('UJI', '0')
    assert ('UJI', '1') not in selected


def test_table_uses_misses_and_preserves_failure_denominator():
    rows = [row(0), row(1, 1e-3), row(2, status='failed')]
    summary = summarize(rows, {'methods': ['ours']})[0]
    assert summary['n_executed'] == 3
    assert summary['n_finite_fits'] == 2
    assert summary['pass_percent'] == 100 / 3
    assert summary['max_squared_error_max'] == .01
    assert summary['mse_mean'] > 5e-5


def test_showcase_does_not_replace_a_pass_with_a_failure_for_diversity():
    grouped = {('UJI', str(i)): {'ours': row(i, group='shared')} for i in range(5)}
    grouped[('UJI', 'failed')] = {'ours': row('failed', mse=.1, group='different')}
    selected = choose_showcase_cases(grouped, 5e-5)['UJI']
    assert len(selected) == 5
    assert ('UJI', 'failed') not in selected
