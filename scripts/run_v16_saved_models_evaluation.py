"""Evaluate existing M16/M32 checkpoints only; never launch training."""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--m16-checkpoint', type=Path, default=Path('outputs/checkpoints/paper_coupled_clean_3090_r2_m16.pt'))
    p.add_argument('--m32-checkpoint', type=Path, default=Path('outputs/checkpoints/paper_coupled_clean_3090_r3_m32.pt'))
    p.add_argument('--m64-checkpoint', type=Path, default=None)
    p.add_argument('--capacities', type=int, nargs='+', choices=(16, 32, 64), default=[16, 32])
    p.add_argument('--data-root', type=Path, default=ROOT / 'data')
    p.add_argument('--device', choices=('cuda', 'cpu', 'auto'), default='cuda')
    p.add_argument('--real-test-fraction', type=float, default=0.1)
    p.add_argument('--selection-seed', type=int, default=20260922)
    p.add_argument('--samples-per-knot-count', type=int, default=10)
    p.add_argument('--mse-tolerance', type=float, default=5e-5)
    p.add_argument('--tag', default='test10pct_20260922')
    p.add_argument('--resume', action='store_true', help='Resume compatible benchmark rows; redraw saved results')
    p.add_argument('--dry-run', action='store_true')
    return p


def build_commands(args):
    if not 0 < args.real_test_fraction <= 1:
        raise ValueError('real-test-fraction must lie in (0,1]')
    if len(set(args.capacities)) != len(args.capacities):
        raise ValueError('capacities must not repeat')
    if not args.tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.tag):
        raise ValueError('tag must contain only letters, digits, underscore or hyphen')
    commands = []
    for capacity in args.capacities:
        checkpoint = getattr(args, f'm{capacity}_checkpoint')
        if checkpoint is None:
            raise ValueError('M64 evaluation requires --m64-checkpoint')
        name = f'{checkpoint.stem}_{args.tag}'
        output = ROOT / 'outputs/comparisons' / name
        figures = ROOT / 'outputs/figures' / name
        benchmark = [sys.executable, str(ROOT / 'scripts/benchmark_v16_datasets.py'),
            '--checkpoint', str(checkpoint), '--output-dir', str(output),
            '--data-root', str(args.data_root), '--device', args.device,
            '--method-set', 'published', '--baseline-protocol', 'adaptation',
            '--no-published-feasibility-safeguard',
            '--min-knot-count', '4', '--max-knot-count', '24', '--synthetic-source-max-knots', '24',
            '--samples-per-knot-count', str(args.samples_per_knot_count), '--seed', '20000',
            '--selection-seed', str(args.selection_seed), '--real-test-fraction', str(args.real_test_fraction),
            '--mse-tolerance', str(args.mse_tolerance),
            '--max-internal-knots', str(capacity), '--paper-initial-knots', str(capacity),
            '--liang-dense-knots', str(capacity), '--paper-admm-iterations', '400',
            '--paper-lambda-bisections', '8', '--paper-relocation-iterations', '8',
            '--liang-feature-samples', '1025', '--dung-scan-intervals', '10',
            '--dung-optimization-iterations', '10', '--luo-de-population', '10', '--luo-de-iterations', '50',
            '--network-warmups', '3', '--network-repeats', '100', '--end-to-end-repeats', '3',
            '--torch-num-threads', '4', '--allow-unqualified-diagnostic']
        if args.resume:
            benchmark.append('--resume')
        commands += [benchmark,
            [sys.executable, str(ROOT / 'scripts/plot_v16_method_comparison.py'),
             '--input', str(output / 'comparison.json'), '--output-dir', str(figures / 'metrics'),
             '--method-set', 'published', '--dpi', '300', '--reference', '--allow-unqualified-diagnostic'],
            [sys.executable, str(ROOT / 'scripts/visualize_v16_six_methods.py'),
             '--benchmark-dir', str(output), '--output-dir', str(figures / 'cases'),
             '--max-cases-per-dataset', '0', '--dpi', '240']]
    return commands


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        commands = build_commands(args)
        if not args.dry_run:
            import torch
            from overnight_datasets import default_manifests
            for path in default_manifests(args.data_root).values():
                if not path.is_file():
                    raise ValueError(f'Required prepared manifest missing: {path}')
            for capacity in args.capacities:
                path = getattr(args, f'm{capacity}_checkpoint')
                checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                if checkpoint.get('model_config', {}).get('max_internal_knots') != capacity:
                    raise ValueError(f'Checkpoint capacity does not match M{capacity}: {path}')
        for command in commands:
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                result = subprocess.run(command, cwd=ROOT, check=False)
                if result.returncode:
                    return result.returncode
    except (OSError, ValueError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
