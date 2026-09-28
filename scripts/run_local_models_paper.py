"""Reproduce the fixed local preview and saved-result paper panels (no training)."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def commands(args):
    result = []
    for capacity in args.capacities:
        prefix = 'paper_coupled_clean_3090_r2_m16' if capacity == 16 else 'paper_coupled_clean_3090_r3_m32'
        checkpoint = args.checkpoint_dir / (prefix + '.pt')
        output = ROOT / f'outputs/comparisons/local_models_paper_20260922_m{capacity}'
        figures = ROOT / f'outputs/figures/local_models_paper_20260922/m{capacity}'
        if not args.plot_only:
            cmd = [sys.executable, str(ROOT / 'scripts/benchmark_v16_datasets.py'),
                '--checkpoint', str(checkpoint), '--output-dir', str(output), '--data-root', str(args.data_root),
                '--device', args.device, '--method-set', 'published', '--baseline-protocol', 'adaptation',
                '--no-published-feasibility-safeguard', '--min-knot-count', '4', '--max-knot-count', '24',
                '--synthetic-source-max-knots', '24', '--samples-per-knot-count', '1', '--seed', '20000',
                '--selection-seed', '20260922', '--real-samples-per-dataset', '10', '--mse-tolerance', '5e-5',
                '--max-internal-knots', str(capacity), '--paper-initial-knots', str(capacity),
                '--liang-dense-knots', str(capacity), '--paper-admm-iterations', '400',
                '--paper-lambda-bisections', '8', '--paper-relocation-iterations', '8', '--liang-feature-samples', '1025',
                '--dung-scan-intervals', '10', '--dung-optimization-iterations', '10',
                '--luo-de-population', '10', '--luo-de-iterations', '50',
                '--network-warmups', '3', '--network-repeats', '10', '--end-to-end-repeats', '1',
                '--torch-num-threads', '4', '--allow-unqualified-diagnostic']
            if args.resume:
                cmd += ['--resume']
            result.append(cmd)
        result.append([sys.executable, str(ROOT / 'scripts/plot_local_model_paper_panels.py'),
                       '--benchmark-dir', str(output), '--output-dir', str(figures), '--seed', '20260922'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('outputs/downloads/server_models_20260922_215418'))
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--capacities', nargs='+', type=int, choices=(16, 32), default=[32, 16])
    parser.add_argument('--device', choices=('cuda', 'cpu', 'auto'), default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    for cmd in commands(args):
        print(subprocess.list2cmdline(cmd), flush=True)
        if not args.dry_run:
            outcome = subprocess.run(cmd, cwd=ROOT, check=False)
            if outcome.returncode:
                return outcome.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
