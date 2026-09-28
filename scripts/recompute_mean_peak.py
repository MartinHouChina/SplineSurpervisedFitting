"""Audit saved per-point residuals and aggregate per-curve peak squared errors."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'outputs/comparisons/historical_best_dual_error_postprune_fresh_20260923/comparison.json'
OUT = ROOT / 'outputs/comparisons/mean_peak_recomputed_20260923'


def main():
    report = json.loads(SRC.read_text(encoding='utf-8'))
    rows = report['measurements']
    assert len(rows) == 366
    methods = report['metadata']['methods']
    sources = list(dict.fromkeys(r['dataset'] for r in rows))
    audited = []
    for r in rows:
        item = {k: r.get(k) for k in ('dataset', 'sample_id', 'method', 'status', 'mse', 'max_squared_error')}
        if r.get('mse') is None or not math.isfinite(r['mse']):
            item['audit'] = 'nonfinite fit excluded from error mean; retained in denominator report'
            audited.append(item)
            continue
        artifact = r['geometry_artifact']
        path = SRC.parent / artifact['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact['sha256']
        geom = json.loads(path.read_text(encoding='utf-8'))
        arrays = path.parent / geom['arrays_file']
        assert hashlib.sha256(arrays.read_bytes()).hexdigest() == geom['arrays_sha256']
        with np.load(arrays, allow_pickle=False) as data:
            residual = data['input_residual_vectors_normalized']
            sq = np.sum(residual**2, axis=-1)
            assert len(sq) == 192 and np.isfinite(sq).all()
            np.testing.assert_allclose(sq, data['input_squared_residuals_normalized'], rtol=1e-9, atol=1e-12)
            peak, mse = float(sq.max()), float(sq.mean())
        np.testing.assert_allclose([mse, peak], [r['mse'], r['max_squared_error']], rtol=1e-6, atol=1e-10)
        item.update(recomputed_mse=mse, recomputed_peak_squared=peak, audit='verified from saved residual vectors')
        audited.append(item)
    summary = []
    for source in ['All', *sources]:
        for method in methods:
            group = [r for r in audited if r['method'] == method and (source == 'All' or r['dataset'] == source)]
            peaks = [r['recomputed_peak_squared'] for r in group if 'recomputed_peak_squared' in r]
            summary.append(dict(dataset=source, method=method, total=len(group), finite=len(peaks),
                                mean_peak_squared=float(np.mean(peaks)), worst_peak_squared=max(peaks)))
    OUT.mkdir(parents=True, exist_ok=True)
    payload = dict(source=str(SRC), source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),
                   operation='Re-evaluation of saved residual vectors; no new fitting or timing',
                   definition='mean_s max_i ||C_s(t_i)-Q_s,i||^2; normalized input coordinates, no square root',
                   protocol=report['metadata']['timing_protocol'], per_curve=audited, summary=summary)
    (OUT / 'mean_peak_audit.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    names = ['Ours', 'Park', 'Liang', 'Dung', 'Kang', 'Luo']
    md = ['# Mean per-curve maximum squared error', '',
          'Recomputed from saved normalized residuals; 61 curves, six methods. Not a new larger-cohort experiment.',
          'Synthetic: 21; UJI, Natural Earth, USGS, industrial offsets: 10 each. Not 10% sampling.',
          'All methods use insertion repair; only Ours additionally uses post-pruning. Maximum internal knots: 32.',
          'Mean includes every finite fit, including threshold violations. Dung has one nonfinite Natural Earth result, excluded rather than replaced by zero.', '']
    tex = [r'\documentclass{article}', r'\usepackage[margin=18mm]{geometry}', r'\usepackage{booktabs}',
           r'\begin{document}', r'\begin{table}[htbp]\centering\small',
           r'\caption{Mean per-curve maximum squared error, recomputed from saved residuals.}',
           r'\begin{tabular}{lrrrrrr}\toprule',
           r'Source & Ours & Park & Liang & Dung & Kang & Luo \\\midrule']
    for source in ['All', *sources]:
        group = [s for s in summary if s['dataset'] == source]
        md += [f'## {source}', '', '| Method | Finite/total | Mean peak squared error | Worst peak squared error |', '|---|---:|---:|---:|']
        for name, s in zip(names, group):
            md.append(f"| {name} | {s['finite']}/{s['total']} | {s['mean_peak_squared']:.6e} | {s['worst_peak_squared']:.6e} |")
        md.append('')
        def sci(x):
            a, b = f'{x:.4e}'.split('e')
            return '$' + a + r'\times10^{' + str(int(b)) + '}$'
        tex.append(source + ' & ' + ' & '.join(sci(s['mean_peak_squared']) for s in group) + r' \\')
    tex += [r'\bottomrule\end{tabular}', r'\end{table}',
            r'\noindent Overall means are pooled over curves, not equal-weighted dataset means. Synthetic: 21 curves; each other source: 10 curves. All finite fits, including tolerance misses, are included. Dung: 9/10 finite Natural Earth fits, 60/61 overall; other methods: 61/61.',
            r'\par Errors are normalized squared Euclidean distances, without square roots. Historical 32-knot protocol: insertion repair for all methods; additional post-pruning only for Ours. This is not a 10\% dataset evaluation.',
            r'\end{document}']
    (OUT / 'mean_peak_tables.md').write_text('\n'.join(md), encoding='utf-8')
    (OUT / 'mean_peak_tables.tex').write_text('\n'.join(tex), encoding='utf-8')
    print(json.dumps(summary[:6], indent=2))
    print('Audited', sum('recomputed_peak_squared' in r for r in audited), 'finite fits of', len(rows))


if __name__ == '__main__':
    main()
