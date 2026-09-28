"""Derived count adjustment, not a rerun or replacement of measured results."""
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
from plot_k32_threshold_relations import OUT, SOURCES, TITLES, NAMES, COLORS, MARKERS


def main():
    source = OUT / 'summary.json'
    data = json.loads(source.read_text())
    meta = data['metadata']
    eps = meta['mse_tolerances']
    ticks = [r'$10^{-5}$', r'$2\times10^{-5}$', r'$5\times10^{-5}$', r'$10^{-4}$']
    plt.rcParams.update({'font.size': 11, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    derived = []
    for ax, dataset, title in zip(axes.flat, SOURCES, TITLES):
        for j, method in enumerate(meta['methods']):
            rows = sorted([r for r in data['summary'] if r['dataset'] == dataset and r['method'] == method and r['stage'] == 'pruned'], key=lambda r: r['mse_tolerance'])
            assert len(rows) == 4
            offset = 2 if method == 'dung_direct_knot_2017_adaptation' else 0
            x = [r['final_k_mean'] + offset for r in rows]
            for r, adjusted in zip(rows, x):
                derived.append(dict(dataset=dataset, method=method, mse_tolerance=r['mse_tolerance'], measured_mean_k=r['final_k_mean'], displayed_mean_k=adjusted, offset=offset))
            ax.plot(x, eps, color=COLORS[j], marker=MARKERS[j], lw=2.8 if j == 0 else 1.4,
                    ms=7 if j == 0 else 5, linestyle='-' if j == 0 else '--', label=NAMES[j], zorder=10 if j == 0 else 3)
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(FixedLocator(eps))
        ax.yaxis.set_major_formatter(FixedFormatter(ticks))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set(title=title, xlabel='Adjusted mean internal-knot count K', ylabel='Specified MSE tolerance')
        ax.grid(True, alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    assert labels == NAMES
    fig.legend(handles, labels, loc='upper center', ncol=6, bbox_to_anchor=(.5, .965), frameon=False)
    fig.suptitle('K32 knot-count adjustment: Dung +2', fontsize=17, y=.995)
    fig.subplots_adjust(top=.845, bottom=.10, hspace=.43, wspace=.35)
    for ext in ('png', 'svg'):
        fig.savefig(OUT / ('knots_vs_threshold_dung_plus2.' + ext), dpi=250)
    plt.close(fig)
    assert sum(r['offset'] == 2 for r in derived) == 24
    (OUT / 'knots_vs_threshold_dung_plus2.json').write_text(json.dumps(dict(
        source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        operation='User-requested fixed +2 to Dung mean internal-knot counts only; no new fitting, timing, feasibility evaluation, or verified capacity claim.',
        rows=derived), indent=2), encoding='utf-8')
    print('Exported adjusted PNG/SVG; original measurements and figures unchanged.')


if __name__ == '__main__':
    main()
