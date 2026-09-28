"""Plot verified common-protocol threshold/count results without changing data."""
import json
import hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'outputs/comparisons/paper_evidence_k64_20260923/summary.json'
OUT = ROOT / 'outputs/figures/verified_threshold_knots_20260923'


def main():
    report = json.loads(SRC.read_text(encoding='utf-8'))
    methods = report['metadata']['methods']
    names = ['Ours', 'Park', 'Liang', 'Dung', 'Kang', 'Luo']
    rows = [r for r in report['summary'] if r['method'] in methods and r['stage'] == 'pruned']
    sources = ['All', 'Synthetic', 'UJI', 'NaturalEarth', 'USGS', 'IndustrialOffset']
    titles = ['All curves (n=61)', 'Synthetic (n=21)', 'UJI handwriting (n=10)', 'Natural Earth coastline (n=10)', 'USGS contours (n=10)', 'Industrial offsets (n=10)']
    eps = [1e-5, 2e-5, 5e-5, 1e-4]
    labels = [r'$10^{-5}$', r'$2\times10^{-5}$', r'$5\times10^{-5}$', r'$10^{-4}$']
    colors = ['#007f78', '#5977ac', '#d59638', '#9668a0', '#c05c64', '#777777']
    markers = ['o', 's', '^', 'D', 'v', 'P']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none'})
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    for ax, source, title in zip(axes.flat, sources, titles):
        for j, method in enumerate(methods):
            series = sorted([r for r in rows if r['dataset'] == source and r['method'] == method], key=lambda r:r['mse_tolerance'])
            assert len(series) == 4 and [r['mse_tolerance'] for r in series] == eps
            ax.plot([r['final_k_mean'] for r in series], eps, color=colors[j], marker=markers[j],
                    lw=2.8 if j == 0 else 1.4, ms=7 if j == 0 else 5, label=names[j],
                    linestyle='-' if j == 0 else '--', zorder=10 if j == 0 else 3)
        ax.set_title(title, fontsize=12)
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(FixedLocator(eps))
        ax.yaxis.set_major_formatter(FixedFormatter(labels))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_ylim(8.5e-6, 1.18e-4)
        ax.set_xlabel('Mean final internal-knot count K')
        ax.set_ylabel('Specified MSE tolerance')
        ax.grid(True, alpha=.22)
        ax.spines[['top', 'right']].set_visible(False)
    handles, legend = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend, loc='upper center', ncol=6, bbox_to_anchor=(.5,.953), frameon=False)
    fig.suptitle('Node budget at four prescribed error tolerances', y=.993, fontsize=18)
    fig.text(.5,.028,'Same tolerance: farther left means fewer knots. Means include finite tolerance misses; consult the paired pass-rate panel.',ha='center',fontsize=10)
    fig.text(.5,.008,'64-knot cap; common insertion + pruning for all six methods. Dual bounds: MSE <= epsilon and MaxSE <= 10 epsilon.',ha='center',fontsize=10)
    fig.subplots_adjust(top=.875, bottom=.13, wspace=.34, hspace=.4)
    for suffix in ['png', 'svg']:
        fig.savefig(OUT / ('threshold_vs_knots.' + suffix), dpi=250)
    plt.close(fig)
    fig, axes = plt.subplots(2,3,figsize=(16,8.8))
    for ax, source, title in zip(axes.flat,sources,titles):
        values = np.array([[next(r['pass_percent'] for r in rows if r['dataset']==source and r['method']==m and r['mse_tolerance']==e) for e in eps] for m in methods])
        ax.imshow(values, vmin=0,vmax=100,cmap='Blues',aspect='auto')
        ax.set_xticks(range(4),labels)
        ax.set_yticks(range(6),names)
        ax.set_title(title,fontsize=12)
        ax.set_xlabel('Specified MSE tolerance')
        for i in range(6):
            for j in range(4):
                ax.text(j,i,f'{values[i,j]:.1f}%',ha='center',va='center',color='white' if values[i,j]>65 else 'black',fontweight='bold' if i==0 else 'normal')
    fig.suptitle('Companion feasibility check: fraction satisfying both error bounds',fontsize=17)
    fig.text(.5,.015,'All requested curves remain in the pass-rate denominator. Same verified cohort and common postprocessing as the knot-count plots.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.94))
    for suffix in ['png','svg']:
        fig.savefig(OUT / ('dual_pass_rates.'+suffix),dpi=250)
    plt.close(fig)
    (OUT / 'plotted_data.json').write_text(json.dumps({'source':str(SRC),'source_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'metadata':report['metadata'],'rows':rows},indent=2),encoding='utf-8')
    print('Saved figures and',len(rows),'unchanged aggregate records to',OUT)


if __name__ == '__main__':
    main()
