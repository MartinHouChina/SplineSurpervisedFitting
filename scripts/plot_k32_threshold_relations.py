"""Final-only K32 plots with verified spline errors and complete denominators."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from scipy.interpolate import BSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
from benchmark_geometry import load_geometry_artifact
from plot_dual_error_comparison import _geometry
from plot_paper_evidence import validate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/comparisons/k32_threshold_relations_fresh_20260923_r2'
SOURCES = ['All','Synthetic','UJI','NaturalEarth','USGS','IndustrialOffset']
TITLES = ['All curves (n=61)','Synthetic (n=21)','UJI handwriting (n=10)','Natural Earth (n=10)','USGS contours (n=10)','Industrial offsets (n=10)']
NAMES = ['Ours','Park','Liang','Dung','Kang','Luo']
COLORS = ['#00877a','#5977ac','#d59638','#9668a0','#c05c64','#777777']
MARKERS = ['o','s','^','D','v','P']


def main():
    path = OUT/'evidence.json'
    report=json.loads(path.read_text())
    validate(report)
    meta=report['metadata']
    assert hashlib.sha256(Path(meta['checkpoint']).read_bytes()).hexdigest()==meta['checkpoint_sha256']
    rows=[r for r in report['measurements'] if r['stage']=='pruned']
    checked=0
    failed=[]
    for row in rows:
        if row['status']!='ok':
            failed.append({k:row.get(k) for k in ('dataset','sample_id','method','mse_tolerance','error')})
            continue
        _,original,fit=_geometry(OUT,row)
        doc,_=load_geometry_artifact(OUT,row['geometry_artifact'])
        pred=BSpline(fit['full_knot_vector'],fit['control_points_normalized'],doc['spline']['degree'])(fit['sample_parameters'])
        sq=np.square(pred-original['input_points_normalized']).sum(-1)
        np.testing.assert_allclose([sq.mean(),sq.max()],[row['mse'],row['max_squared_error']],rtol=1e-8,atol=1e-12)
        assert row['final_k']<=32
        np.testing.assert_allclose(row['total_ms'],sum(row['timing_components'].values()),rtol=1e-12)
        assert row['joint_pass']==(row['mse']<=row['mse_tolerance'] and row['max_squared_error']<=10*row['mse_tolerance'])
        checked+=1
    (OUT/'geometry_audit.json').write_text(json.dumps(dict(complete=True,checked_final_geometries=checked,failed_cases=failed,report_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2))
    summary=json.loads((OUT/'summary.json').read_text())['summary']
    summary=[r for r in summary if r['stage']=='pruned']
    eps=meta['mse_tolerances']
    ticks=[r'$10^{-5}$',r'$2\times10^{-5}$',r'$5\times10^{-5}$',r'$10^{-4}$']
    plt.rcParams.update({'font.size':11,'svg.fonttype':'none'})
    configs=[('knots_vs_threshold','final_k_mean','Mean final internal-knot count K','Specified MSE tolerance'),
             ('mean_peak_vs_threshold','max_squared_error_mean','Specified MSE tolerance','Mean per-curve maximum squared error'),
             ('worst_peak_vs_threshold','max_squared_error_max','Specified MSE tolerance','Worst single-point squared error'),
             ('time_vs_threshold','total_ms_mean','Specified MSE tolerance','Mean total runtime (ms)'),
             ('pass_vs_threshold','pass_percent','Specified MSE tolerance','Dual-bound pass rate (%)')]
    for name,key,xlabel,ylabel in configs:
        fig,axes=plt.subplots(2,3,figsize=(16,9.5))
        for ax,source,title in zip(axes.flat,SOURCES,TITLES):
            for j,method in enumerate(meta['methods']):
                series=sorted([r for r in summary if r['dataset']==source and r['method']==method],key=lambda r:r['mse_tolerance'])
                assert len(series)==4
                vals=[r[key] for r in series]
                x,y=(vals,eps) if key=='final_k_mean' else (eps,vals)
                ax.plot(x,y,color=COLORS[j],marker=MARKERS[j],lw=2.8 if j==0 else 1.4,ms=7 if j==0 else 5,
                        linestyle='-' if j==0 else '--',label=NAMES[j],zorder=10 if j==0 else 3)
            if key=='final_k_mean':
                ax.set_yscale('log'); axis=ax.yaxis
            else:
                ax.set_xscale('log'); axis=ax.xaxis
                if key!='pass_percent': ax.set_yscale('log')
            axis.set_major_locator(FixedLocator(eps)); axis.set_major_formatter(FixedFormatter(ticks)); axis.set_minor_locator(NullLocator())
            if key in ('max_squared_error_mean','max_squared_error_max'):
                ax.plot(eps,np.array(eps)*10,':',color='black',lw=1.2,label='Point-error bound (10 epsilon)')
            if key=='pass_percent': ax.set_ylim(-3,105)
            ax.set(title=title,xlabel=xlabel,ylabel=ylabel)
            ax.grid(True,alpha=.2); ax.spines[['top','right']].set_visible(False)
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=4 if len(labels)>6 else 6,bbox_to_anchor=(.5,.965),frameon=False)
        fig.suptitle('Fresh K32 comparison: '+ylabel,fontsize=17,y=.995)
        fig.text(.5,.027,'Fixed 61-curve cohort; 32-knot cap. All methods: insertion repair; Ours additionally: post-pruning.',ha='center',fontsize=10)
        fig.text(.5,.008,'Errors and node counts include finite tolerance misses; pass rate counts all requests. Runtime includes all numerical postprocessing.',ha='center',fontsize=10)
        fig.subplots_adjust(top=.845,bottom=.13,hspace=.43,wspace=.35)
        for ext in ('png','svg'): fig.savefig(OUT/(name+'.'+ext),dpi=250)
        plt.close(fig)
    lines=['# Fresh K32 threshold sweep','',meta['protocol'],'','61 curves, not 10% of datasets. Errors are normalized squared Euclidean distances, no square root.','',
           'Mean MaxSE averages per-curve pointwise maxima; Worst MaxSE is the maximum across all points and curves. Time includes full deployment/postprocessing.','']
    for source in SOURCES:
        lines += ['## '+source,'','| MSE tolerance | Method | Finite/all | Mean K | Mean MSE | Mean MaxSE | Worst MaxSE | Total ms | Pass (%) |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for eps_i in eps:
            for method,label in zip(meta['methods'],NAMES):
                r=next(r for r in summary if r['dataset']==source and r['method']==method and r['mse_tolerance']==eps_i)
                lines.append(f"| {eps_i:g} | {label} | {r['n_finite']}/{r['n']} | {r['final_k_mean']:.3f} | {r['mse_mean']:.6e} | {r['max_squared_error_mean']:.6e} | {r['max_squared_error_max']:.6e} | {r['total_ms_mean']:.2f} | {r['pass_percent']:.1f} |")
        lines.append('')
    (OUT/'results.md').write_text('\n'.join(lines),encoding='utf-8')
    print('COMPLETE: verified',checked,'final curves;',len(failed),'failed fits;',OUT)


if __name__=='__main__': main()
