"""Fresh paired K32 sweep with the historical Ours-only pruning protocol."""
from pathlib import Path
import json
import time
import hashlib
import torch
import benchmark_paper_evidence as b

OUT = b.ROOT / 'outputs/comparisons/k32_threshold_relations_fresh_20260923_r2'
CHECKPOINT = b.ROOT / 'outputs/checkpoints/overnight_stable_k32_3090_r1.pt'
EPS = [1e-5, 2e-5, 5e-5, 1e-4]


def main():
    source, cases = b.load_cases(b.DEFAULT_REPORT)
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    model, _, _ = b.build_model_from_checkpoint(ckpt)
    assert model.max_internal_knots == 32
    device = torch.device('cuda')
    torch.set_num_threads(1)
    torch.manual_seed(20260923)
    model.to(device).eval()
    meta = dict(checkpoint=str(CHECKPOINT), checkpoint_sha256=b.file_sha256(CHECKPOINT),
                max_internal_knots=32, mse_tolerances=EPS, peak_ratio=10,
                methods=list(b.PUBLISHED_METHODS), variants=[], ablation_tolerance=5e-5,
                source_report=str(b.DEFAULT_REPORT), source_sha256=b.file_sha256(b.DEFAULT_REPORT),
                protocol='Fresh raw fit + common insertion repair for all; post-pruning only for Ours. Historical K32 protocol.',
                timing_protocol='Cumulative measured network/transfer/refit or baseline plus repair plus Ours pruning; export excluded.',
                hardware=dict(gpu=torch.cuda.get_device_name(device), torch_threads=1),
                cases=[{k:c.get(k) for k in ('dataset','sample_id','input_points_sha256')} for c in cases],
                code_sha256={str(p.relative_to(b.ROOT)):b.file_sha256(p) for p in [Path(__file__), Path(b.__file__), *list((b.ROOT/'src/spline_fitting/evaluation').glob('*.py'))]})
    meta['fingerprint'] = hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = OUT/'protocol.json'
    if protocol.exists():
        assert json.loads(protocol.read_text()) == meta, 'Resume fingerprint mismatch'
    b.write_json(protocol, meta)
    for _ in range(3):
        b.network_forward(model, ckpt, cases[0]['points'],device,5e-5)
    total = len(cases)*len(EPS)*len(b.PUBLISHED_METHODS)
    done = 0
    for eps in EPS:
        for method in b.PUBLISHED_METHODS:
            for case in cases:
                dest = OUT/'runs'/b.job_id(case,method,eps)/'record.json'
                if dest.exists():
                    done += 1
                    continue
                start = time.perf_counter()
                params = None
                try:
                    if method == 'ours':
                        fit,params,components,details,_ = b.initial_network(model,ckpt,case,device,eps,3)
                        snaps,post = b.run_chain(case,method,fit,params,components,eps,10.,32)
                        details.update(post)
                    else:
                        result,duration = b.timed(lambda:b.run_published_baseline(method,case['points'],**b.baseline_options(source,eps,32)))
                        fit,params = result.fit,result.parameters
                        components = dict(baseline_ms=duration)
                        def snap(stage, f):
                            row = b.row_from_fit(case,method,f,params,sum(components.values()),eps,eps*10)
                            row.update(stage=stage,mse_tolerance=eps,max_squared_error_tolerance=eps*10,timing_components=dict(components))
                            return row,f
                        snaps = [snap('raw',fit)]
                        repaired = b.repair_knots_to_dual_tolerance(params,case['points'],fit,mse_tolerance=eps,
                                max_squared_error_tolerance=eps*10,max_internal_knots=32,interpolate_endpoints=method not in b.UNCONSTRAINED)
                        components['repair_ms'] = repaired.elapsed_repair_ms
                        snaps.append(snap('repaired',repaired.final_fit))
                        # Final snapshot is explicitly no additional pruning for baselines.
                        snaps.append(snap('pruned',repaired.final_fit))
                        details = dict(initial=result.diagnostics, pruning='not applied under historical protocol')
                except (ValueError,RuntimeError,TypeError,KeyError) as error:
                    snaps = b.failed_snapshots(case,method,eps,10.,error,(time.perf_counter()-start)*1000)
                    details = dict(error=str(error))
                b.save_job(OUT,case,method,eps,snaps,params,details,meta)
                done += 1
                last = snaps[-1][0]
                print(f"[{done}/{total}] eps={eps:g} {case['dataset']} {method} K={last['final_k']} pass={last['joint_pass']} ms={last['total_ms']:.1f}",flush=True)
    rows = []
    for p in sorted((OUT/'runs').glob('*/record.json')):
        rec=json.loads(p.read_text())
        assert rec['fingerprint']==meta['fingerprint']
        rows.extend(rec['measurements'])
    assert len(rows)==total*3
    report = dict(metadata=meta,complete=True,measurements=rows)
    b.write_json(OUT/'evidence.json',report)
    from plot_paper_evidence import summarize
    b.write_json(OUT/'summary.json',dict(metadata=meta,summary=summarize(report)))
    print('COMPLETE',OUT,flush=True)


if __name__ == '__main__':
    main()
