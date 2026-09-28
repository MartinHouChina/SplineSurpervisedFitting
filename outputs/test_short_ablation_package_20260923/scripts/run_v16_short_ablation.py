"""Short, paired V16 ablations; reuse weights, never restart Proposal training.

This is a diagnostic experiment, not a new architecture or a six-baseline run.
Only synthetic TRAIN curves supervise optimization. External validation/test
curves are disjoint, and the held-out test never selects an epoch or an arm.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from overnight_datasets import default_manifests, validate_source_records
from spline_fitting.checkpointing import (
    build_model_from_checkpoint, V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
)
from spline_fitting.data.real_world import RealWorldCurveDataset, read_curve_manifest
from spline_fitting.data.synthetic import SyntheticCubicBSplineDataset
from spline_fitting.evaluation.bspline_inference import refit_bspline_control_points
from spline_fitting.training.v16_short_ablation import configure_ablation, ablation_loss
from spline_fitting.training.v16_ablation_teacher import build_fixed_teacher


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/checkpoints/overnight_stable_k32_3090_r1.pt")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--train-size", type=int, default=256)
    p.add_argument("--val-size", type=int, default=64)
    p.add_argument("--test-size", type=int, default=64)
    p.add_argument("--real-per-source", type=int, default=8)
    p.add_argument("--skip-real-data", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-points", type=int, default=192)
    p.add_argument("--min-source-knots", type=int, default=4)
    p.add_argument("--max-source-knots", type=int, default=24)
    p.add_argument("--max-internal-knots", type=int, default=32)
    p.add_argument("--teacher-max-deletions", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--decoder-lr", type=float, default=1e-5)
    p.add_argument("--count-weight", type=float, default=0.5)
    p.add_argument("--parameter-weight", type=float, default=1.0)
    p.add_argument("--knot-weight", type=float, default=1.0)
    p.add_argument("--fit-weight", type=float, default=0.1)
    p.add_argument("--mse-tolerance", type=float, default=5e-5)
    p.add_argument("--max-squared-error-tolerance", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=230923)
    p.add_argument("--torch-num-threads", type=int, default=4)
    p.add_argument("--modes", nargs="+", choices=("fixed", "legacy", "geometry"), default=["fixed", "legacy", "geometry"])
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(argv)
    for name in ("epochs", "train_size", "val_size", "test_size", "batch_size", "num_points", "max_internal_knots", "torch_num_threads"):
        if getattr(a, name) < 1:
            p.error(f"{name} must be positive")
    if a.real_per_source < 0 or a.teacher_max_deletions < 0:
        p.error("real-per-source and teacher-max-deletions must be nonnegative")
    if not 0 <= a.min_source_knots <= a.max_source_knots <= a.max_internal_knots:
        p.error("require 0 <= min-source-knots <= max-source-knots <= max-internal-knots")
    if a.max_internal_knots > 32:
        p.error("this bounded ablation is limited to at most 32 internal candidates")
    if a.num_points < a.max_internal_knots + 4:
        p.error("num-points must be at least internal capacity + 4 for cubic least squares")
    for name in ("learning_rate", "decoder_lr", "mse_tolerance", "max_squared_error_tolerance"):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) <= 0:
            p.error(f"{name} must be finite and positive")
    for name in ("count_weight", "parameter_weight", "knot_weight", "fit_weight"):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) < 0:
            p.error(f"{name} must be finite and nonnegative")
    if a.seed < 0 or len(set(a.modes)) != len(a.modes):
        p.error("seed must be nonnegative; modes must not repeat")
    return a


def build_panels(a):
    panels = {}
    for split, size, offset in (("train", a.train_size, 0), ("val", a.val_size, 1000000), ("test", a.test_size, 2000000)):
        cases = []
        # Source K is a generation parameter, NOT a certified minimal label.
        # No canonicalization/expensive rejection is needed: the shared teacher
        # solves the actual proposed representation of these observed points.
        for i in range(size):
            k = a.min_source_knots + i % (a.max_source_knots - a.min_source_knots + 1)
            seed = a.seed * 10000000 + offset + i
            sample = SyntheticCubicBSplineDataset(
                size=1, num_points=a.num_points, min_control_points=k + 4,
                max_control_points=k + 4, seed=seed, noise_std=.001,
                canonical_knot_tolerance=0.0, certified_minimal_source=False,
                cache_samples=False,
            )[0]
            cases.append(dict(dataset="Synthetic", sample_id=f"seed{seed}", group_id=f"seed{seed}",
                              split=split, source_k=k, points=sample["points"].cpu(),
                              source_kind="synthetic_not_certified_minimal"))
        panels[split] = cases
    if not a.skip_real_data and a.real_per_source:
        for name, manifest in default_manifests(a.data_root).items():
            if not manifest.is_file():
                raise FileNotFoundError(f"Required manifest missing: {manifest}. Supply --data-root or explicitly --skip-real-data for a synthetic smoke run.")
            validate_source_records(name, read_curve_manifest(manifest))
            for split in ("val", "test"):
                data = RealWorldCurveDataset(manifest, split=split, num_points=a.num_points)
                indices = list(range(len(data)))
                random.Random(f"{a.seed}:{name}:{split}").shuffle(indices)
                picked, groups = [], set()
                for index in indices:
                    if data.records[index]["group_id"] not in groups:
                        picked.append(index)
                        groups.add(data.records[index]["group_id"])
                    if len(picked) == a.real_per_source:
                        break
                if len(picked) < a.real_per_source:
                    raise ValueError(f"{name}/{split}: need {a.real_per_source} independent groups, found {len(picked)}")
                for index in picked:
                    sample = data[index]
                    panels[split].append(dict(dataset=name, sample_id=sample["curve_id"], group_id=sample["group_id"],
                                              split=split, source_k=None, points=sample["points"].cpu(),
                                              source_kind="procedural_CAD_not_measured" if name == "IndustrialOffset" else "external_geometry"))
    validate_panels(panels)
    return panels


def validate_panels(panels):
    seen_groups, seen_ids, seen_points = {}, {}, {}
    for split, cases in panels.items():
        if not cases:
            raise ValueError(f"Empty panel: {split}")
        for case in cases:
            if case["split"] != split:
                raise ValueError("Panel split mismatch")
            if split == "train" and case["dataset"] != "Synthetic":
                raise ValueError("This ablation prohibits real-data optimization")
            for value, seen in (((case["dataset"], case["group_id"]), seen_groups),
                                ((case["dataset"], case["sample_id"]), seen_ids),
                                (tensor_hash(case["points"]), seen_points)):
                if value in seen and seen[value] != split:
                    raise ValueError(f"Train/validation/test leakage: {value}")
                seen[value] = split


def panel_metadata(panels):
    return {split: [{**{k: v for k, v in c.items() if k != "points"}, "points_sha256": tensor_hash(c["points"])} for c in cases]
            for split, cases in panels.items()}


def build_cache(model, cases, path, a, protocol_hash):
    fingerprint = dict(protocol_sha256=protocol_hash,
                       panel=[tensor_hash(c["points"]) for c in cases])
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["fingerprint"] != fingerprint or len(payload["labels"]) != len(cases):
            raise ValueError(f"Teacher cache fingerprint mismatch: {path}")
        return payload["labels"]
    partial = path.with_suffix(".partial.pt")
    labels = []
    if partial.exists():
        payload = torch.load(partial, map_location="cpu", weights_only=True)
        if payload["fingerprint"] != fingerprint or len(payload["labels"]) > len(cases):
            raise ValueError(f"Partial teacher cache mismatch: {partial}")
        labels = payload["labels"]
    started = time.perf_counter()
    for start in range(len(labels), len(cases), a.batch_size):
        points = torch.stack([c["points"] for c in cases[start:start + a.batch_size]])
        labels.extend(build_fixed_teacher(model, points, mse_tolerance=a.mse_tolerance,
                                          max_squared_error_tolerance=a.max_squared_error_tolerance,
                                          max_deletions=a.teacher_max_deletions))
        atomic_torch(partial, dict(fingerprint=fingerprint, labels=labels))
        print(f"Teacher {path.stem} [{len(labels)}/{len(cases)}] feasible={sum(bool(r['feasible']) for r in labels)}/{len(labels)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    atomic_torch(path, dict(fingerprint=fingerprint, labels=labels))
    return labels


def summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset"]].append(row)
    def aggregate(items):
        valid = [r for r in items if r["status"] == "ok"]
        return dict(n=len(items), failures=len(items)-len(valid),
                    dual_pass_rate=sum(r.get("joint_pass", False) for r in items)/len(items),
                    mse_pass_rate=sum(r.get("fit_pass", False) for r in items)/len(items),
                    mean_k=float(np.mean([r["k"] for r in valid])) if valid else None,
                    mean_mse=float(np.mean([r["mse"] for r in valid])) if valid else None,
                    worst_max_squared_error=max((r["max_squared_error"] for r in valid), default=None),
                    mean_total_ms=float(np.mean([r["total_ms"] for r in valid])) if valid else None,
                    mean_network_ms=float(np.mean([r["network_ms"] for r in valid])) if valid else None)
    per_source = {source: aggregate(items) for source, items in groups.items()}
    return dict(overall=aggregate(rows), per_source=per_source,
                macro_dual_pass=float(np.mean([s["dual_pass_rate"] for s in per_source.values()])),
                worst_source_dual_pass=min(s["dual_pass_rate"] for s in per_source.values()))


def validation_rank(report):
    s = report["summary"]
    finite = lambda v: v if v is not None else float("inf")
    return (-s["macro_dual_pass"], -s["worst_source_dual_pass"],
            s["overall"]["failures"], finite(s["overall"]["mean_k"]), finite(s["overall"]["mean_mse"]))


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(model, cases, a, *, geometry=False, label="validation"):
    model.eval()
    weight = next(model.parameters())
    device = weight.device
    # Warm-up is outside measured calls, identical for every arm.
    try:
        model.forward_deployment(cases[0]["points"][None].to(device), mse_tolerance=a.mse_tolerance)
    except (RuntimeError, ValueError):
        pass  # The same case is evaluated and explicitly counted below.
    sync(device)
    rows = []
    for index, case in enumerate(cases):
        row = {k: v for k, v in case.items() if k != "points"}
        try:
            points = case["points"].double()
            batch = case["points"][None].to(device=device, dtype=weight.dtype)
            sync(device)
            started = time.perf_counter()
            output = model.forward_deployment(batch, mse_tolerance=a.mse_tolerance)
            sync(device)
            network_ms = (time.perf_counter()-started)*1000
            mask = output["learned_keep_mask"][0].cpu()
            params = output["params"][0].cpu().double()
            knots = output["internal_knots"][0].cpu().double()[mask]
            if knots.numel() > a.max_internal_knots:
                raise ValueError("Deployed count exceeds shared capacity")
            fit = refit_bspline_control_points(params, points, knots, degree=model.degree,
                                               interpolate_endpoints=True, smoothness_weight=0., control_ridge=0.)
            total_ms = (time.perf_counter()-started)*1000
            squared = (fit.evaluate(params)-points).square().sum(-1)
            if not torch.isfinite(squared).all():
                raise ValueError("Nonfinite deployed residual")
            mse, peak = float(squared.mean()), float(squared.max())
            row.update(status="ok", mse=mse, max_squared_error=peak, k=int(knots.numel()),
                       fit_pass=mse <= a.mse_tolerance,
                       joint_pass=mse <= a.mse_tolerance and peak <= a.max_squared_error_tolerance,
                       network_ms=network_ms, total_ms=total_ms,
                       mask=mask.tolist(), parameter_shift_max=float((params-output["proposal_params"][0].cpu().double()).abs().max()))
            if geometry:
                dense_t = torch.linspace(0, 1, 401, dtype=torch.float64)
                row["geometry"] = dict(points=points.tolist(), params=params.tolist(),
                                       internal_knots=knots.tolist(), knot_vector=fit.knot_vector.tolist(),
                                       control_points=fit.control_points.tolist(),
                                       squared_residuals=squared.tolist(),
                                       curve_points=fit.evaluate(dense_t).tolist(),
                                       knots_on_curve=fit.evaluate(knots).tolist())
        except (RuntimeError, ValueError) as error:
            row.update(status="failed", fit_pass=False, joint_pass=False,
                       error=f"{type(error).__name__}: {error}")
        rows.append(row)
        if (index+1) % 32 == 0 or index+1 == len(cases):
            print(f"  {label} [{index+1}/{len(cases)}] dual={sum(r.get('joint_pass', False) for r in rows)/len(rows):.1%}", flush=True)
    return dict(rows=rows, summary=summary(rows))


@torch.no_grad()
def teacher_replay(model, cases, labels, a):
    """Unchanged teacher masks before/after native decoder, for causal diagnosis."""
    device = next(model.parameters()).device
    rows = []
    for case, teacher in zip(cases, labels):
        row = dict(dataset=case["dataset"], sample_id=case["sample_id"],
                   teacher_feasible=bool(teacher["feasible"]), teacher_k=int(teacher["mask"].sum()))
        for output_key, teacher_key in (("mse", "mse"), ("max_squared_error", "maxse")):
            value = float(teacher[teacher_key])
            row["teacher_"+output_key] = value if math.isfinite(value) else None
        points = case["points"][None].to(device)
        try:
            context = model.encode_candidates(points, mse_tolerance=a.mse_tolerance)
            mask = teacher["mask"][None].to(device)
            output = model.decode_subset(context, mask)
            t = output["params"][0].cpu().double()
            u = output["internal_knots"][0, mask[0]].cpu().double()
            fit = refit_bspline_control_points(t, case["points"].double(), u,
                                              degree=model.degree, interpolate_endpoints=True,
                                              smoothness_weight=0., control_ridge=0.)
            squared = (fit.evaluate(t)-case["points"].double()).square().sum(-1)
            mse, peak = float(squared.mean()), float(squared.max())
            if not math.isfinite(mse) or not math.isfinite(peak):
                raise ValueError("Nonfinite redecoded error")
            row.update(decoded_mse=mse, decoded_max_squared_error=peak,
                       decoded_pass=mse <= a.mse_tolerance and peak <= a.max_squared_error_tolerance)
        except (RuntimeError, ValueError) as error:
            row.update(decoded_pass=False, error=str(error))
        rows.append(row)
    feasible = [r for r in rows if r["teacher_feasible"]]
    return dict(rows=rows, n=len(rows), fixed_teacher_pass_rate=len(feasible)/len(rows),
                decoded_pass_among_feasible_teachers=sum(r["decoded_pass"] for r in feasible)/len(feasible) if feasible else None,
                note="Teacher diagnostics on validation only; not a deployable search or a test selection score.")


def teacher_batch(labels, ids, device):
    return {key: torch.stack([torch.as_tensor(labels[i][key]) for i in ids]).to(device)
            for key in ("mask", "params", "knots", "feasible")}


def checkpoint_payload(model, original, mode, epoch, protocol, history, optimizer=None, best_rank=None):
    # Do not inherit ancestor's eligibility, best epoch, or training metrics.
    return dict(objective_version=original["objective_version"], model_config=model.get_config(),
                model_state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                stage="short_ablation", epoch=epoch, ablation_mode=mode,
                formal_qualification=False, role="diagnostic_short_finetune_not_formal_paper_model",
                parent_checkpoint_sha256=protocol["checkpoint_sha256"],
                training_config=protocol["arguments"], ablation_protocol=protocol,
                history=history, optimizer_state_dict=optimizer.state_dict() if optimizer else None,
                best_rank=list(best_rank) if best_rank is not None else None)


def run_arm(mode, original, panels, teachers, a, protocol):
    torch.manual_seed(a.seed)
    model, _, _ = build_model_from_checkpoint(original)
    model.to(a.device).eval()
    metadata = configure_ablation(model, mode)
    named = dict(model.named_parameters())
    groups = [dict(params=[named[n] for n in metadata["selector_parameter_names"]], lr=a.learning_rate)]
    if metadata["decoder_parameter_names"]:
        groups.append(dict(params=[named[n] for n in metadata["decoder_parameter_names"]], lr=a.decoder_lr))
    optimizer = torch.optim.AdamW(groups, weight_decay=0.)
    folder = a.output_dir / mode
    folder.mkdir(exist_ok=True)
    last_path, best_path = folder / "last.pt", folder / "best.pt"
    atomic_json(folder / "trainable_parameters.json", metadata)
    history, start, best_rank = [], 1, None
    if a.resume and last_path.exists():
        last = torch.load(last_path, map_location=a.device, weights_only=True)
        if last["ablation_protocol"] != protocol or last["ablation_mode"] != mode:
            raise ValueError(f"Resume mismatch: {last_path}")
        model.load_state_dict(last["model_state_dict"], strict=True)
        optimizer.load_state_dict(last["optimizer_state_dict"])
        history, start, best_rank = last["history"], int(last["epoch"])+1, tuple(last["best_rank"])
        if not best_path.exists():
            raise FileNotFoundError(f"Resume requires saved validation-best checkpoint: {best_path}")
    else:
        report = evaluate(model, panels["val"], a, label=f"{mode} epoch0 val")
        best_rank = validation_rank(report)
        history.append(dict(epoch=0, train=None, validation=report["summary"], selected=True))
        atomic_torch(best_path, checkpoint_payload(model, original, mode, 0, protocol, history, best_rank=best_rank))
        atomic_torch(last_path, checkpoint_payload(model, original, mode, 0, protocol, history, optimizer, best_rank))
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    for epoch in range(start, a.epochs+1):
        model.eval()  # Frozen Proposal is deterministic; gradients still enabled.
        ids = list(range(len(panels["train"])))
        random.Random(a.seed+epoch).shuffle(ids)
        aggregate, batches, skipped = defaultdict(float), 0, 0
        started = time.perf_counter()
        for offset in range(0, len(ids), a.batch_size):
            selected = ids[offset:offset+a.batch_size]
            points = torch.stack([panels["train"][i]["points"] for i in selected]).to(a.device)
            target = teacher_batch(teachers["train"], selected, torch.device(a.device))
            if not bool(target["feasible"].any()):
                skipped += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = ablation_loss(model, points, target, mode,
                                         mse_tolerance=a.mse_tolerance,
                                         max_squared_error_tolerance=a.max_squared_error_tolerance,
                                         count_weight=a.count_weight, parameter_weight=a.parameter_weight,
                                         knot_weight=a.knot_weight, fit_weight=a.fit_weight)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Nonfinite loss in {mode}/{epoch}; no optimizer step taken")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            if not bool(torch.isfinite(norm)):
                raise RuntimeError(f"Nonfinite gradient in {mode}/{epoch}; no optimizer step taken")
            optimizer.step()
            batches += 1
            aggregate["loss"] += float(loss.detach())
            for name, value in metrics.items():
                if name != "loss":
                    aggregate[name] += float(value)
            if batches % 4 == 0 or offset+a.batch_size >= len(ids):
                print(f"{mode} epoch {epoch}/{a.epochs} [{min(offset+a.batch_size,len(ids))}/{len(ids)}] loss={float(loss.detach()):.4f}", flush=True)
        for name, reference in frozen.items():
            if not torch.equal(named[name].detach(), reference):
                raise RuntimeError(f"Frozen parameter unexpectedly changed: {mode}/{name}")
        train = {k: v/max(batches, 1) for k, v in aggregate.items()}
        train.update(optimized_batches=batches, skipped_infeasible_batches=skipped,
                     elapsed_seconds=time.perf_counter()-started)
        val = evaluate(model, panels["val"], a, label=f"{mode} epoch{epoch} val")
        rank = validation_rank(val)
        improved = rank < best_rank
        if improved:
            best_rank = rank
        entry = dict(epoch=epoch, train=train, validation=val["summary"], selected=improved)
        history.append(entry)
        payload = checkpoint_payload(model, original, mode, epoch, protocol, history, optimizer, best_rank)
        if improved:
            atomic_torch(best_path, payload)
        atomic_torch(last_path, payload)
        atomic_json(folder / "history.json", history)
        s = val["summary"]["overall"]
        print(f"{mode} epoch {epoch}: val dual={s['dual_pass_rate']:.1%} K={s['mean_k']} MSE={s['mean_mse']} best_updated={improved}", flush=True)
    # No test results influence epoch selection. Preserve terminal as well as best.
    results = {}
    for kind, path in (("best", best_path), ("last", last_path)):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        restored, _, _ = build_model_from_checkpoint(checkpoint)
        restored.to(a.device).eval()
        report = evaluate(restored, panels["test"], a, geometry=True, label=f"{mode}/{kind} test")
        report.update(epoch=int(checkpoint["epoch"]), checkpoint=str(path), checkpoint_sha256=sha256(path))
        atomic_json(folder / f"test_{kind}.json", report)
        results[kind] = {k: report[k] for k in ("epoch", "checkpoint", "checkpoint_sha256", "summary")}
        if kind == "best":
            atomic_json(folder / "teacher_replay_validation.json", teacher_replay(restored, panels["val"], teachers["val"], a))
        del restored
    return dict(mode=mode, trainable=metadata, **results)


def write_summary(output, result):
    entries = [("baseline", "untrained", None, result["baseline"]["summary"])]
    for arm in result["arms"]:
        for kind in ("best", "last"):
            entries.append((arm["mode"], kind, arm[kind]["epoch"], arm[kind]["summary"]))
    rows = []
    for arm, kind, epoch, s in entries:
        for dataset, stats in [("Overall", s["overall"]), *s["per_source"].items()]:
            rows.append(dict(arm=arm, checkpoint_selection=kind, epoch=epoch, dataset=dataset, **stats))
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Code-native plot: all arm summaries, no hand-adjusted errors or watermark.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fields = [("dual_pass_rate", "Dual-constraint pass (%)", 100), ("mean_k", "Mean internal knots", 1),
              ("mean_mse", "Mean squared Euclidean error", 1),
              ("worst_max_squared_error", "Worst point squared error", 1), ("mean_total_ms", "Network + refit (ms)", 1)]
    for choice, filename in (("best", "ablation_comparison.png"), ("last", "ablation_last_comparison.png")):
        shown = [r for r in rows if r["dataset"] == "Overall" and r["checkpoint_selection"] in ("untrained", choice)]
        fig, axes = plt.subplots(1, 5, figsize=(18, 4.2), constrained_layout=True)
        labels = [r["arm"] + (f" (e{r['epoch']})" if r["epoch"] is not None else "") for r in shown]
        for ax, (field, title, factor) in zip(axes, fields):
            values = [r[field]*factor if r[field] is not None else float("nan") for r in shown]
            ax.bar(labels, values, color=["#777777", "#1976b5", "#db8c24", "#188c74"][:len(shown)])
            ax.set_title(title, fontsize=10)
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", alpha=.2)
            if field in ("mean_mse", "worst_max_squared_error"):
                ax.set_yscale("log")
                threshold = result["protocol"]["arguments"]["mse_tolerance" if field == "mean_mse" else "max_squared_error_tolerance"]
                ax.axhline(threshold, ls="--", color="#c33", lw=1)
        selection_label = "validation-selected checkpoints" if choice == "best" else "same terminal epoch (paired A/B selector weights checked)"
        fig.suptitle(f"Short paired ablation | {selection_label} | held-out test\nTiming: exploratory, one network forward + one standard B-spline refit; no search", fontsize=11)
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)


def paired_selector_check(output):
    """A/B terminal weights must match to isolate only decoder execution."""
    paths = [output / mode / "last.pt" for mode in ("fixed", "legacy")]
    if not all(p.exists() for p in paths):
        return dict(checked=False, reason="fixed and legacy arms were not both run")
    payloads = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    names = json.loads((output / "fixed/trainable_parameters.json").read_text(encoding="utf-8"))["selector_parameter_names"]
    differences = [name for name in names if not torch.equal(payloads[0]["model_state_dict"][name], payloads[1]["model_state_dict"][name])]
    return dict(checked=True, same_terminal_epoch=payloads[0]["epoch"] == payloads[1]["epoch"],
                selector_weights_bitwise_equal=not differences, different_parameters=differences,
                note="Use same-epoch last results for decoder isolation; validation-best epochs may differ. GPU numerical variation must be reported, not hidden.")


def main(argv=None):
    a = parse_args(argv)
    if not a.checkpoint.is_file():
        raise FileNotFoundError(f"Warm-start checkpoint missing: {a.checkpoint}")
    if a.output_dir.exists() and not a.resume:
        raise FileExistsError(f"Refusing to overwrite run: {a.output_dir}; choose a new run or --resume")
    torch.set_num_threads(a.torch_num_threads)
    torch.manual_seed(a.seed)
    if torch.device(a.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu for a smoke check")
    original = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    if original.get("objective_version") != V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION:
        raise ValueError("This isolated experiment requires a native V16 checkpoint")
    model, config, _ = build_model_from_checkpoint(original)
    if model.max_internal_knots != a.max_internal_knots or model.point_dim != 2 or model.degree != 3:
        raise ValueError("No implicit capacity/architecture transfer: checkpoint must be cubic 2D with requested internal capacity")
    if "fixed" in a.modes and len(model.coupled_proposal_blocks):
        raise ValueError("Fixed-geometry isolation requires a checkpoint without coupled Proposal blocks; use the selected Stable K32 checkpoint")
    manifests = {} if a.skip_real_data or not a.real_per_source else default_manifests(a.data_root)
    source_files = [Path(__file__), ROOT / "src/spline_fitting/training/v16_short_ablation.py",
                    ROOT / "src/spline_fitting/training/v16_ablation_teacher.py",
                    ROOT / "src/spline_fitting/models/v16_network.py",
                    ROOT / "src/spline_fitting/evaluation/bspline_inference.py",
                    ROOT / "src/spline_fitting/losses/deployment_bspline_loss.py",
                    ROOT / "src/spline_fitting/spline/bspline_deletion_teacher.py",
                    ROOT / "src/spline_fitting/data/synthetic.py",
                    ROOT / "src/spline_fitting/data/real_world.py",
                    ROOT / "src/spline_fitting/checkpointing.py",
                    ROOT / "scripts/overnight_datasets.py"]
    arguments = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(a).items()
                 if k not in ("resume", "output_dir", "checkpoint", "data_root", "device")}
    protocol = dict(schema_version=1, arguments=arguments, checkpoint_sha256=sha256(a.checkpoint),
                    parent_epoch=original.get("epoch"), parent_model_config=config,
                    manifest_sha256={k: sha256(v) for k, v in manifests.items()},
                    source_sha256={str(p.relative_to(ROOT)).replace('\\', '/'): sha256(p) for p in source_files},
                    selection="validation source-macro dual pass, worst-source dual pass, failures, K, MSE; epoch0 eligible; no test-based selection",
                    teacher="offline fixed-Proposal greedy dual-constraint subsets; infeasible teachers excluded from optimization, never evaluation",
                    error_scope="normalized input samples, squared Euclidean MSE and maximum single-point squared error, no square root",
                    limitation="Short diagnostic, not formal paper evidence. Pretrained ancestor exposure cannot be undone. CAD offsets are procedural, not measured.")
    protocol_path = a.output_dir / "protocol.json"
    if a.resume:
        if not protocol_path.exists() or json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Resume protocol/source/checkpoint changed, or run does not exist")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(protocol_path, protocol)
    atomic_json(a.output_dir / "runtime.json", dict(device=a.device, torch_version=str(torch.__version__),
                                                   gpu=torch.cuda.get_device_name() if torch.device(a.device).type == "cuda" else None))
    panel_path = a.output_dir / "panels.pt"
    if panel_path.exists():
        panels = torch.load(panel_path, map_location="cpu", weights_only=True)
        validate_panels(panels)
    else:
        panels = build_panels(a)
        atomic_torch(panel_path, panels)
    metadata = panel_metadata(panels)
    panel_json = a.output_dir / "panels.json"
    if panel_json.exists() and json.loads(panel_json.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Saved panel content changed")
    atomic_json(panel_json, metadata)
    print(f"Short ablation: source K={a.min_source_knots}..{a.max_source_knots}, capacity={a.max_internal_knots}, MSE<={a.mse_tolerance:g}, MaxSE<={a.max_squared_error_tolerance:g}")
    print(f"Proposal epochs=0; {a.epochs} fine-tuning epochs per arm. Panels: " + str({k: len(v) for k, v in panels.items()}), flush=True)
    # Align frozen Proposal execution (including attention kernel paths) with
    # the arms. No weights or original decoder behavior are changed here.
    configure_ablation(model, "legacy")
    model.to(a.device).eval()
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    teachers = {split: build_cache(model, panels[split], a.output_dir / f"teacher_{split}.pt", a, protocol_hash) for split in ("train", "val")}
    atomic_json(a.output_dir / "teacher_replay_before.json", teacher_replay(model, panels["val"], teachers["val"], a))
    baseline = evaluate(model, panels["test"], a, geometry=True, label="untrained baseline test")
    atomic_json(a.output_dir / "baseline_test.json", baseline)
    del model
    result = dict(protocol=protocol, baseline=dict(summary=baseline["summary"]), arms=[])
    for mode in a.modes:
        result["arms"].append(run_arm(mode, original, panels, teachers, a, protocol))
        atomic_json(a.output_dir / "results.json", result)
    result["paired_selector_check"] = paired_selector_check(a.output_dir)
    atomic_json(a.output_dir / "results.json", result)
    write_summary(a.output_dir, result)
    print(f"Completed. Results: {a.output_dir / 'summary.csv'}; plot: {a.output_dir / 'ablation_comparison.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
