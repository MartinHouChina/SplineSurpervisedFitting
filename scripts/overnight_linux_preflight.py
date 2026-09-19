"""Check the historical overnight environment/data before a long Linux run.

This is orchestration only: no network weights or training objectives are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from overnight_datasets import (
    VALIDATION_SOURCE_NAMES, default_manifests, parse_manifests,
    source_description, validate_source_records,
)


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def checkpoint_record(path, *, deployment, tolerance, candidate_knots=64, resize_candidates=False):
    import torch
    from spline_fitting.checkpointing import (
        V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION, build_model_from_checkpoint,
    )

    source = Path(path).resolve()
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if deployment:
        if payload.get("objective_version") != V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION:
            raise ValueError("evaluation checkpoint is not the historical native v16 objective")
        _, config, _ = build_model_from_checkpoint(payload)
        source_capacity = config.get("max_internal_knots")
        if resize_candidates:
            if not isinstance(source_capacity, int) or source_capacity <= candidate_knots:
                raise ValueError("candidate warm-start resizing requires a strictly smaller target capacity")
        elif source_capacity != candidate_knots:
            raise ValueError(f"overnight comparison requires a {candidate_knots}-internal-candidate checkpoint; "
                             "a baseline cap does not resize network weights")
        recorded = payload.get("training_config", {}).get("mse_tolerance")
        if recorded != tolerance:
            raise ValueError(f"checkpoint MSE tolerance {recorded} != requested {tolerance}")
    return {
        "path": str(source), "sha256": file_hash(source),
        "epoch": payload.get("epoch"), "stage": payload.get("stage"),
        "objective_version": payload.get("objective_version"),
        "training_config": payload.get("training_config", {}),
        "real_data_provenance": payload.get("real_data_provenance", []),
        "source_candidate_knots": payload.get("model_config", {}).get("max_internal_knots"),
        "requested_candidate_knots": candidate_knots,
        "resize_candidate_warm_start": resize_candidates,
        "note": ("Evaluation of existing weights; unknown ancestor exposure is not inferred."
                 if deployment else
                 "Warm-start source provenance; real_fraction=0 in the new run does not erase prior exposure."),
    }


def resume_status(path, training_arguments):
    """Validate a same-run resume on CPU, without datasets or training.

The caller passes the exact train_v16 arguments after ``--``. Reusing its
parser/serializer keeps defaults and immutable configuration checks aligned
with the historical trainer even when training has already finished.
"""
    import torch
    from collections.abc import Mapping
    from spline_fitting.checkpointing import (
        V16_ADAPTIVE_SELECTION_REVISION,
        V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        V16_SIMPLIFICATION_CONTRACT,
        build_model_from_checkpoint,
    )
    from train_v16 import (
        EPOCH_SEED_STRIDE, parser, serial_args, training_config_changes, validate_args,
    )

    args = parser().parse_args(training_arguments)
    validate_args(args)
    output = args.output.resolve()
    source = Path(path).resolve()
    expected_last = output.with_name(output.stem + ".last.pt")
    if source != expected_last or args.resume is None or args.resume.resolve() != source:
        raise ValueError("resume status requires this output's original .last.pt and matching --resume")
    if args.real_fraction != 0:
        raise ValueError("overnight resume requires the original synthetic-only training run")
    current = serial_args(args)
    current.update(train_seed=args.seed, train_seed_stride=EPOCH_SEED_STRIDE)
    ignored = {"epochs", "resume", "init_checkpoint", "warm_start_checkpoint", "output", "device", "num_workers",
               "torch_num_threads", "log_every_batches"}

    def checked_payload(checkpoint_path):
        if not checkpoint_path.is_file():
            raise ValueError(f"resume requires saved checkpoint artifact: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError(f"resume checkpoint is not a mapping: {checkpoint_path}")
        for key, expected in (
            ("objective_version", V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION),
            ("architecture_revision", V16_ADAPTIVE_SELECTION_REVISION),
            ("simplification_contract", V16_SIMPLIFICATION_CONTRACT),
        ):
            if payload.get(key) != expected:
                raise ValueError(f"resume requires the historical native v16 {key}: {checkpoint_path}")
        previous = payload.get("training_config")
        if not isinstance(previous, Mapping):
            raise ValueError(f"resume checkpoint has no training_config: {checkpoint_path}")
        saved_output = previous.get("output")
        if (not isinstance(saved_output, str) or not Path(saved_output).is_absolute()
                or Path(saved_output).resolve() != output):
            raise ValueError("resume must use the original absolute --output; relocated Windows checkpoints cannot resume")
        if previous.get("mse_tolerance") != args.mse_tolerance:
            raise ValueError("resume checkpoint MSE tolerance does not match the requested run")
        records = payload.get("real_data_provenance", [])
        expected_manifests = {str(path.resolve()) for path in args.real_manifest}
        if not isinstance(records, list) or any(
            not isinstance(record, Mapping)
            or not isinstance(record.get("manifest"), str)
            or not isinstance(record.get("manifest_sha256"), str)
            for record in records
        ):
            raise ValueError("resume real-data provenance is malformed")
        if {
            str(Path(record["manifest"]).resolve()) for record in records
        } != expected_manifests:
            raise ValueError("resume real-data provenance differs from the requested validation sources")
        for record in records:
            if file_hash(record["manifest"]) != record["manifest_sha256"]:
                raise ValueError("resume validation manifest changed since the saved experiment")
        model_config = payload.get("model_config", {})
        if (not isinstance(model_config, Mapping)
                or model_config.get("max_internal_knots") != args.candidate_knots):
            raise ValueError(f"resume capacity mismatch: requires a {args.candidate_knots}-internal-candidate checkpoint; "
                             "capacity cannot change during resume")
        if model_config.get("mse_tolerance") != args.mse_tolerance:
            raise ValueError("resume model MSE tolerance does not match the requested run")
        # Strict CPU restoration rejects incompatible or corrupt state tensors;
        # no optimizer step, forward pass, CUDA access, or source mutation occurs.
        build_model_from_checkpoint(payload)
        return payload, previous

    payload, previous = checked_payload(source)
    stage = payload.get("stage")
    if stage not in {"proposal", "joint"}:
        raise ValueError("resume checkpoint has an invalid training stage")
    epoch = payload.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError("resume checkpoint has an invalid completed epoch")
    if not isinstance(previous.get("epochs"), int) or epoch > previous["epochs"]:
        raise ValueError("resume checkpoint epoch exceeds its saved training budget")
    if stage == "proposal" and args.proposal_epochs >= previous.get("proposal_epochs", args.proposal_epochs + 1):
        ignored.add("proposal_epochs")
    changed = training_config_changes(current, previous, ignored)
    if changed:
        raise ValueError(f"resume data/training configuration mismatch: {changed}")
    complete = epoch >= args.epochs
    proposal = output.with_name(output.stem + ".proposal.pt")
    companions = [proposal]
    if stage == "joint" or complete:
        companions.append(output)
    for companion in companions:
        _, saved = checked_payload(companion)
        # A best/proposal artifact can predate a legal Proposal-stage extension.
        changed = training_config_changes(current, saved, ignored | {"proposal_epochs"})
        if changed:
            raise ValueError(f"resume companion configuration mismatch in {companion}: {changed}")
    return "completed" if complete else "needed"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--validate-real-splits", action="store_true",
                        help="Check nonempty disjoint train/val/test manifests before reliable training")
    parser.add_argument("--resume-status", action="store_true",
                        help="Validate --checkpoint as same-run .last.pt; print completed or needed")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--manifest", action="append", default=[], metavar="NAME=PATH",
                        help="Additional evaluation-only source; the four default sources remain required")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--initializer", type=Path)
    parser.add_argument("--warm-start-checkpoint", type=Path)
    parser.add_argument("--mse-tolerance", type=float, default=5e-5)
    parser.add_argument("--candidate-knots", type=int, default=64,
                        help="Expected shared internal-knot capacity; evaluated weights must match")
    parser.add_argument("--resize-candidate-warm-start", action="store_true",
                        help="Allow a larger full-model initializer, never an evaluation/resume mismatch")
    parser.add_argument("--output", type=Path)
    parser.add_argument("training_arguments", nargs=argparse.REMAINDER,
                        help="For --resume-status only: exact train_v16 options after --")
    args = parser.parse_args(argv)
    if not 24 <= args.candidate_knots <= 188:
        parser.error("candidate-knots must be 24..188 for the overnight source range and 192 points")
    if args.resize_candidate_warm_start and (
        args.warm_start_checkpoint is None or args.checkpoint is not None
        or args.initializer is not None or args.resume_status
    ):
        parser.error("resize-candidate-warm-start requires only a full-model warm-start checkpoint")
    if sys.version_info < (3, 11):
        parser.error("the complete historical benchmark requires Python 3.11 or newer")
    if args.resume_status:
        if args.runtime_only or args.checkpoint is None or not args.training_arguments:
            parser.error("--resume-status requires --checkpoint and train_v16 arguments after --")
        training_arguments = args.training_arguments
        if training_arguments[0] != "--":
            parser.error("resume training arguments must follow --")
        try:
            status = resume_status(args.checkpoint, training_arguments[1:])
        except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
            parser.error(str(error))
        print(status, flush=True)
        return status
    if args.training_arguments:
        parser.error("extra training arguments require --resume-status")
    import torch
    import matplotlib
    import scipy
    from spline_fitting.data.real_world import (
        load_manifest_points, read_curve_manifest, resolve_points_path,
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable in this Python/PyTorch environment")
    record = {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
        "requested_device": args.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if args.runtime_only:
        print(json.dumps(record, ensure_ascii=False), flush=True)
        return record
    record.update(candidate_knots=args.candidate_knots,
                  full_cubic_knot_vector_cap=args.candidate_knots + 8)
    manifests = parse_manifests(args.manifest, defaults=default_manifests(args.data_root))
    record["datasets"] = {}
    if args.validate_real_splits:
        from spline_fitting.data.v16_mixed import load_real_sources
        _, provenance = load_real_sources(
            [manifests[name] for name in VALIDATION_SOURCE_NAMES], num_points=192, point_dim=2,
            progress=lambda message: print(message, flush=True),
        )
        record["validation_provenance"] = provenance
        record["real_data_role"] = "validation_only; held-out test excluded from model selection"
    seen_sources = set()
    for name, manifest in manifests.items():
        print(f"{name}: reading required manifest {manifest}", flush=True)
        records = read_curve_manifest(manifest)
        validate_source_records(name, records, seen_sources=seen_sources)
        rows = [row for row in records if row["split"] == "test"]
        for index, row in enumerate(records, 1):
            point_path = resolve_points_path(manifest, row)
            if not point_path.is_file():
                raise FileNotFoundError(
                    f"{name}: missing point file {point_path}; copy the complete data tree, "
                    "not only JSONL manifests (Windows absolute paths are not Linux paths)"
                )
            if index % 2000 == 0:
                print(f"{name}: checked {index}/{len(records)} point-file paths", flush=True)
        points = load_manifest_points(manifest, rows[0])
        if points.shape[-1] != 2 or not torch.isfinite(points).all():
            raise ValueError(f"{name}: expected finite 2D ordered points")
        record["datasets"][name] = {
            "manifest": str(manifest.resolve()), "sha256": file_hash(manifest),
            "test_curves": len(rows), "test_groups": len({row["group_id"] for row in rows}),
            **source_description(name, records),
            "used_for_validation_by_this_runner": bool(args.validate_real_splits and name in VALIDATION_SOURCE_NAMES),
            "evaluation_role": "held_out_test_only; extra sources never added to training or checkpoint selection",
        }
        print(f"{name}: {len(rows)} held-out test curves; point paths checked", flush=True)
    if args.checkpoint:
        record["evaluation_checkpoint"] = checkpoint_record(
            args.checkpoint, deployment=True, tolerance=args.mse_tolerance,
            candidate_knots=args.candidate_knots,
        )
    if args.initializer:
        record["warm_start"] = checkpoint_record(
            args.initializer, deployment=False, tolerance=args.mse_tolerance,
            candidate_knots=args.candidate_knots,
        )
        print("Warm-start provenance recorded; this is not a from-scratch study.", flush=True)
    if args.warm_start_checkpoint:
        record["full_model_warm_start"] = checkpoint_record(
            args.warm_start_checkpoint, deployment=True, tolerance=args.mse_tolerance,
            candidate_knots=args.candidate_knots, resize_candidates=args.resize_candidate_warm_start,
        )
        record["full_model_warm_start"]["note"] = (
            "All compatible model weights initialize a new run; not optimizer/epoch resume. "
            "Ancestor exposure and checkpoint selection provenance remain relevant."
        )
        print("Full-model initialization provenance recorded; strict configuration checks follow in training.", flush=True)
        if args.resize_candidate_warm_start:
            print("Explicit capacity migration: candidate queries will be resized for a NEW training run; "
                  "the old model is not a smaller-capacity deployment checkpoint.", flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n",
                               encoding="utf-8")
    return record


if __name__ == "__main__":
    main()
