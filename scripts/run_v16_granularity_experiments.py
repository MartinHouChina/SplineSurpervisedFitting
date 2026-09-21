"""Run controlled depth/point-error ablations and synthetic morphological specialists.

Default ``--route all`` schedules TWELVE independent training + six-method
benchmark + visualization pipelines, sequentially; this is not a one-night
runtime promise. Specialist training is synthetic-only. Real train splits and
test-case routing are not used; the named real source is used for validation.
Every model is still tested on Synthetic K=4..24 and all four external sources.
Smaller-capacity models intentionally retain those out-of-training-range tests.

``--dry-run`` only prints the plan; it never loads a checkpoint, writes files,
prepares data, or launches the shell. All actual subprocess calls use argv lists
with shell=False. A failed pipeline stops the experiment sequence immediately.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEPTH_VARIANTS = ("baseline", "deep", "peak", "deep_peak")
DOMAINS = ("UJI", "IndustrialOffset", "NaturalEarth", "USGS")
DOMAIN_SETTINGS = {
    "UJI": ("handwriting", 16, "uji"),
    "IndustrialOffset": ("industrial", 24, "industrial_offset"),
    "NaturalEarth": ("terrain", 24, "natural_earth"),
    "USGS": ("terrain", 24, "usgs"),
}
PEAK_ARGS = ["--max-point-error-weight", "0.05", "--max-point-error-tolerance", "5e-4",
             "--max-point-error-tail-fraction", "0.05"]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--route", choices=("depth", "specialists", "all"), default="all")
    result.add_argument("--depth-variants", choices=DEPTH_VARIANTS, nargs="+",
                        default=list(DEPTH_VARIANTS))
    result.add_argument("--domains", choices=DOMAINS, nargs="+", default=list(DOMAINS))
    result.add_argument("--warm-start-checkpoint", type=Path, required=True,
                        help="Common shallow K32 initializer; not an optimizer resume")
    result.add_argument("--data-root", type=Path, default=None,
                        help="Shared data tree; default repository data/")
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    result.add_argument("--run-prefix", required=True,
                        help="Fresh artifact prefix, e.g. granularity_k32_r1; never overwritten")
    result.add_argument("--epochs", type=int, default=24, help="Total epochs per model, not Joint alone")
    result.add_argument("--proposal-epochs", type=int, default=4)
    result.add_argument("--joint-geometry-calibration-epochs", type=int, default=4)
    result.add_argument("--train-size", type=int, default=1500)
    result.add_argument("--val-size", type=int, default=500)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--benchmark-profile", choices=("quick", "full"), default="full")
    result.add_argument("--real-samples-per-dataset", type=int, default=20)
    result.add_argument("--visual-samples-per-dataset", type=int, default=6)
    result.add_argument("--prepare-real-data", action="store_true")
    result.add_argument("--dry-run", action="store_true", help="Print only; no reads of weights, writes, or execution")
    result.add_argument("--bash", default="bash", help="Bash executable; never passed through shell=True")
    result.add_argument("--python", default=sys.executable, help="Python executable forwarded to each Bash pipeline")
    return result


def validate_args(args) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_prefix) or ".." in args.run_prefix:
        raise ValueError("run-prefix must be a plain artifact name, without paths, spaces, or '..'")
    for field in ("epochs", "train_size", "val_size", "batch_size",
                  "real_samples_per_dataset", "visual_samples_per_dataset"):
        if getattr(args, field) <= 0:
            raise ValueError(f"{field.replace('_', '-')} must be positive")
    if not 1 <= args.proposal_epochs < args.epochs:
        raise ValueError("proposal-epochs must be at least one and less than total epochs")
    if not 0 <= args.joint_geometry_calibration_epochs <= args.epochs - args.proposal_epochs:
        raise ValueError("joint-geometry-calibration-epochs must fit within the Joint stage")
    for field in ("depth_variants", "domains"):
        selected = getattr(args, field)
        if len(set(selected)) != len(selected):
            raise ValueError(f"{field.replace('_', '-')} cannot contain duplicates")


def build_plan(args, *, root: Path | None = None) -> dict:
    """Pure plan construction; deliberately does not inspect data or weights."""
    validate_args(args)
    root = (ROOT if root is None else root).resolve()
    checkpoint = args.warm_start_checkpoint.resolve()
    data_root = (args.data_root or root / "data").resolve()
    common = [
        args.bash, str(root / "scripts/run_v16_1070_overnight_linux.sh"),
        "--python", args.python, "--anchored-selection",
        "--warm-start-checkpoint", str(checkpoint), "--data-root", str(data_root),
        "--device", args.device, "--epochs", str(args.epochs),
        "--proposal-epochs", str(args.proposal_epochs),
        "--joint-geometry-calibration-epochs", str(args.joint_geometry_calibration_epochs),
        "--train-size", str(args.train_size), "--val-size", str(args.val_size),
        "--batch-size", str(args.batch_size), "--mse-tolerance", "5e-5",
        "--training-source", "synthetic", "--benchmark-profile", args.benchmark_profile,
        "--real-samples-per-dataset", str(args.real_samples_per_dataset),
        "--visual-samples-per-dataset", str(args.visual_samples_per_dataset),
    ]
    if args.prepare_real_data:
        common += ["--prepare-real-data"]
    runs = []

    def add_run(name, route, *, cap, source_max, domain, shape, deep, peak,
                shape_fraction=None, simple_fraction=None):
        command = common + [
            "--run-name", name, "--candidate-knots", str(cap),
            "--source-max-knots", str(source_max), "--validation-source", domain,
            "--synthetic-shape-domain", shape,
            "--candidate-refinement-layers", "2" if deep else "0",
            "--selection-refinement-layers", "2" if deep else "0",
            "--decoder-refinement-layers", "2" if deep else "0",
        ]
        if peak:
            command += PEAK_ARGS
        else:
            command += ["--max-point-error-weight", "0"]
        if cap < 32:
            command += ["--resize-candidate-warm-start"]
        if shape_fraction is not None:
            command += ["--synthetic-shape-fraction", str(shape_fraction),
                        "--synthetic-simple-fraction", str(simple_fraction)]
        runs.append({
            "run_name": name, "route": route, "candidate_internal_knots": cap,
            "source_internal_knots": [4, source_max], "validation_source": domain,
            "synthetic_shape_domain": shape, "extra_depth_per_module": 2 if deep else 0,
            "point_error_loss_enabled": peak, "command": command,
        })

    if args.route in ("depth", "all"):
        for variant in args.depth_variants:
            add_run(f"{args.run_prefix}_depth_{variant}", "depth", cap=32, source_max=24,
                    domain="all", shape="mixed", deep=variant in ("deep", "deep_peak"),
                    peak=variant in ("peak", "deep_peak"))
    if args.route in ("specialists", "all"):
        for domain in args.domains:
            shape, smaller, slug = DOMAIN_SETTINGS[domain]
            for capacity in (smaller, 32):
                # Both members of a domain pair see the SAME source-K support.
                add_run(f"{args.run_prefix}_specialist_{slug}_k{capacity}", "specialists",
                        cap=capacity, source_max=smaller, domain=domain, shape=shape,
                        deep=False, peak=True, shape_fraction=.5, simple_fraction=.25)
    return {
        "schema_version": 1, "run_prefix": args.run_prefix, "root": str(root),
        "warm_start_checkpoint": str(checkpoint), "data_root": str(data_root),
        "seed": 42, "mse_tolerance": 5e-5,
        "point_error_squared_target_when_enabled": 5e-4,
        "training_data": "synthetic only; synthetic morphological specialists are not proven dataset-optimal models",
        "evaluation": {
            "synthetic_source_k": [4, 24],
            "sources": ["Synthetic", *DOMAINS],
            "methods": ["Ours", "Park", "Liang", "Dung", "Kang", "Luo"],
            "capacity_policy": "all six methods share the current model's internal-knot cap",
            "pass_criterion": "MSE only; peak squared error is reported separately",
        },
        "run_count": len(runs), "runs": runs,
    }


def validate_initializer(path: Path) -> None:
    """Lightweight capacity check; the launcher still performs full preflight."""
    if not path.is_file():
        raise ValueError(f"warm-start checkpoint does not exist: {path}")
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("model_config", {})
    if config.get("max_internal_knots") != 32:
        raise ValueError("granularity experiments require a common K32 warm-start checkpoint")
    for key in ("proposal_refinement_layers", "selection_refinement_layers", "survivor_refinement_layers"):
        if config.get(key, 0) != 0:
            raise ValueError("use a shallow (zero added refinement layers) common initializer for controlled ablations")


def check_artifact_conflicts(plan: dict) -> Path:
    root = Path(plan["root"])
    manifest = root / "outputs/experiment_plans" / (plan["run_prefix"] + ".json")
    targets = [manifest]
    for run in plan["runs"]:
        name = run["run_name"]
        targets.extend(root / "outputs/checkpoints" / (name + suffix)
                       for suffix in (".pt", ".last.pt", ".proposal.pt", ".history.json"))
        targets.extend(root / "outputs" / folder / name
                       for folder in ("logs", "comparisons", "figures"))
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise ValueError("refusing to overwrite existing experiment artifacts; choose a fresh --run-prefix:\n"
                         + "\n".join(existing))
    return manifest


def main(argv=None) -> int:
    args_parser = parser()
    args = args_parser.parse_args(argv)
    try:
        plan = build_plan(args)
    except ValueError as error:
        args_parser.error(str(error))
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    print(f"\nPlanned pipelines: {plan['run_count']}; serial training, six-method tests, and case figures.", flush=True)
    print("No overnight runtime guarantee. MSE pass remains 5e-5; optional peak squared target is 5e-4.", flush=True)
    for index, run in enumerate(plan["runs"], 1):
        print(f"[{index}/{plan['run_count']}] {shlex.join(run['command'])}", flush=True)
    if args.dry_run:
        print("DRY RUN: no weights/data read, no files written, no commands launched.", flush=True)
        return 0
    try:
        bash = shutil.which(args.bash)
        if bash is None:
            raise ValueError(f"Bash executable not found: {args.bash}")
        launcher = Path(plan["root"]) / "scripts/run_v16_1070_overnight_linux.sh"
        if not launcher.is_file():
            raise ValueError(f"Bash pipeline does not exist: {launcher}")
        validate_initializer(Path(plan["warm_start_checkpoint"]))
        manifest = check_artifact_conflicts(plan)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation is a second race-safe safeguard. A plan is not a
        # claim that its experiments have completed; each pipeline owns logs.
        recorded = {**plan, "created_utc": datetime.now(timezone.utc).isoformat(), "status": "planned"}
        with manifest.open("x", encoding="utf-8") as output:
            json.dump(recorded, output, ensure_ascii=False, indent=2)
            output.write("\n")
        for index, run in enumerate(plan["runs"], 1):
            print(f"\nRUN {index}/{plan['run_count']}: {run['run_name']}", flush=True)
            command = [bash, *run["command"][1:]]
            completed = subprocess.run(command, cwd=plan["root"], shell=False, check=False)
            if completed.returncode:
                print(f"STOP: {run['run_name']} failed (exit {completed.returncode}); later models were not started. "
                      f"Inspect outputs/logs/{run['run_name']}. Plan: {manifest}", file=sys.stderr)
                return completed.returncode if completed.returncode > 0 else 1
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("STOP: interrupted; later models were not started. Preserve the per-run checkpoints/logs.", file=sys.stderr)
        return 130
    print(f"Completed all {plan['run_count']} pipelines. Plan: {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
