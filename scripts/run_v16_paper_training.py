"""Two independent capacity models, coupled t/U training and auditable paper exports.

This is a training protocol, NOT a claim of achieved accuracy or native-paper
reproduction. Native baselines without a verified implementation are reported
unavailable, never replaced by the historical repaired adaptations.
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

from run_v16_granularity_experiments import check_artifact_conflicts

ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-prefix", required=True)
    p.add_argument("--m16-checkpoint", type=Path,
                   default=Path("outputs/checkpoints/universal_m16_m32_3090_r1_m16.pt"))
    p.add_argument("--m32-checkpoint", type=Path,
                   default=Path("outputs/checkpoints/universal_m16_m32_3090_r1_m32.pt"))
    p.add_argument("--capacities", nargs="+", type=int, choices=(16, 32), default=[16, 32])
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--proposal-epochs", type=int, default=12)
    p.add_argument("--joint-geometry-calibration-epochs", type=int, default=8)
    p.add_argument("--train-size", type=int, default=3000)
    p.add_argument("--val-size", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--parameter-chord-blend", type=float, default=0.,
                   help="Optional fixed-blend ablation; default uses chord as learned-update evidence only")
    p.add_argument("--coupled-proposal-steps", type=int, default=2)
    p.add_argument("--coupled-subset-steps", type=int, default=2)
    p.add_argument("--baseline-protocol", choices=("native", "adaptation"), default="native")
    p.add_argument("--real-samples-per-dataset", type=int, default=100)
    p.add_argument("--visual-samples-per-dataset", type=int, default=12)
    p.add_argument("--all-real-test-samples", action="store_true")
    p.add_argument("--prepare-real-data", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--bash", default="bash")
    p.add_argument("--python", default=sys.executable)
    return p


def build_plan(args, *, root=None):
    root = (ROOT if root is None else Path(root)).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_prefix) or ".." in args.run_prefix:
        raise ValueError("run-prefix must be a fresh plain artifact name")
    if len(set(args.capacities)) != len(args.capacities):
        raise ValueError("capacities must not repeat")
    if not 0 < args.proposal_epochs < args.epochs:
        raise ValueError("epochs is total; require 0 < proposal-epochs < epochs")
    if not 0 <= args.joint_geometry_calibration_epochs <= args.epochs - args.proposal_epochs:
        raise ValueError("calibration must fit inside Joint")
    if not 0 <= args.parameter_chord_blend <= 1:
        raise ValueError("parameter-chord-blend must lie in [0,1]")
    if min(args.coupled_proposal_steps, args.coupled_subset_steps, args.seed) < 0:
        raise ValueError("steps and seed must be nonnegative")
    if min(args.train_size, args.val_size, args.batch_size,
           args.real_samples_per_dataset, args.visual_samples_per_dataset) < 1:
        raise ValueError("dataset sizes and batch size must be positive")
    runs = []
    for capacity in args.capacities:
        checkpoint = getattr(args, f"m{capacity}_checkpoint").resolve()
        command = [args.bash, str(root / "scripts/run_v16_1070_overnight_linux.sh"),
                   "--python", args.python, "--anchored-selection", "--paper-output",
                   "--native-baselines", "--baseline-protocol", args.baseline_protocol,
                   "--warm-start-checkpoint", str(checkpoint),
                   "--run-name", f"{args.run_prefix}_m{capacity}",
                   "--data-root", str(args.data_root.resolve()), "--device", args.device,
                   "--candidate-knots", str(capacity), "--source-max-knots", str(min(capacity, 24)),
                   "--epochs", str(args.epochs), "--proposal-epochs", str(args.proposal_epochs),
                   "--joint-geometry-calibration-epochs", str(args.joint_geometry_calibration_epochs),
                   "--train-size", str(args.train_size), "--val-size", str(args.val_size),
                   "--real-val-size", "100", "--batch-size", str(args.batch_size),
                   "--seed", str(args.seed), "--mse-tolerance", "5e-5",
                   "--joint-lr", "1e-5", "--joint-proposal-lr-scale", "0.1",
                   "--joint-decoder-lr-scale", "0.25",
                   "--candidate-refinement-layers", "2", "--selection-refinement-layers", "2",
                   "--decoder-refinement-layers", "2",
                   "--coupled-proposal-steps", str(args.coupled_proposal_steps),
                   "--coupled-subset-steps", str(args.coupled_subset_steps),
                   "--parameter-chord-blend", str(args.parameter_chord_blend),
                   "--max-point-error-weight", "0.05", "--max-point-error-tolerance", "5e-4",
                   "--benchmark-profile", "full", "--synthetic-samples-per-k", "10",
                   "--real-samples-per-dataset", str(args.real_samples_per_dataset),
                   "--visual-samples-per-dataset", str(args.visual_samples_per_dataset)]
        if args.prepare_real_data:
            command += ["--prepare-real-data"]
        if args.all_real_test_samples:
            command += ["--all-real-test-samples"]
        runs.append(dict(run_name=f"{args.run_prefix}_m{capacity}", capacity=capacity,
                         initializer=str(checkpoint), command=command))
    return dict(schema_version=1, root=str(root), run_prefix=args.run_prefix,
                run_count=len(runs), runs=runs, baseline_protocol=args.baseline_protocol,
                native_fidelity="unverified methods are unavailable, never silently adapted",
                training="mixed synthetic only; external validation/test splits disjoint",
                test_scope="all sources; all test records" if args.all_real_test_samples else
                           f"all sources; up to {args.real_samples_per_dataset} held-out curves/source",
                source_ranges="M16 K4..16; M32 K4..24; not a pure capacity-only ablation",
                deployment="one discrete mask; fixed-depth numerical t/U updates; one final refit",
                chord_reference=("every coupled step reads chord and t-minus-chord; no fixed initial blend"
                                 if args.parameter_chord_blend == 0 else
                                 f"explicit initial chord blend {args.parameter_chord_blend}; subsequent updates remain learned"),
                error_units="normalized squared Euclidean: MSE 5e-5; peak target 5e-4")


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        plan = build_plan(args)
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        if args.dry_run:
            print("DRY RUN: no weights/data read, no files written or training launched.")
            return 0
        bash = shutil.which(args.bash)
        if bash is None:
            raise ValueError(f"Bash executable not found: {args.bash}")
        import torch
        for run in plan["runs"]:
            path = Path(run["initializer"])
            if not path.is_file():
                raise ValueError(f"initializer does not exist: {path}")
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if checkpoint.get("model_config", {}).get("max_internal_knots") != run["capacity"]:
                raise ValueError(f"M{run['capacity']} requires its own same-capacity initializer: {path}")
        manifest = check_artifact_conflicts(plan)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("x", encoding="utf-8") as handle:
            json.dump({**plan, "created_utc": datetime.now(timezone.utc).isoformat()}, handle,
                      ensure_ascii=False, indent=2)
        for run in plan["runs"]:
            command = [bash, *run["command"][1:]]
            print(shlex.join(command), flush=True)
            result = subprocess.run(command, cwd=ROOT, shell=False, check=False)
            if result.returncode:
                return max(1, result.returncode)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
