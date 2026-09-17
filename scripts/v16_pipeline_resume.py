"""Conservative, non-destructive stage recovery for the native v16 runner.

Training uses the trainer's parser/contracts. Benchmark recovery retains its
experiment fingerprint. Derived outputs use hashed receipts and fresh attempt
directories, never an unchecked --overwrite of an interrupted figure run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def training_status(arguments: list[str]) -> str:
    """Only bypass the trainer after checking a completed, compatible run.

    Unfinished runs still use the trainer's full legacy-migration/resume checks.
    An explicit epoch extension resumes training; a reduced budget is rejected.
    """
    import torch
    import train_v16 as train

    args = train.parser().parse_args(arguments)
    train.validate_args(args)
    if args.resume is None:
        raise ValueError("training-status requires --resume")
    saved = torch.load(args.resume, map_location="cpu", weights_only=True)
    previous = saved["training_config"]
    if args.epochs < int(previous["epochs"]):
        raise ValueError("resume cannot reduce the saved epoch budget")
    if int(saved["epoch"]) < args.epochs:
        return "resume"
    if int(saved["epoch"]) != args.epochs:
        raise ValueError("checkpoint epoch disagrees with the requested epoch budget")

    current = train.serial_args(args)
    ignored = {
        "resume", "device", "num_workers", "torch_num_threads",
        "feasible_teacher_batch_size", "log_every_batches",
        "init_checkpoint", "init_full_checkpoint", "initial_keep_fraction",
    }
    # Old runs may predate these explicit scope/default metadata fields. Other
    # missing fields are not silently accepted when skipping the native trainer.
    previous = dict(previous)
    previous.setdefault("study_scope", "legacy")
    changed = [key for key, value in current.items()
               if key not in ignored and previous.get(key) != value]
    if changed:
        raise ValueError(f"completed-run training configuration mismatch: {changed}")
    if saved.get("synthetic_data_contract") != train.synthetic_data_contract(args):
        raise ValueError("completed run has a different synthetic source-range contract")
    expected_objective = {
        "synthetic_ground_truth": train.V16_SUPERVISED_SUBSET_OBJECTIVE_VERSION,
        "offline_feasible_teacher": train.V16_FEASIBLE_TEACHER_OBJECTIVE_VERSION,
        "online_teacher": train.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
    }[args.joint_supervision]
    expected_simplification = (train.V16_FEASIBLE_TEACHER_CONTRACT
        if args.joint_supervision == "offline_feasible_teacher"
        else train.V16_SIMPLIFICATION_CONTRACT)
    expected_architecture = (
        "v16_count_structure_coupled_mass_topk_pilot_v1"
        if args.one_shot_selection_policy == "mass_topk" and args.count_structure_coupling
        else train.V16_ADAPTIVE_SELECTION_REVISION
        if args.one_shot_selection_policy == "mass_topk"
        else "v16_adaptive_beta_threshold_ablation"
    )
    for key, expected in (
        ("objective_version", expected_objective),
        ("simplification_contract", expected_simplification),
        ("architecture_revision", expected_architecture),
    ):
        if saved.get(key) != expected:
            raise ValueError(f"completed-run {key} mismatch; cannot skip training")
    migration = train.resume_loss_semantics_migration(saved, joint_supervision=args.joint_supervision)
    if migration and saved.get("loss_semantics_revision") != train.V16_OFFLINE_LOSS_SEMANTICS_REVISION:
        raise ValueError("completed run requires a loss-semantics migration; extend training explicitly")
    _, provenance = train.load_real_sources(
        args.real_manifest, num_points=args.num_points, point_dim=args.point_dim,
    )
    if saved.get("real_data_provenance") != provenance:
        raise ValueError("completed-run manifest fingerprints changed")
    output = args.output.resolve()
    proposal = output.with_name(output.stem + ".proposal.pt")
    if saved.get("stage") != "joint":
        raise ValueError("a completed two-stage run must have a Joint last checkpoint")
    schedule_fields = {"one_shot_safety_sigma", "one_shot_safety_knots"}
    for path in (output, proposal):
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        expected_stage = "joint" if path == output else "proposal"
        if artifact.get("stage") != expected_stage:
            raise ValueError(f"expected {expected_stage} checkpoint, got a different stage: {path}")
        train.saved_checkpoint_rank(
            path, rank_key="best_joint_rank" if path == output else "best_proposal_rank",
            objective_version=expected_objective, architecture_revision=expected_architecture,
            simplification_contract=expected_simplification, expected_output=output,
        )
        if Path(artifact["training_config"]["output"]).resolve() != output:
            raise ValueError(f"checkpoint belongs to a different run: {path}")
        artifact_model = {key: value for key, value in artifact.get("model_config", {}).items()
                          if key not in schedule_fields}
        saved_model = {key: value for key, value in saved.get("model_config", {}).items()
                       if key not in schedule_fields}
        if artifact_model != saved_model:
            raise ValueError(f"checkpoint model configuration mismatch: {path}")
        if artifact.get("objective_version") != expected_objective:
            raise ValueError(f"checkpoint objective mismatch: {path}")
        train.build_model_from_checkpoint(artifact)
    return "complete"


def command_option(command: list[str], option: str) -> str:
    try:
        return command[command.index(option) + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"stage command requires {option}") from error


def sample_hashes(cases: list[dict]) -> list[str]:
    return [hashlib.sha256(case["points"].numpy().tobytes() + (
        case["reference"].numpy().tobytes() if case["reference"] is not None else b""
    )).hexdigest() for case in cases]


def benchmark_context(command: list[str]) -> tuple[dict, set[tuple]]:
    """Rebuild the native benchmark's identity without running any solver."""
    import torch
    import benchmark_v15_datasets as benchmark

    args = benchmark.parser().parse_args(command[2:])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    benchmark.resolve_comparison_capacities(args, checkpoint)
    torch.set_num_threads(args.torch_num_threads)
    torch.manual_seed(args.selection_seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    cases, provenance = benchmark.prepare_cases(args, checkpoint, checkpoint["model_config"])
    methods = benchmark.PUBLISHED_METHODS if args.method_set == "published" else benchmark.METHODS
    if args.include_verified_ours:
        methods = (methods[0], "ours_verified", *methods[1:])
    identity = {
        "checkpoint_sha256": digest(args.checkpoint),
        "configuration": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()
                          if key not in {"resume", "output_dir"}},
        "datasets": provenance,
        "sample_content_sha256": sample_hashes(cases),
        "hardware": {"device": str(device),
                     "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                     "cpu": platform.processor(), "torch": str(torch.__version__),
                     "threads": torch.get_num_threads(), "python": platform.python_version()},
    }
    return identity, {(case["dataset"], case["sample_id"], method)
                      for case in cases for method in methods}


def benchmark_resume_state(directory: Path, command: list[str]) -> str:
    """Validate existing measurements before choosing complete/partial reuse."""
    metadata = read_json(directory / "experiment.json")
    payload = {key: value for key, value in metadata.items() if key != "fingerprint"}
    if metadata.get("fingerprint") != json_digest(payload):
        raise ValueError("benchmark experiment fingerprint is invalid")
    code = metadata.get("code_sha256", {})
    required = {"scripts/benchmark_v15_datasets.py", "scripts/benchmark_v16_datasets.py"}
    required.update(path.relative_to(ROOT).as_posix() for path in (ROOT / "src").rglob("*.py"))
    normalized_code = {Path(name).as_posix(): value for name, value in code.items()}
    if not required.issubset(normalized_code):
        raise ValueError("benchmark has an incomplete source-code fingerprint")
    for name, expected in normalized_code.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or digest(path) != expected:
            raise ValueError(
                f"benchmark source fingerprint changed: {name}; existing results were not overwritten. "
                "Use the original source revision to resume measurements, or plot the saved "
                "comparison.json separately into a new directory with "
                "scripts/plot_v16_method_comparison.py --allow-unqualified-diagnostic."
            )
    identity, expected = benchmark_context(command)
    changed = [key for key, value in identity.items() if metadata.get(key) != value]
    if changed:
        raise ValueError(f"benchmark resume fingerprint inputs changed: {changed}; use a new output directory")
    report_path = directory / "comparison.json"
    if not report_path.is_file():
        return "partial"
    try:
        report = read_json(report_path)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "partial"  # A torn report can be reconstructed from the native journal.
    rows = report.get("measurements", [])
    identities = [(row["dataset"], row["sample_id"], row["method"]) for row in rows]
    if len(identities) != len(set(identities)) or not set(identities).issubset(expected):
        raise ValueError("comparison contains duplicate or unexpected measurements")
    if report.get("metadata") != metadata:
        raise ValueError("comparison and experiment metadata disagree")
    if set(identities) != expected:
        return "partial"
    import benchmark_v15_datasets as benchmark
    if report.get("summary") != benchmark.summarize(rows):
        raise ValueError("comparison summary disagrees with its measurements")
    journal = directory / "measurements.jsonl"
    try:
        journal_rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return "partial"
    if journal_rows != rows:
        raise ValueError("completed comparison disagrees with its measurement journal")
    if not all((directory / name).is_file() for name in ("report.md", "summary.csv", "measurements.csv")):
        return "partial"
    return "complete"


def run_benchmark(command: list[str], *, resume: bool) -> int:
    directory = Path(command_option(command, "--output-dir"))
    nonempty = directory.exists() and any(directory.iterdir())
    if nonempty and not resume:
        raise ValueError(f"refusing existing benchmark output without --resume-run: {directory}")
    if (directory / "experiment.json").is_file():
        state = benchmark_resume_state(directory, command)
        if state == "complete":
            print(f"Reusing complete, fingerprint-verified benchmark: {directory}", flush=True)
            return 0
        command = [*command, "--resume"]
        print(f"Resuming the fingerprint-verified benchmark journal: {directory}", flush=True)
    elif nonempty:
        raise ValueError(f"benchmark directory is nonempty but has no experiment fingerprint: {directory}")
    return subprocess.call(command)


def derived_signature(command: list[str], kind: str) -> dict:
    """Hash the requested computation, its inputs and its source dependencies."""
    files = {}
    for option in ("--input", "--checkpoint", "--old-checkpoint", "--new-checkpoint"):
        if option in command:
            path = Path(command_option(command, option)).resolve()
            files[str(path)] = digest(path)
    for index, value in enumerate(command[:-1]):
        if value == "--manifest":
            path = Path(command[index + 1].partition("=")[2]).resolve()
            files[str(path)] = digest(path)
    dependencies = sorted(set((ROOT / "scripts").glob("*.py")) | set((ROOT / "src").rglob("*.py")))
    signature = {"command": command, "inputs": files,
                 "code_sha256": {str(path.relative_to(ROOT)): digest(path) for path in dependencies}}
    if kind == "visualize":
        import torch
        import benchmark_v15_datasets as benchmark
        import visualize_v16_real_deployments as visualize
        args = visualize.parser().parse_args(command[2:])
        args.skip_synthetic, args.skip_real = True, False
        args.seed, args.scan_size = 0, 1
        args.min_knot_count = args.max_knot_count = 0
        args.samples_per_knot_count = 1
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        cases, provenance = benchmark.prepare_cases(args, checkpoint, checkpoint["model_config"])
        signature.update(datasets=provenance, sample_content_sha256=sample_hashes(cases))
    elif kind == "paired":
        import torch
        import quick_compare_v16_checkpoints as paired
        args = paired.parser().parse_args(command[2:])
        checkpoint = torch.load(args.old_checkpoint, map_location="cpu", weights_only=True)
        config = paired._dataset_config_from_checkpoint(checkpoint, checkpoint["model_config"])
        cases = list(paired._real_cases(args, num_points=config["num_points"], point_dim=config["point_dim"]))
        signature["real_cases"] = [
            {"dataset": case["dataset"], "case_id": case["case_id"],
             "points_sha256": hashlib.sha256(case["points"].numpy().tobytes()).hexdigest()}
            for case in cases
        ]
    if kind in {"visualize", "paired"}:
        import torch
        requested_device = command_option(command, "--device")
        device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                              if requested_device == "auto" else requested_device)
        signature["hardware"] = {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "platform": platform.platform(), "processor": platform.processor(),
            "torch": str(torch.__version__), "python": platform.python_version(),
        }
    return signature


def save_manifest(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def checked_artifacts(directory: Path, kind: str) -> dict[str, str]:
    if kind == "plot":
        paths = sorted(directory.glob("v16_published_methods_*.png"))
        if not paths:
            raise ValueError("plot stage did not produce a metric PNG")
    elif kind == "visualize":
        report_path = directory / "deployment_visualizations.json"
        report = read_json(report_path)
        records = report.get("records", [])
        expected = sum(row["selected_count"] for row in report.get("dataset_provenance", []))
        if not records or len(records) != expected:
            raise ValueError("visualization stage did not finish all selected cases")
        paths = [report_path, *(Path(row["image"]) for row in records)]
        if report.get("overview_image"):
            paths.append(Path(report["overview_image"]))
    else:
        paths = [path for path in directory.glob("*.json") if not path.name.startswith(".")]
        if not paths:
            raise ValueError("paired stage did not produce its report")
    result = {}
    for path in paths:
        path = path.resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError(f"stage output is missing or outside its attempt directory: {path}")
        result[str(path.relative_to(directory.resolve()))] = digest(path)
    return result


def run_derived(command: list[str], *, kind: str, resume: bool) -> int:
    output_option = "--output-json" if kind == "paired" else "--output-dir"
    requested = Path(command_option(command, output_option)).resolve()
    directory = requested.parent if kind == "paired" else requested
    nonempty = directory.exists() and any(directory.iterdir())
    if nonempty and not resume:
        raise ValueError(f"refusing existing stage output without --resume-run: {directory}")
    receipt_path = directory / ".pipeline-stage.json"
    receipt = read_json(receipt_path) if receipt_path.exists() else {"version": 1, "attempts": []}
    if receipt.get("version") != 1 or not isinstance(receipt.get("attempts"), list):
        raise ValueError(f"invalid stage receipt: {receipt_path}")
    signature = derived_signature(command, kind)
    for attempt in reversed(receipt["attempts"]):
        if attempt.get("signature") != signature or attempt.get("state") != "complete":
            continue
        target = Path(attempt["directory"]).resolve()
        if not target.is_relative_to(directory):
            raise ValueError("stage receipt points outside its output directory")
        artifacts = checked_artifacts(target, kind)
        if artifacts == attempt.get("artifacts"):
            print(f"Reusing verified {kind} outputs: {target}", flush=True)
            return 0
        raise ValueError(f"completed {kind} outputs changed; preserved without overwriting: {target}")
    directory.mkdir(parents=True, exist_ok=True)
    # A previous process may have stopped after writing any individual PNG.
    # Keep every prior artifact; retries get a new, recorded destination.
    target = directory / ("attempt_" + uuid.uuid4().hex[:12]) if nonempty else directory
    target.mkdir(parents=True, exist_ok=True)
    attempt = {"signature": signature, "directory": str(target), "state": "running"}
    receipt["attempts"].append(attempt)
    save_manifest(receipt_path, receipt)
    command = list(command)
    command[command.index(output_option) + 1] = str(target / requested.name if kind == "paired" else target)
    print(f"Running {kind} stage in: {target}", flush=True)
    status = subprocess.call(command)
    if status:
        attempt["state"], attempt["exit_code"] = "interrupted", status
        save_manifest(receipt_path, receipt)
        return status
    attempt["artifacts"] = checked_artifacts(target, kind)
    attempt["state"] = "complete"
    save_manifest(receipt_path, receipt)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("training-status", "benchmark", "plot", "visualize", "paired"))
    parser.add_argument("--resume-run", action="store_true")
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        separator = arguments.index("--")
    except ValueError:
        parser.error("separate the stage command with --")
    args = parser.parse_args(arguments[:separator])
    command = arguments[separator + 1:]
    try:
        if args.action == "training-status":
            print(training_status(command))
            return 0
        if args.action == "benchmark":
            return run_benchmark(command, resume=args.resume_run)
        return run_derived(command, kind=args.action, resume=args.resume_run)
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.exit(2, f"pipeline recovery error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
