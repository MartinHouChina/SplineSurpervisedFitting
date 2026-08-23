from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.checkpointing import CANDIDATE_PRUNING_OBJECTIVE_VERSION
from spline_fitting.data.synthetic import SyntheticCubicBSplineDataset
from spline_fitting.losses import CandidatePruningLoss, CandidatePruningLossWeights
from spline_fitting.models import SplineFittingNetwork
from spline_fitting.training.trainer import Trainer


def _checkpoint_with_metadata(
    checkpoint: dict[str, object],
    *,
    model_config: dict[str, object],
    dataset_config: dict[str, object],
    loss_config: dict[str, object],
    training_config: dict[str, object],
    deployment_config: dict[str, object],
    histories: dict[str, list[dict[str, float]]],
) -> dict[str, object]:
    checkpoint.update(
        {
            "model_config": model_config,
            "dataset_config": dataset_config,
            "dataset_type": "synthetic_open_cubic_bspline",
            "objective_version": CANDIDATE_PRUNING_OBJECTIVE_VERSION,
            "loss_config": loss_config,
            "training_config": training_config,
            "deployment_config": deployment_config,
            "selected_stage": "joint_candidate_pruning",
            "stage_histories": histories,
        }
    )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train high-recall knot proposals plus interactive pruning for the "
            "minimum knot count reachable under a hard geometric RMS threshold."
        )
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--candidate-pretrain-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--log-every-batches", type=int, default=20)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--val-seed", type=int, default=10000)
    parser.add_argument("--num-points", type=int, default=192)
    parser.add_argument("--point-dim", type=int, choices=(2, 3), default=2)
    parser.add_argument("--min-control-points", type=int, default=8)
    parser.add_argument("--max-control-points", type=int, default=24)
    parser.add_argument("--candidate-knots", type=int, default=28)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--noise-std", type=float, default=0.001)
    parser.add_argument("--knot-nonuniformity", type=float, default=0.65)
    parser.add_argument("--sampling-nonuniformity", type=float, default=0.45)
    parser.add_argument("--turn-strength", type=float, default=0.45)
    parser.add_argument(
        "--fit-tolerance",
        type=float,
        default=5e-3,
        help=(
            "Normalized mean Euclidean RMS threshold. It defines canonical "
            "training labels and the hard deployment stopping rule."
        ),
    )
    parser.add_argument("--candidate-match-tolerance", type=float, default=0.02)
    parser.add_argument("--min-knot-gap", type=float, default=1e-3)
    parser.add_argument("--pruning-residual-bandwidth", type=float, default=0.05)
    parser.add_argument("--lambda-poly", type=float, default=1e-6)
    parser.add_argument("--lambda-knot", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-fit", type=float, default=0.25)
    parser.add_argument("--lambda-threshold-violation", type=float, default=5.0)
    parser.add_argument("--lambda-true-params", type=float, default=0.05)
    parser.add_argument("--lambda-candidate-coverage", type=float, default=5.0)
    parser.add_argument("--lambda-candidate-repulsion", type=float, default=0.05)
    parser.add_argument("--lambda-keep", type=float, default=0.25)
    parser.add_argument("--lambda-remove-action", type=float, default=1.0)
    parser.add_argument("--lambda-knot-position", type=float, default=2.0)
    parser.add_argument("--lambda-deletion-cost", type=float, default=0.05)
    parser.add_argument("--positive-keep-weight", type=float, default=2.0)
    parser.add_argument("--knot-position-beta", type=float, default=0.01)
    parser.add_argument(
        "--exact-deletion-supervision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use batched standard-B-spline single-deletion RMS as the "
            "remove/STOP teacher during joint training."
        ),
    )
    parser.add_argument(
        "--resample-train-each-epoch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Regenerate curves each epoch. Off by default because threshold "
            "canonicalization at K=20 is deliberately expensive."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "candidate_pruning_v7.pt",
    )
    args = parser.parse_args()

    if args.epochs < 2:
        parser.error("--epochs must be at least 2")
    if not 0 <= args.candidate_pretrain_epochs < args.epochs:
        parser.error("--candidate-pretrain-epochs must lie in [0, epochs)")
    if args.train_size <= 0 or args.val_size <= 0 or args.batch_size <= 0:
        parser.error("dataset and batch sizes must be positive")
    if args.log_every_batches < 0:
        parser.error("--log-every-batches must be non-negative")
    if args.fit_tolerance <= 0.0:
        parser.error("--fit-tolerance must be positive")
    if args.candidate_match_tolerance <= 0.0:
        parser.error("--candidate-match-tolerance must be positive")
    if args.attention_heads <= 0 or args.hidden_dim % args.attention_heads:
        parser.error("--attention-heads must divide --hidden-dim")
    max_source_knots = args.max_control_points - 4
    if args.candidate_knots < max_source_knots:
        parser.error(
            "--candidate-knots must cover the maximum source knot count "
            f"({max_source_knots})"
        )
    if args.min_knot_gap * (args.candidate_knots + 1) >= 1.0:
        parser.error("--min-knot-gap leaves no free parameter interval")
    nonnegative = (
        args.lambda_poly,
        args.lambda_knot,
        args.weight_decay,
        args.lambda_fit,
        args.lambda_threshold_violation,
        args.lambda_true_params,
        args.lambda_candidate_coverage,
        args.lambda_candidate_repulsion,
        args.lambda_keep,
        args.lambda_remove_action,
        args.lambda_knot_position,
        args.lambda_deletion_cost,
    )
    if any(value < 0.0 for value in nonnegative):
        parser.error("regularization and loss weights must be non-negative")

    torch.manual_seed(args.train_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_config: dict[str, object] = {
        "num_points": args.num_points,
        "point_dim": args.point_dim,
        "min_control_points": args.min_control_points,
        "max_control_points": args.max_control_points,
        "noise_std": args.noise_std,
        "knot_nonuniformity": args.knot_nonuniformity,
        "sampling_nonuniformity": args.sampling_nonuniformity,
        "turn_strength": args.turn_strength,
        # The label and deployment constraint intentionally share one value.
        "canonical_knot_tolerance": args.fit_tolerance,
        "normalize": True,
        "return_ground_truth": True,
    }
    train_set = SyntheticCubicBSplineDataset(
        size=args.train_size,
        seed=args.train_seed,
        resample_each_epoch=args.resample_train_each_epoch,
        **dataset_config,
    )
    val_set = SyntheticCubicBSplineDataset(
        size=args.val_size,
        seed=args.val_seed,
        resample_each_epoch=False,
        **dataset_config,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    model_config: dict[str, object] = {
        "point_dim": args.point_dim,
        "degree": 3,
        "hidden_dim": args.hidden_dim,
        "encoder_layers": args.encoder_layers,
        "max_internal_knots": args.candidate_knots,
        "min_parameter_gap": 1e-4,
        "min_knot_gap": args.min_knot_gap,
        "gap_parameterization": "strict",
        "lambda_poly": args.lambda_poly,
        "lambda_knot": args.lambda_knot,
        "structure_mode": "candidate_pruning",
        "structure_attention_heads": args.attention_heads,
        "geometry_feature_mode": "chord_derivatives",
        "pruning_residual_bandwidth": args.pruning_residual_bandwidth,
        "pruning_initial_keep_probability": 0.9,
        "compute_first_derivative": False,
    }
    model = SplineFittingNetwork(**model_config)
    joint_weights = CandidatePruningLossWeights(
        fit=args.lambda_fit,
        threshold_violation=args.lambda_threshold_violation,
        true_parameter=args.lambda_true_params,
        candidate_coverage=args.lambda_candidate_coverage,
        candidate_repulsion=args.lambda_candidate_repulsion,
        keep=args.lambda_keep,
        remove_action=args.lambda_remove_action,
        knot_position=args.lambda_knot_position,
        count_consistency=0.0,
        deletion_cost=args.lambda_deletion_cost,
    )
    pretrain_weights = replace(
        joint_weights,
        keep=0.0,
        remove_action=0.0,
        count_consistency=0.0,
        deletion_cost=0.0,
    )

    def make_loss(
        weights: CandidatePruningLossWeights,
        *,
        exact_deletion_supervision: bool,
    ) -> CandidatePruningLoss:
        return CandidatePruningLoss(
            weights,
            knot_position_beta=args.knot_position_beta,
            candidate_match_tolerance=args.candidate_match_tolerance,
            fit_tolerance=args.fit_tolerance,
            positive_keep_weight=args.positive_keep_weight,
            exact_deletion_supervision=exact_deletion_supervision,
            deletion_smoothness_weight=1e-6,
            deletion_control_ridge=0.0,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    trainer = Trainer(
        model,
        make_loss(pretrain_weights, exact_deletion_supervision=False),
        optimizer,
        device,
        knot_match_tolerance=args.candidate_match_tolerance,
        log_every_batches=args.log_every_batches,
    )

    print(
        "v7 objective: minimize retained knots subject to normalized Euclidean "
        f"RMS <= {args.fit_tolerance:g}",
        flush=True,
    )
    print(
        "Data: "
        f"train={args.train_size}, val={args.val_size}, points={args.num_points}, "
        f"source K={args.min_control_points - 4}..{max_source_knots}, "
        f"candidate Kc={args.candidate_knots}",
        flush=True,
    )
    print(
        "Labels use standard-curve greedy canonical reduction at the same RMS "
        "threshold. Deployment independently rechecks every accepted deletion.",
        flush=True,
    )
    if args.resample_train_each_epoch:
        print(
            "WARNING: online K<=20 canonicalization will repeat every epoch and "
            "can dominate training time.",
            flush=True,
        )

    histories: dict[str, list[dict[str, float]]] = {}
    if args.candidate_pretrain_epochs:
        histories["candidate_pretrain"] = trainer.fit(
            train_loader,
            val_loader,
            args.candidate_pretrain_epochs,
            checkpoint_path=None,
            stage_name="candidate_pretrain",
        )

    trainer.loss_fn = make_loss(
        joint_weights,
        exact_deletion_supervision=args.exact_deletion_supervision,
    ).to(device)
    joint_epochs = args.epochs - args.candidate_pretrain_epochs
    histories["joint_candidate_pruning"] = trainer.fit(
        train_loader,
        val_loader,
        joint_epochs,
        checkpoint_path=args.output,
        epoch_offset=args.candidate_pretrain_epochs,
        stage_name="joint_candidate_pruning",
    )

    loss_config: dict[str, object] = {
        "weights": asdict(joint_weights),
        "knot_position_beta": args.knot_position_beta,
        "candidate_match_tolerance": args.candidate_match_tolerance,
        "fit_tolerance": args.fit_tolerance,
        "positive_keep_weight": args.positive_keep_weight,
        "exact_deletion_supervision": args.exact_deletion_supervision,
        "deletion_smoothness_weight": 1e-6,
        "deletion_control_ridge": 0.0,
        "count_consistency_is_deployment_rule": False,
        "analytic_deletion_cost_is_auxiliary_only": True,
    }
    training_config: dict[str, object] = {
        "structure_mode": "candidate_pruning",
        "epochs": args.epochs,
        "candidate_pretrain_epochs": args.candidate_pretrain_epochs,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "train_seed": args.train_seed,
        "val_seed": args.val_seed,
        "resample_train_each_epoch": args.resample_train_each_epoch,
        "fit_tolerance": args.fit_tolerance,
        "candidate_match_tolerance": args.candidate_match_tolerance,
        "exact_deletion_supervision": args.exact_deletion_supervision,
        "weight_decay": args.weight_decay,
    }
    deployment_config: dict[str, object] = {
        "selection_rule": "greedy_standard_bspline_hard_rms",
        "error_tolerance": args.fit_tolerance,
        "min_internal_knots": 0,
        "smoothness_weight": 1e-6,
        "control_ridge": 0.0,
        "interpolate_endpoints": True,
        "learned_keep_threshold_is_final": False,
        "global_minimum_guaranteed": False,
    }

    best = torch.load(args.output, map_location="cpu", weights_only=True)
    best = _checkpoint_with_metadata(
        best,
        model_config=model_config,
        dataset_config=dataset_config,
        loss_config=loss_config,
        training_config=training_config,
        deployment_config=deployment_config,
        histories=histories,
    )
    torch.save(best, args.output)

    last_path = args.output.with_name(args.output.stem + "_last" + args.output.suffix)
    last = {
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "stage": "joint_candidate_pruning_last",
        "selection_metric": "last_epoch_not_selected",
        "history": histories["joint_candidate_pruning"],
    }
    last = _checkpoint_with_metadata(
        last,
        model_config=model_config,
        dataset_config=dataset_config,
        loss_config=loss_config,
        training_config=training_config,
        deployment_config=deployment_config,
        histories=histories,
    )
    torch.save(last, last_path)
    print(
        f"Selected epoch {best['epoch']} | candidate recall="
        f"{best.get('best_candidate_recall', float('nan')):.3f}",
        flush=True,
    )
    print(f"Saved best checkpoint: {args.output}", flush=True)
    print(f"Saved last checkpoint: {last_path}", flush=True)


if __name__ == "__main__":
    main()
