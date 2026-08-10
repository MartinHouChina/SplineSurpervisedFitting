from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.checkpointing import CURRENT_OBJECTIVE_VERSION
from spline_fitting.data.synthetic import SyntheticCubicBSplineDataset
from spline_fitting.losses.total_loss import LossWeights, SplineFittingLoss
from spline_fitting.models.spline_network import SplineFittingNetwork
from spline_fitting.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train canonical ordinal knot-count prediction and shared "
            "count-conditioned ordered knot decoding."
        )
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--num-points", type=int, default=64)
    parser.add_argument("--point-dim", type=int, choices=(2, 3), default=2)
    parser.add_argument("--min-control-points", type=int, default=5)
    parser.add_argument("--max-control-points", type=int, default=10)
    parser.add_argument("--noise-std", type=float, default=0.001)
    parser.add_argument("--knot-nonuniformity", type=float, default=0.65)
    parser.add_argument("--sampling-nonuniformity", type=float, default=0.45)
    parser.add_argument("--turn-strength", type=float, default=0.45)
    parser.add_argument(
        "--canonical-knot-tolerance",
        type=float,
        default=5e-3,
        help="Normalized RMS tolerance used by greedy ground-truth knot removal.",
    )
    parser.add_argument("--max-knots", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--count-attention-heads", type=int, default=4)
    parser.add_argument("--count-query-count", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-poly", type=float, default=1e-6)
    parser.add_argument("--lambda-knot", type=float, default=1e-5)
    parser.add_argument("--lambda-true-params", type=float, default=5e-2)
    parser.add_argument("--lambda-parameter-prior", type=float, default=0.0)
    parser.add_argument("--lambda-knot-position", type=float, default=5e-2)
    parser.add_argument(
        "--lambda-count",
        type=float,
        default=5e-3,
        help="Weight for ordinal knot-count supervision.",
    )
    parser.add_argument(
        "--lambda-over-count",
        type=float,
        default=2e-3,
        help="Asymmetric penalty on expected knot over-prediction.",
    )
    parser.add_argument("--knot-position-beta", type=float, default=0.02)
    parser.add_argument("--knot-match-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "count_conditioned_v5.pt",
    )
    args = parser.parse_args()

    if args.epochs <= 0 or args.train_size <= 0 or args.val_size <= 0:
        parser.error("epochs and dataset sizes must be positive")
    if args.max_knots < 0:
        parser.error("--max-knots must be non-negative")
    if args.count_attention_heads <= 0 or args.hidden_dim % args.count_attention_heads:
        parser.error("--count-attention-heads must divide --hidden-dim")
    if args.count_query_count <= 0:
        parser.error("--count-query-count must be positive")
    if args.knot_position_beta <= 0.0:
        parser.error("--knot-position-beta must be positive")
    if args.knot_match_tolerance < 0.0:
        parser.error("--knot-match-tolerance must be non-negative")
    if args.canonical_knot_tolerance < 0.0:
        parser.error("--canonical-knot-tolerance must be non-negative")
    regularizers = (
        args.lambda_poly,
        args.lambda_knot,
        args.lambda_true_params,
        args.lambda_parameter_prior,
        args.lambda_knot_position,
        args.lambda_count,
        args.lambda_over_count,
    )
    if any(value < 0.0 for value in regularizers):
        parser.error("loss and solver regularization weights must be non-negative")

    max_true_internal_knots = args.max_control_points - 4
    if args.max_knots < max_true_internal_knots:
        parser.error(
            "ordinal count prediction requires --max-knots to cover the source dataset; "
            f"need at least {max_true_internal_knots}, got {args.max_knots}"
        )

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_config = {
        "num_points": args.num_points,
        "point_dim": args.point_dim,
        "min_control_points": args.min_control_points,
        "max_control_points": args.max_control_points,
        "noise_std": args.noise_std,
        "knot_nonuniformity": args.knot_nonuniformity,
        "sampling_nonuniformity": args.sampling_nonuniformity,
        "turn_strength": args.turn_strength,
        "canonical_knot_tolerance": args.canonical_knot_tolerance,
        "normalize": True,
        "return_ground_truth": True,
    }
    train_set = SyntheticCubicBSplineDataset(
        size=args.train_size, seed=42, **dataset_config
    )
    val_set = SyntheticCubicBSplineDataset(
        size=args.val_size, seed=10000, **dataset_config
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    model_config = {
        "point_dim": args.point_dim,
        "degree": 3,
        "hidden_dim": args.hidden_dim,
        "encoder_layers": 3,
        "max_internal_knots": args.max_knots,
        "min_parameter_gap": 1e-4,
        "min_knot_gap": 1e-3,
        "gap_parameterization": "strict",
        "lambda_poly": args.lambda_poly,
        "lambda_knot": args.lambda_knot,
        "structure_mode": "count_conditioned",
        "count_attention_heads": args.count_attention_heads,
        "count_head_mode": "ordinal_local_attention",
        "count_query_count": args.count_query_count,
        "count_decoder_mode": "shared_count_embedding",
        "geometry_feature_mode": "chord_derivatives",
        "compute_first_derivative": False,
    }
    model = SplineFittingNetwork(**model_config)
    loss_config = {
        "weights": {
            "fit": 1.0,
            "l0": 0.0,
            "activity": 0.0,
            "binary": 0.0,
            "orthogonal": 0.0,
            "gap": 0.0,
            "parameter_prior": args.lambda_parameter_prior,
            "true_parameter": args.lambda_true_params,
            "existence": 0.0,
            "knot_position": args.lambda_knot_position,
            "count": args.lambda_count,
            "over_count": args.lambda_over_count,
        },
        "min_knot_gap": 1e-3,
        "knot_position_beta": args.knot_position_beta,
        "count_loss": "ordinal_binary_cross_entropy",
        "checkpoint_selection_metric": "knot_match_f1_then_precision_mae_loss",
        "knot_match_tolerance": args.knot_match_tolerance,
    }
    loss_fn = SplineFittingLoss(
        LossWeights(**loss_config["weights"]),
        min_knot_gap=loss_config["min_knot_gap"],
        knot_position_beta=loss_config["knot_position_beta"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    trainer = Trainer(
        model,
        loss_fn,
        optimizer,
        device,
        knot_match_tolerance=args.knot_match_tolerance,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        checkpoint_path=args.output,
        stage_name="count_conditioned",
    )

    checkpoint = torch.load(args.output, map_location="cpu", weights_only=True)
    checkpoint["selected_stage"] = "count_conditioned"
    checkpoint["stage_histories"] = {"count_conditioned": history}
    checkpoint["model_config"] = model_config
    checkpoint["dataset_config"] = dataset_config
    checkpoint["dataset_type"] = "synthetic_open_cubic_bspline"
    checkpoint["objective_version"] = CURRENT_OBJECTIVE_VERSION
    checkpoint["loss_config"] = loss_config
    checkpoint["training_config"] = {
        "structure_mode": "count_conditioned",
        "train_size": args.train_size,
        "val_size": args.val_size,
        "count_attention_heads": args.count_attention_heads,
        "count_query_count": args.count_query_count,
        "canonical_knot_tolerance": args.canonical_knot_tolerance,
        "lambda_true_params": args.lambda_true_params,
        "lambda_parameter_prior": args.lambda_parameter_prior,
        "lambda_knot_position": args.lambda_knot_position,
        "lambda_count": args.lambda_count,
        "lambda_over_count": args.lambda_over_count,
        "knot_position_beta": args.knot_position_beta,
        "knot_match_tolerance": args.knot_match_tolerance,
    }
    torch.save(checkpoint, args.output)
    print(
        f"Selected epoch {checkpoint['epoch']} | "
        f"knot F1={checkpoint.get('best_knot_match_f1', 0.0):.3f} | "
        f"count accuracy={checkpoint.get('best_count_accuracy', 0.0):.3f}"
    )
    print(f"Saved checkpoint to: {args.output}")


if __name__ == "__main__":
    main()
