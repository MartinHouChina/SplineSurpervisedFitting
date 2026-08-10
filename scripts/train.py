from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.data.synthetic import SyntheticCubicBSplineDataset
from spline_fitting.checkpointing import CURRENT_OBJECTIVE_VERSION
from spline_fitting.losses.total_loss import LossWeights, SplineFittingLoss
from spline_fitting.models.spline_network import SplineFittingNetwork
from spline_fitting.training.trainer import Trainer


def ramp(epoch: int, start: int, duration: int) -> float:
    if epoch < start:
        return 0.0
    return min(1.0, (epoch - start + 1) / max(duration, 1))


def linear_anneal(
    epoch: int,
    start_epoch: int,
    duration: int,
    start_value: float,
    end_value: float,
) -> float:
    if epoch < start_epoch:
        return start_value
    fraction = min(1.0, (epoch - start_epoch + 1) / max(duration, 1))
    return start_value + fraction * (end_value - start_value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train independent knot queries with supervised existence/position "
            "targets and Hard-Concrete deployment gates."
        )
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-points", type=int, default=64)
    parser.add_argument("--point-dim", type=int, choices=(2, 3), default=2)
    parser.add_argument("--min-control-points", type=int, default=5)
    parser.add_argument("--max-control-points", type=int, default=10)
    parser.add_argument("--noise-std", type=float, default=0.001)
    parser.add_argument("--knot-nonuniformity", type=float, default=0.65)
    parser.add_argument("--sampling-nonuniformity", type=float, default=0.45)
    parser.add_argument("--turn-strength", type=float, default=0.45)
    parser.add_argument("--max-knots", type=int, default=8)
    parser.add_argument(
        "--activity-threshold",
        type=float,
        default=0.5,
        help="Fixed probability threshold used for validation and deployment gates.",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-l0", type=float, default=0.0)
    parser.add_argument("--lambda-poly", type=float, default=1e-6)
    parser.add_argument("--lambda-knot", type=float, default=1e-5)
    parser.add_argument(
        "--lambda-true-params",
        type=float,
        default=1e-2,
        help="Supervision weight for synthetic ground-truth point parameters.",
    )
    parser.add_argument(
        "--lambda-parameter-prior",
        type=float,
        default=0.0,
        help="Optional chord-length prior; disabled when true-parameter supervision is used.",
    )
    parser.add_argument(
        "--lambda-existence",
        type=float,
        default=5e-3,
        help="Supervised BCE weight for per-query knot existence.",
    )
    parser.add_argument(
        "--lambda-knot-position",
        type=float,
        default=1e-2,
        help="Smooth-L1 weight for matched independent knot positions.",
    )
    parser.add_argument(
        "--lambda-count",
        type=float,
        default=2e-3,
        help="Weight for normalized expected-versus-true knot count.",
    )
    parser.add_argument(
        "--knot-position-beta",
        type=float,
        default=0.02,
        help="Smooth-L1 transition width for matched knot positions.",
    )
    parser.add_argument("--gate-warmup-epochs", type=int, default=0)
    parser.add_argument("--l0-ramp-epochs", type=int, default=0)
    parser.add_argument("--gate-temperature-start", type=float, default=2.0 / 3.0)
    parser.add_argument("--gate-temperature-end", type=float, default=0.5)
    parser.add_argument("--gate-anneal-epochs", type=int, default=40)
    parser.add_argument("--hard-concrete-gamma", type=float, default=-0.1)
    parser.add_argument("--hard-concrete-zeta", type=float, default=1.1)
    parser.add_argument("--activity-initial-bias", type=float, default=-2.0)
    parser.add_argument("--activity-context-bandwidth", type=float, default=0.08)
    parser.add_argument("--knot-attention-heads", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "independent_queries_v2.pt",
    )
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if not 0.0 <= args.activity_threshold <= 1.0:
        parser.error("--activity-threshold must lie in [0, 1]")
    if (
        args.lambda_l0 < 0.0
        or args.lambda_poly < 0.0
        or args.lambda_knot < 0.0
        or args.lambda_true_params < 0.0
        or args.lambda_parameter_prior < 0.0
        or args.lambda_existence < 0.0
        or args.lambda_knot_position < 0.0
        or args.lambda_count < 0.0
    ):
        parser.error("loss and solver regularization weights must be non-negative")
    if args.gate_temperature_start <= 0.0 or args.gate_temperature_end <= 0.0:
        parser.error("Hard-Concrete temperatures must be positive")
    if args.knot_position_beta <= 0.0:
        parser.error("--knot-position-beta must be positive")
    if args.activity_context_bandwidth <= 0.0:
        parser.error("--activity-context-bandwidth must be positive")
    if args.knot_attention_heads <= 0 or args.hidden_dim % args.knot_attention_heads:
        parser.error("--knot-attention-heads must divide --hidden-dim")
    if (
        args.gate_warmup_epochs < 0
        or args.l0_ramp_epochs < 0
        or args.gate_anneal_epochs < 0
    ):
        parser.error("gate schedule lengths must be non-negative")

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
        "normalize": True,
        "return_ground_truth": True,
    }
    train_set = SyntheticCubicBSplineDataset(size=512, seed=42, **dataset_config)
    val_set = SyntheticCubicBSplineDataset(size=128, seed=10000, **dataset_config)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    max_true_internal_knots = args.max_control_points - 4
    if args.max_knots < max_true_internal_knots:
        parser.error(
            "supervised matching requires --max-knots to cover the dataset; "
            f"need at least {max_true_internal_knots}, got {args.max_knots}"
        )

    model_config = {
        "point_dim": args.point_dim,
        "degree": 3,
        "hidden_dim": args.hidden_dim,
        "encoder_layers": 3,
        "max_internal_knots": args.max_knots,
        "min_parameter_gap": 1e-4,
        "min_knot_gap": 1e-3,
        "lambda_poly": args.lambda_poly,
        "lambda_knot": args.lambda_knot,
        "gate_eps": 0.0,
        "gate_mode": "hard_concrete",
        "gate_temperature": args.gate_temperature_start,
        "hard_concrete_gamma": args.hard_concrete_gamma,
        "hard_concrete_zeta": args.hard_concrete_zeta,
        "activity_threshold": args.activity_threshold,
        "activity_initial_bias": args.activity_initial_bias,
        "gap_parameterization": "strict",
        "activity_use_local_context": True,
        "activity_use_query_features": True,
        "activity_context_bandwidth": args.activity_context_bandwidth,
        "activity_use_pilot_importance": False,
        "detach_activity_gate_for_fit": True,
        "knot_use_local_cross_attention": True,
        "knot_attention_heads": args.knot_attention_heads,
        "knot_parameterization": "independent_queries",
        "compute_first_derivative": False,
    }
    model = SplineFittingNetwork(**model_config)
    loss_config = {
        "weights": {
            "fit": 1.0,
            "l0": args.lambda_l0,
            "activity": 0.0,
            "binary": 0.0,
            "orthogonal": 0.0,
            "gap": 0.0,
            "parameter_prior": args.lambda_parameter_prior,
            "true_parameter": args.lambda_true_params,
            "existence": args.lambda_existence,
            "knot_position": args.lambda_knot_position,
            "count": args.lambda_count,
        },
        "min_knot_gap": 1e-3,
        "knot_position_beta": args.knot_position_beta,
        "checkpoint_selection_metric": "existence_f1_then_loss",
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
        activity_threshold=args.activity_threshold,
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        l0_schedule=lambda e: ramp(
            e,
            start=args.gate_warmup_epochs,
            duration=args.l0_ramp_epochs,
        ),
        gate_temperature_schedule=lambda e: linear_anneal(
            e,
            start_epoch=args.gate_warmup_epochs,
            duration=args.gate_anneal_epochs,
            start_value=args.gate_temperature_start,
            end_value=args.gate_temperature_end,
        ),
        gate_warmup_epochs=args.gate_warmup_epochs,
        checkpoint_selection_start_epoch=min(
            max(args.gate_warmup_epochs + args.l0_ramp_epochs - 1, 0),
            max(args.epochs - 1, 0),
        ),
        checkpoint_path=args.output,
    )

    checkpoint = torch.load(args.output, map_location="cpu", weights_only=True)
    checkpoint["model_config"] = model_config
    checkpoint["dataset_config"] = dataset_config
    checkpoint["dataset_type"] = "synthetic_open_cubic_bspline"
    checkpoint["objective_version"] = CURRENT_OBJECTIVE_VERSION
    checkpoint["loss_config"] = loss_config
    checkpoint["training_config"] = {
        "gate_scheme": "supervised_existence_hard_concrete",
        "gate_warmup_epochs": args.gate_warmup_epochs,
        "l0_ramp_epochs": args.l0_ramp_epochs,
        "gate_temperature_start": args.gate_temperature_start,
        "gate_temperature_end": args.gate_temperature_end,
        "gate_anneal_epochs": args.gate_anneal_epochs,
        "activity_threshold": args.activity_threshold,
        "activity_initial_bias": args.activity_initial_bias,
        "activity_context_bandwidth": args.activity_context_bandwidth,
        "activity_use_pilot_importance": False,
        "detach_activity_gate_for_fit": True,
        "activity_use_query_features": True,
        "knot_use_local_cross_attention": True,
        "knot_attention_heads": args.knot_attention_heads,
        "knot_parameterization": "independent_queries",
        "lambda_true_params": args.lambda_true_params,
        "lambda_parameter_prior": args.lambda_parameter_prior,
        "lambda_existence": args.lambda_existence,
        "lambda_knot_position": args.lambda_knot_position,
        "lambda_count": args.lambda_count,
        "knot_position_beta": args.knot_position_beta,
    }
    torch.save(checkpoint, args.output)
    print(f"Saved checkpoint to: {args.output}")


if __name__ == "__main__":
    main()
