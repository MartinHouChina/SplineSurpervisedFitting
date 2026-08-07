from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .models.spline_network import SplineFittingNetwork


PREVIOUS_OBJECTIVE_VERSION = "cross_attention_true_params_hard_concrete_v1"
CURRENT_OBJECTIVE_VERSION = "independent_query_supervised_hard_concrete_v1"


LEGACY_LOSS_CONFIG: dict[str, Any] = {
    "weights": {
        "fit": 1.0,
        "l0": 0.0,
        "activity": 2e-3,
        "binary": 2e-4,
        "orthogonal": 5e-2,
        "gap": 1e-2,
        "parameter_prior": 1e-2,
        "true_parameter": 0.0,
        "existence": 0.0,
        "knot_position": 0.0,
        "count": 0.0,
    },
    "min_knot_gap": 1e-3,
}

A_SCHEME_WITH_ORTHOGONAL_LOSS_CONFIG: dict[str, Any] = {
    "weights": {
        "fit": 1.0,
        "l0": 2e-5,
        "activity": 0.0,
        "binary": 0.0,
        "orthogonal": 5e-2,
        "gap": 1e-2,
        "parameter_prior": 1e-2,
        "true_parameter": 0.0,
        "existence": 0.0,
        "knot_position": 0.0,
        "count": 0.0,
    },
    "min_knot_gap": 1e-3,
}

PREVIOUS_OBJECTIVE_LOSS_CONFIG: dict[str, Any] = {
    "weights": {
        "fit": 1.0,
        "l0": 2e-5,
        "activity": 0.0,
        "binary": 0.0,
        "orthogonal": 0.0,
        "gap": 1e-2,
        "parameter_prior": 0.0,
        "true_parameter": 1e-2,
        "existence": 0.0,
        "knot_position": 0.0,
        "count": 0.0,
    },
    "min_knot_gap": 1e-3,
}

A_SCHEME_LOSS_CONFIG: dict[str, Any] = {
    "weights": {
        "fit": 1.0,
        "l0": 0.0,
        "activity": 0.0,
        "binary": 0.0,
        # Compatibility field only; it is absent from the current objective.
        "orthogonal": 0.0,
        "gap": 0.0,
        "parameter_prior": 0.0,
        "true_parameter": 1e-2,
        "existence": 1e-3,
        "knot_position": 1e-2,
        "count": 1e-3,
    },
    "min_knot_gap": 1e-3,
}


def migrate_model_config(
    checkpoint: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Return an explicit model config and whether the checkpoint is legacy.

    Pre-A checkpoints have no ``gate_mode``.  Treating those sigmoid-head
    weights as Hard-Concrete logits would silently change their meaning, so
    absence of the field is deliberately migrated to ``legacy_soft``.
    """
    if "model_config" not in checkpoint:
        raise KeyError("checkpoint is missing model_config")
    config = dict(checkpoint["model_config"])
    legacy = "gate_mode" not in config
    if legacy:
        config["gate_mode"] = "legacy_soft"
        config["activity_use_local_context"] = False
        config["gap_parameterization"] = "legacy"
        config["activity_use_pilot_importance"] = False
    elif "activity_use_pilot_importance" not in config:
        # Preserve checkpoints trained before the pilot-importance feature was
        # introduced instead of silently changing their activity logits.
        config["activity_use_pilot_importance"] = False
    if "knot_use_local_cross_attention" not in config:
        # Every checkpoint created before the cross-attention KnotHead used the
        # global-only MLP. Reconstruct that exact module layout for strict load.
        config["knot_use_local_cross_attention"] = False
    if "knot_parameterization" not in config:
        # v0.4 and earlier shared K+1 interval logits. Recreate that exact
        # parameter layout; every new checkpoint records this field explicitly.
        config["knot_parameterization"] = "interval"
    if "activity_use_query_features" not in config:
        config["activity_use_query_features"] = False
    if "compute_first_derivative" not in config:
        saved_weights = checkpoint.get("loss_config", {}).get("weights", {})
        if "orthogonal" in saved_weights:
            needs_derivative = float(saved_weights["orthogonal"]) != 0.0
        else:
            # Every checkpoint predating CURRENT_OBJECTIVE_VERSION used the
            # projection-orthogonality objective (or its historical default).
            needs_derivative = checkpoint.get("objective_version") not in {
                CURRENT_OBJECTIVE_VERSION,
                PREVIOUS_OBJECTIVE_VERSION,
            }
        config["compute_first_derivative"] = needs_derivative
    return config, legacy


def migrate_loss_config(
    checkpoint: Mapping[str, Any],
    *,
    legacy: bool,
) -> tuple[dict[str, Any], bool]:
    """Load loss metadata without inventing an L0 term for old checkpoints."""
    assumed = "loss_config" not in checkpoint
    objective_version = checkpoint.get("objective_version")
    current_objective = objective_version == CURRENT_OBJECTIVE_VERSION
    no_orthogonal_objective = objective_version in {
        CURRENT_OBJECTIVE_VERSION,
        PREVIOUS_OBJECTIVE_VERSION,
    }
    if assumed:
        if legacy:
            default_config = LEGACY_LOSS_CONFIG
        elif current_objective:
            default_config = A_SCHEME_LOSS_CONFIG
        elif objective_version == PREVIOUS_OBJECTIVE_VERSION:
            default_config = PREVIOUS_OBJECTIVE_LOSS_CONFIG
        else:
            default_config = A_SCHEME_WITH_ORTHOGONAL_LOSS_CONFIG
        config = deepcopy(default_config)
    else:
        config = deepcopy(checkpoint["loss_config"])
        weights = config.setdefault("weights", {})
        weights.setdefault(
            "l0",
            0.0 if legacy else A_SCHEME_LOSS_CONFIG["weights"]["l0"],
        )
        weights.setdefault("activity", 0.0)
        weights.setdefault("binary", 0.0)
        weights.setdefault("true_parameter", 0.0)
        weights.setdefault("existence", 0.0)
        weights.setdefault("knot_position", 0.0)
        weights.setdefault("count", 0.0)
        weights.setdefault(
            "orthogonal",
            (
                0.0
                if no_orthogonal_objective
                else LEGACY_LOSS_CONFIG["weights"]["orthogonal"]
            ),
        )
    config.setdefault("knot_position_beta", 0.02)
    return config, assumed


def build_model_from_checkpoint(
    checkpoint: Mapping[str, Any],
) -> tuple[SplineFittingNetwork, dict[str, Any], bool]:
    """Construct a model with explicit legacy migration and restore gate state."""
    config, legacy = migrate_model_config(checkpoint)
    model = SplineFittingNetwork(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    temperature = checkpoint.get("gate_temperature")
    if temperature is not None and hasattr(model, "set_gate_temperature"):
        model.set_gate_temperature(float(temperature))
    if hasattr(model, "set_force_open_gates"):
        model.set_force_open_gates(False)
    return model, config, legacy
