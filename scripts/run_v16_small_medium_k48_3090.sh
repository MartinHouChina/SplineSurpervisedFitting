#!/usr/bin/env bash
# Source K=4..20, 48 internal candidates, full cubic knot vector size 56.
# MSE=5e-5; Proposal 64 + Joint 64 = 128 epochs (overridable).
# Keep the K24 entry point intact for historical runs.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$SCRIPT_DIR/run_v16_mse1e-4_3090.sh" --small-medium-k48 "$@"
