#!/usr/bin/env bash
# Source K=4..20, 24 internal candidates, full cubic knot vector size 32.
# A scoped diagnostic; it does not claim the Keep-selection issue is solved.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$SCRIPT_DIR/run_v16_mse1e-4_3090.sh" --small-medium-k24 "$@"
