#!/usr/bin/env bash
# Execute the paired full-chain sweep from this checkout; no Git/Docker actions.
set -Eeuo pipefail
LI_CHAIN_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)
exec "${LI_CHAIN_PYTHON:-python3}" "$LI_CHAIN_ROOT/tests/operators/lifetch/full_chain_compare.py" "$@"
