#!/usr/bin/env bash
# Direct B16 entry; default30 score AIC/60 score AIV, tail64 AIV.
set -Eeuo pipefail
LI_B16_COMPARE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
exec bash "$LI_B16_COMPARE_DIR/run_li_radix_a5.sh" --batch 16 --score-cores 30 "$@"
