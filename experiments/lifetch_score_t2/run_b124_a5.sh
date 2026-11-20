#!/usr/bin/env bash
# Same bench configuration for MR23, window and Q-parallel window routes.
set -euo pipefail
LI_B124_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON:-python3}" "$LI_B124_SCRIPT_DIR/../../tests/operators/lifetch/test_lifetch.py" bench \
  --input synthetic --transport urma \
  --batches 1,2,4 --miss-percent 5,10,15,20,25,30 \
  --prefetch-tokens 512 --prefetch-rank-start 1024 \
  --intermediate-fetches 4 --li-score-parts-map 1:16,2:16,4:8 \
  --warmup 2 --iterations 10 --timeout-seconds 3600 \
  --compare-routes mr23,window_budget,query_budget "$@"
