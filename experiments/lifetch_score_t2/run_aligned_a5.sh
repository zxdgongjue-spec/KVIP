#!/usr/bin/env bash
# MR23 baseline, MR23+MR14 window score, MR23+MR14 full-H score/TopK/classify.
set -euo pipefail
LI_ALIGNED_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON:-python3}" "$LI_ALIGNED_DIR/run_a5.py" \
  --cases mr23,window_budget,aligned_budget --prefetch 0 \
  --batches 1,2,4,8,16 --miss-percent 5,10,15,20,25,30 "$@"
