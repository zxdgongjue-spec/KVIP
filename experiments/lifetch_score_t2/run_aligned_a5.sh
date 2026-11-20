#!/usr/bin/env bash
# Same operands/timing: MR23, window, MR14 full-H score/TopK + MR23 completed tail.
set -euo pipefail
LI_ALIGNED_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON:-python3}" "$LI_ALIGNED_DIR/run_a5.py" \
  --cases mr23,window_budget,aligned_budget --prefetch 0 "$@"
