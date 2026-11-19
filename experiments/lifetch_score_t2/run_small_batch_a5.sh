#!/usr/bin/env bash
# Small-batch control: original MR23, current window, query-parallel TopK/post.
set -euo pipefail
LI_SMALL_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON:-python3}" "$LI_SMALL_DIR/run_a5.py" \
  --batches 2,4 --cases mr23,window_budget,query_budget "$@"
