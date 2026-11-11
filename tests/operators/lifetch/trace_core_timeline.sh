#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
kvaf="${1:?Usage: trace_core_timeline.sh KVAF_ROOT DEVICE HOST:PORT [BATCH]}"
device="${2:?Missing device}"
server="${3:?Missing server}"
batch="${4:-16}"
run="$repo/tests/results/lifetch-core-timeline/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$run"
source "$repo/scripts/env.sh"
python "$repo/tests/operators/lifetch/test_lifetch.py" \
  --transport urma --streaming --streaming-fetch --fused-final --q-workers 2 \
  --device "$device" --kvaf-root "$kvaf" --kvaf-server "$server" \
  --batches "$batch" --history 131072 --resident 16384 --index-heads 32 \
  --intermediate-fetches 8 --prefetch-tokens 512 --miss-percent 10 \
  --warmup 4 --iterations 20 --trace-rounds 3 --timeout-seconds 900 \
  --output "$run" 2>&1 | tee "$run/run.log"
python "$repo/tests/operators/lifetch/support/summarize_core_timeline.py" "$run"
