#!/usr/bin/env bash
# Same descriptor snapshot, queue lease and KVAF device entry on both sides.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
kvaf="${1:?Usage: compare_fetch_only.sh KVAF_ROOT DEVICE HOST:PORT [BATCH]}"
device="${2:?Missing device}"
server="${3:?Missing server}"
batch="${4:-16}"
source "$repo/scripts/env.sh"
python "$repo/tests/operators/lifetch/test_lifetch.py" \
  --transport urma --streaming --streaming-fetch --fused-final --q-workers 2 \
  --fetch-only-compare --device "$device" --kvaf-root "$kvaf" --kvaf-server "$server" \
  --batches "$batch" --history 131072 --resident 16384 --index-heads 32 \
  --intermediate-fetches 8 --prefetch-tokens 512 --miss-percent 10 \
  --warmup 10 --iterations 50 --trace-rounds 0 --timeout-seconds 900
