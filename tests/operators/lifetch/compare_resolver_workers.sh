#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
kvaf="${1:?Usage: compare_resolver_workers.sh KVAF_ROOT DEVICE HOST:PORT}"
device="${2:?Missing device ID}"
server="${3:?Missing KVAF CPU service address}"
kvaf="$(cd "$kvaf" && pwd)"
run="$repo/tests/results/resolver-workers/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$run"
trap 'status=$?; if ((status != 0)); then printf "FAIL exit=%s; logs=%s\n" "$status" "$run" | tee "$run/summary.log"; fi' EXIT
source "$repo/scripts/env.sh"
echo "Results: $run"
for workers in 1 2 4; do
    echo "=== final workers/request=$workers; batches=1,2,4,8,16 ==="
    python "$repo/tests/operators/lifetch/test_lifetch.py" \
      --transport urma --avoid-prefetch-slots --final-workers "$workers" \
      --device "$device" --kvaf-root "$kvaf" --kvaf-server "$server" \
      --batches 1,2,4,8,16 --history 131072 --resident 16384 --index-heads 32 \
      --intermediate-fetches 8 --prefetch-tokens 512 --miss-percent 10 \
      --warmup 4 --iterations 20 --trace-rounds 3 --timeout-seconds 1800 \
      --output "$run/w$workers" 2>&1 | tee "$run/w$workers.log"
done
python3 "$repo/tests/operators/lifetch/compare_resolver_workers.py" "$run"
