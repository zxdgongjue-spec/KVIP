#!/usr/bin/env bash
set -eo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
kvaf_root=${1:?Usage: compare.sh KVAF_ROOT DEVICE SERVER [BATCHES] [MISS]}
device=${2:?Missing device}
server=${3:?Missing server}
batches=${4:-16}
miss=${5:-10}
run="$repo/tests/results/incremental-lifetch/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$run"
python "$repo/tests/experiments/incremental_lifetch/prepare.py" "$run/incremental-source"
python "$repo/tests/experiments/incremental_lifetch/prepare.py" "$run/baseline-source" --baseline
common=(--transport urma --device "$device" --kvaf-root "$kvaf_root" --kvaf-server "$server"
  --batches "$batches" --miss-percent "$miss" --history 131072 --resident 16384 --index-heads 32
  --prefetch-tokens 512 --intermediate-fetches 8 --warmup 4 --iterations 20 --timeout-seconds 600)
for variant in baseline incremental; do
  source_root="$run/baseline-source"
  if [[ "$variant" == incremental ]]; then source_root="$run/incremental-source"; fi
  echo "=== $variant: BS=$batches miss=$miss%; graph timing + level 3 ==="
  python "$source_root/tests/operators/lifetch/test_lifetch.py" diagnose "${common[@]}" --level 3 \
    --output "$run/$variant" 2>&1 | tee "$run/$variant.log"
done
python "$repo/tests/experiments/incremental_lifetch/summarize.py" "$run" --require-detailed
