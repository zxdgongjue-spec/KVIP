#!/usr/bin/env bash
set -eo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
mode=incremental-lifetch
prepare_flags=()
if [[ ${1:-} == --window-topk-only ]]; then
  mode=window-topk-only
  prepare_flags=(--window-topk-only)
  shift
elif [[ ${1:-} == --final-topk-only ]]; then
  mode=final-topk-only
  prepare_flags=(--final-topk-only)
  shift
elif [[ ${1:-} == --topk-prefetch ]]; then
  mode=topk-prefetch
  prepare_flags=(--topk-prefetch)
  shift
elif [[ ${1:-} == --recent-index ]]; then
  mode=recent-index
  shift
fi
kvaf_root=${1:?Usage: compare.sh KVAF_ROOT DEVICE SERVER [BATCHES] [MISS] [WINDOWS]}
device=${2:?Missing device}
server=${3:?Missing server}
batches=${4:-16}
miss=${5:-10}
default_windows=8
if [[ "$mode" == recent-index ]]; then default_windows=4; fi
windows=${6:-$default_windows}
abc_tail_tokens=${7:-384}
if [[ ! "$windows" =~ ^[1-9][0-9]*$ ]]; then
  echo "WINDOWS must be a positive integer" >&2
  exit 2
fi
run="$repo/tests/results/$mode/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$run"
printf '%s\n' "$mode" > "$run/experiment.txt"
echo "Experiment: $mode; results=$run"
if [[ "$mode" == recent-index ]]; then
  python "$repo/tests/experiments/incremental_lifetch/prepare_recent.py" "$run/baseline-source" --baseline --abc-tail-tokens "$abc_tail_tokens"
  python "$repo/tests/experiments/incremental_lifetch/prepare_recent.py" "$run/incremental-source" --abc-tail-tokens "$abc_tail_tokens"
else
  python "$repo/tests/experiments/incremental_lifetch/prepare.py" "$run/incremental-source" "${prepare_flags[@]}"
  python "$repo/tests/experiments/incremental_lifetch/prepare.py" "$run/baseline-source" --baseline "${prepare_flags[@]}"
fi
common=(--transport urma --device "$device" --kvaf-root "$kvaf_root" --kvaf-server "$server"
  --batches "$batches" --miss-percent "$miss" --history 131072 --resident 16384 --index-heads 32
  --prefetch-tokens 512 --intermediate-fetches "$windows" --warmup 4 --iterations 20 --timeout-seconds 600)
for variant in baseline incremental; do
  source_root="$run/baseline-source"
  if [[ "$variant" == incremental ]]; then source_root="$run/incremental-source"; fi
  echo "=== $variant: BS=$batches miss=$miss% windows=$windows; graph timing + level 3 ==="
  python "$source_root/tests/operators/lifetch/test_lifetch.py" diagnose "${common[@]}" --level 3 \
    --output "$run/$variant" 2>&1 | tee "$run/$variant.log"
done
python "$repo/tests/experiments/incremental_lifetch/summarize.py" "$run" --require-detailed
