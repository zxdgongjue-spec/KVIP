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
elif [[ ${1:-} == --ranked-tagged ]]; then
  mode=ranked-tagged
  shift
elif [[ ${1:-} == --ranked-bitmap ]]; then
  mode=ranked-bitmap
  shift
elif [[ ${1:-} == --ranked-miss ]]; then
  mode=ranked-miss
  shift
elif [[ ${1:-} == --ranked-floor ]]; then
  mode=ranked-floor
  shift
elif [[ ${1:-} == --ranked-prefetch ]]; then
  mode=ranked-prefetch
  shift
elif [[ ${1:-} == --recent-index ]]; then
  mode=recent-index
  shift
fi
prefetch_tokens=512
positional=()
while (($#)); do
  case "$1" in
    --prefetch-tokens)
      if [[ ! ${2:-} =~ ^[0-9]+$ ]]; then
        echo "--prefetch-tokens requires a nonnegative integer" >&2
        exit 2
      fi
      prefetch_tokens=$2
      shift 2
      ;;
    *) positional+=("$1"); shift ;;
  esac
done
set -- "${positional[@]}"
kvaf_root=${1:?Usage: compare.sh KVAF_ROOT DEVICE SERVER [BATCHES] [MISS] [WINDOWS] [ABC_TAIL_TOKENS] [--prefetch-tokens N]}
device=${2:?Missing device}
server=${3:?Missing server}
batches=${4:-16}
miss=${5:-10}
default_windows=8
if [[ "$mode" == recent-index || "$mode" == ranked-prefetch || "$mode" == ranked-floor || "$mode" == ranked-miss || "$mode" == ranked-bitmap || "$mode" == ranked-tagged ]]; then default_windows=4; fi
windows=${6:-$default_windows}
abc_tail_tokens=${7:-384}
if [[ ! "$windows" =~ ^[1-9][0-9]*$ ]]; then
  echo "WINDOWS must be a positive integer" >&2
  exit 2
fi
run="$repo/tests/results/$mode/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$run"
printf '%s\n' "$mode" > "$run/experiment.txt"
echo "Experiment: $mode; prefetch_tokens=$prefetch_tokens; results=$run"
if [[ "$mode" == ranked-tagged ]]; then
  python "$repo/tests/experiments/incremental_lifetch/prepare_ranked.py" "$run/baseline-source" --minimum-half --true-miss --abc-tail-tokens "$abc_tail_tokens"
  python "$repo/tests/experiments/incremental_lifetch/prepare_ranked.py" "$run/incremental-source" --minimum-half --tagged-miss --abc-tail-tokens "$abc_tail_tokens"
elif [[ "$mode" == ranked-bitmap ]]; then
  python "$repo/tests/experiments/incremental_lifetch/prepare_ranked.py" "$run/baseline-source" --minimum-half --true-miss --abc-tail-tokens "$abc_tail_tokens"
  python "$repo/tests/experiments/incremental_lifetch/prepare_ranked.py" "$run/incremental-source" --minimum-half --bitmap-miss --abc-tail-tokens "$abc_tail_tokens"
elif [[ "$mode" == ranked-miss ]]; then
  python "$repo/tests/experiments/incremental_lifetch/prepare_ranked.py" "$run/baseline-source" --minimum-half --abc-tail-tokens "$abc_tail_tokens"
  python "$repo/tests/experiments/incremental_lifetch/prepare_ranked.py" "$run/incremental-source" --minimum-half --true-miss --abc-tail-tokens "$abc_tail_tokens"
elif [[ "$mode" == ranked-floor ]]; then
  python "$repo/tests/experiments/incremental_lifetch/prepare_ranked.py" "$run/baseline-source" --abc-tail-tokens "$abc_tail_tokens"
  python "$repo/tests/experiments/incremental_lifetch/prepare_ranked.py" "$run/incremental-source" --minimum-half --abc-tail-tokens "$abc_tail_tokens"
elif [[ "$mode" == ranked-prefetch ]]; then
  python "$repo/tests/experiments/incremental_lifetch/prepare_ranked.py" "$run/baseline-source" --baseline --abc-tail-tokens "$abc_tail_tokens"
  python "$repo/tests/experiments/incremental_lifetch/prepare_ranked.py" "$run/incremental-source" --abc-tail-tokens "$abc_tail_tokens"
elif [[ "$mode" == recent-index ]]; then
  python "$repo/tests/experiments/incremental_lifetch/prepare_recent.py" "$run/baseline-source" --baseline --abc-tail-tokens "$abc_tail_tokens"
  python "$repo/tests/experiments/incremental_lifetch/prepare_recent.py" "$run/incremental-source" --abc-tail-tokens "$abc_tail_tokens"
else
  python "$repo/tests/experiments/incremental_lifetch/prepare.py" "$run/incremental-source" "${prepare_flags[@]}"
  python "$repo/tests/experiments/incremental_lifetch/prepare.py" "$run/baseline-source" --baseline "${prepare_flags[@]}"
fi
common=(--transport urma --device "$device" --kvaf-root "$kvaf_root" --kvaf-server "$server"
  --batches "$batches" --miss-percent "$miss" --history 131072 --resident 16384 --index-heads 32
  --prefetch-tokens "$prefetch_tokens" --intermediate-fetches "$windows" --warmup 4 --iterations 20 --timeout-seconds 600)
for variant in baseline incremental; do
  source_root="$run/baseline-source"
  if [[ "$variant" == incremental ]]; then source_root="$run/incremental-source"; fi
  echo "=== $variant: BS=$batches miss=$miss% windows=$windows; graph timing + level 3 ==="
  python "$source_root/tests/operators/lifetch/test_lifetch.py" diagnose "${common[@]}" --level 3 \
    --output "$run/$variant" 2>&1 | tee "$run/$variant.log"
done
python "$repo/tests/experiments/incremental_lifetch/summarize.py" "$run" --require-detailed
