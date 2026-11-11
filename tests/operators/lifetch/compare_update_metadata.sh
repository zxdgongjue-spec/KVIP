#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
kvaf="${1:?Usage: compare_update_metadata.sh KVAF_ROOT DEVICE HOST:PORT [BATCHES] [MISS_PERCENT]}"
device="${2:?Missing device}"
server="${3:?Missing server}"
batches="${4:-16}"
miss="${5:-10}"
kvaf="$(cd "$kvaf" && pwd)"
run="$repo/tests/results/lifetch-update-metadata-ab/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$run"
# Both versions leave parallel Fetch finalization disabled.
a_ref=00e6b61
b_ref="$(git -C "$repo" rev-parse HEAD)"
printf 'A=%s\nB=%s\n' "$(git -C "$repo" rev-parse "$a_ref")" "$b_ref" | tee "$run/versions.log"
trap 'status=$?; if ((status != 0)); then echo "FAIL exit=$status logs=$run"; fi' EXIT
source "$repo/scripts/env.sh"
for variant in A B; do
    ref="$a_ref"
    extra=()
    if [[ "$variant" == B ]]; then ref="$b_ref"; extra+=(--prepare-update-metadata); fi
    code="$run/$variant-source"
    git -C "$repo" worktree add --detach "$code" "$ref"
    mkdir -p "$code/csrc/operators/lidu/_custom_opp"
    ln -s "$repo/csrc/operators/lidu/_custom_opp/vendors" "$code/csrc/operators/lidu/_custom_opp/vendors"
    echo "=== $variant; batches=$batches miss=$miss% ==="
    (
      cd "$code"
      export PYTHONPATH="$code/python:$kvaf/build-npu/python${PYTHONPATH:+:$PYTHONPATH}"
      python tests/operators/lifetch/test_lifetch.py \
        --transport urma --streaming --streaming-fetch --fused-final --q-workers 2 \
        --device "$device" --kvaf-root "$kvaf" --kvaf-server "$server" \
        --batches "$batches" --history 131072 --resident 16384 --index-heads 32 \
        --intermediate-fetches 8 --prefetch-tokens 512 --miss-percent "$miss" \
        --warmup 4 --iterations 20 --trace-rounds 3 --timeout-seconds 900 \
        --output "$run/$variant" "${extra[@]}"
    ) 2>&1 | tee "$run/$variant.log"
    python "$repo/tests/operators/lifetch/support/summarize_core_timeline.py" "$run/$variant"
done
python "$repo/tests/operators/lifetch/support/summarize_update_metadata_ab.py" "$run"
