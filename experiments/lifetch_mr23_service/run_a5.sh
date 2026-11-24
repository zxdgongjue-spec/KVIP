#!/usr/bin/env bash
set -euo pipefail

readonly baseline_commit=a56154c37bebb0928da4a4095e9670898e6b21c7
readonly candidate_commit=7b4e3d52d7a85451d1982a6e04823e4e18ecabfb
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)

for revision in "$baseline_commit" "$candidate_commit"; do
    if ! git -C "$repo" cat-file -e "${revision}^{commit}" 2>/dev/null; then
        printf 'Missing %s. Fetch refs/heads/codex/lifetch-score-t2 from the KVIP main repository.\n' "$revision" >&2
        exit 2
    fi
done

kvaf_root=${KVIP_KVAF_ROOT:-"$repo/third_party/KVAF"}
case "$kvaf_root" in
    /*) ;;
    *) kvaf_root="$PWD/$kvaf_root" ;;
esac

if [[ -n ${KVIP_LIDU_OPP:-} ]]; then
    opp=$(cd -- "$KVIP_LIDU_OPP" && pwd)
    if [[ ! -f "$opp/op_api/lib/libcust_opapi.so" ]]; then
        printf 'KVIP_LIDU_OPP must point to the built LIDU vendor directory.\n' >&2
        exit 2
    fi
    export ASCEND_CUSTOM_OPP_PATH="$opp${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
fi

output=${KVIP_AB_OUTPUT:-"$PWD/results/mr23_service"}
mkdir -p -- "$output"
output=$(cd -- "$output" && pwd)
run_dir=$(mktemp -d "$output/run.XXXXXX")
source_dir="$run_dir/score-t2"
git clone --quiet --shared --no-checkout -- "$repo" "$source_dir"
git -C "$source_dir" checkout --quiet --detach "$candidate_commit"
git -C "$source_dir" apply --check --whitespace=error "$here/runner.patch"
git -C "$source_dir" apply --whitespace=error "$here/runner.patch"
printf 'Experiment source: %s\n' "$source_dir"
cd "$source_dir"

exec "${KVIP_PYTHON:-python3}" -B tests/operators/lifetch/test_lifetch.py bench \
    --input synthetic --transport urma --device "${KVIP_A5_DEVICE:-0}" \
    --kvaf-root "$kvaf_root" --kvaf-server "${KVIP_KVAF_SERVER:-127.0.0.1:28985}" \
    --batches 1,8,16 --miss-percent 10 --queries 4 \
    --history 131072 --resident 16384 --index-heads 32 --seed 43 \
    --prefetch-tokens 512 --prefetch-rank-start 1024 --intermediate-fetches 4 \
    --compare-routes mr23,window_budget,full_budget --li-score-parts-map 1:16,8:4,16:2 \
    --warmup 2 --iterations 10 --comparison-rounds 1 --timeout-seconds 1800 \
    --output "$run_dir/results" "$@"
