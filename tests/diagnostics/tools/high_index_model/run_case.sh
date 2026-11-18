#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail

if [[ $# -ne 4 ]] || [[ $1 != npu && $1 != sim ]]; then
    echo 'Usage: run_case.sh npu|sim KVIP_WORKTREE KVAF_INCLUDE NEW_OR_EXISTING_CASE_DIR' >&2
    exit 2
fi
mode=$1
repo=$(cd "$2" && pwd -P)
include_dir=$(cd "$3" && pwd -P)
mkdir -p "$4"
output=$(cd "$4" && pwd -P)
self=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cann=${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/cann-9.2.0}
soc=${SOC_VERSION:-Ascend950DT_9582}
mkdir -p "$output/tmp"
export TMPDIR="$output/tmp"
source "$cann/set_env.sh"
set -u

python3 -B "$self/prepare.py" "$repo" "$output" "$mode" "$soc" "$include_dir"
if [[ ! -x "$output/high_index_case" ]]; then
    # CANN's preprocess step rewrites relocatable objects in place. Reuse a
    # finished executable, and retry an interrupted build in a new directory.
    build=$(mktemp -d "$output/build.XXXXXX")
    if ! { cmake -S "$output" -B "$build" -DCMAKE_BUILD_TYPE=Release -DRUN_MODE="$mode" \
               -DSOC_VERSION="$soc" -DASCEND_CANN_PACKAGE_PATH="$cann" -DKVAF_INCLUDE="$include_dir" &&
           cmake --build "$build" --parallel "${MAX_JOBS:-2}"; } > "$build/build.log" 2>&1; then
        tail -25 "$build/build.log" >&2
        exit 1
    fi
    ln -s "$build/high_index_case" "$output/high_index_case"
fi
if [[ $mode == sim ]]; then
    export CAMODEL_CONFIG_PATH="$self/model"
    export LD_LIBRARY_PATH="$cann/tools/simulator/Ascend950PR_9582/lib:${LD_LIBRARY_PATH:-}"
fi
cd "$output"
# The native executable sees one logical device after ASCEND_RT_VISIBLE_DEVICES is set.
read -r -a layouts <<< "${INDEX_LAYOUTS:-even skew}"
for layout in "${layouts[@]}"; do
    case "$layout" in even|skew|even-mixed|skew-mixed|even-events|skew-events|mixed-fallback|all-fallback) ;; *) echo "Unknown layout $layout" >&2; exit 2;; esac
    run_dir=$(mktemp -d "$output/$layout.XXXXXX")
    if [[ $mode == sim ]]; then export CAMODEL_LOG_PATH="$run_dir"; fi
    if ! (cd "$run_dir" && "$output/high_index_case" "$layout") >"$run_dir/run.log" 2>&1; then
        tail -20 "$run_dir/run.log" >&2
        exit 1
    fi
    grep -E '^(INDEX_MODEL_START|DEVICE_INDEX_US|COLD_PLAN_MODEL|TOUCHED_PLAN_MODEL|INDEX_WORK|MODEL_EXACT_PASS)' "$run_dir/run.log"
done
