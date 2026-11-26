#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Five real Q4 traces paired with the strong two-worker serial path.
set -eo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
work=${PEER_G2_WORK:-/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold}
reuse_serial=
archive=
fixed=0
fixed_archive=
probe_archive=
batches=(20 24 28 32)
groups=(10 4 17 7 2)
if (( $# )); then
    if [[ $# == 2 && $1 == --candidate-only ]]; then
        archive=$(cd "$2" && pwd)
    elif [[ $# == 2 && $1 == --reuse-g10-bs20-serial ]]; then
        reuse_serial=$2
    elif [[ $1 == --fixed-q4-probes && $# == 2 ]]; then
        fixed=1
        probe_archive=$(cd "$2" && pwd)
    elif [[ $1 == --fixed-q4 && $# -le 2 ]]; then
        fixed=1
        if (( $# == 2 )); then
            fixed_archive=$2
        fi
        read -ra batches <<<"${FIXED_Q4_BATCHES:-1 2 4 8 16 20 24 28 32}"
        read -ra groups <<<"${FIXED_Q4_GROUPS:-10 4 17 7 2}"
        validate_selection() {
            local valid=$1 seen=' ' v
            shift
            (( $# > 0 )) || return 1
            for v do
                [[ " $valid " == *" $v "* && "$seen" != *" $v "* ]] || return 1
                seen+="$v "
            done
        }
        validate_selection '1 2 4 8 16 20 24 28 32' "${batches[@]}" &&
            validate_selection '10 4 17 7 2' "${groups[@]}" || { echo "Invalid FIXED_Q4_BATCHES/GROUPS" >&2; exit 2; }
    else
        echo "Usage: $0 [--fixed-q4 [LARGE_ARCHIVE] | --fixed-q4-probes NPU3_ARCHIVE | --candidate-only RESULT_ROOT | --reuse-g10-bs20-serial RESULT_ROOT]" >&2
        exit 2
    fi
fi
peer_physical_npu=4
if (( fixed )); then
    peer_physical_npu=${PEER_PHYSICAL_NPU:-${ASCEND_RT_VISIBLE_DEVICES:-4}}
    if [[ $peer_physical_npu != 3 && $peer_physical_npu != 4 ]] ||
        [[ -n ${ASCEND_RT_VISIBLE_DEVICES:-} && $ASCEND_RT_VISIBLE_DEVICES != "$peer_physical_npu" ]]; then
        echo "Select one matching physical card (3 or 4) for PEER_PHYSICAL_NPU and ASCEND_RT_VISIBLE_DEVICES" >&2
        exit 2
    fi
elif [[ ${PEER_PHYSICAL_NPU:-4} == 3 || ${ASCEND_RT_VISIBLE_DEVICES:-4} == 3 ]]; then
    echo "Use --fixed-q4 to run the matrix on physical NPU3" >&2
    exit 2
fi
export PEER_PHYSICAL_NPU="$peer_physical_npu" ASCEND_RT_VISIBLE_DEVICES="$peer_physical_npu"
if [[ -n $probe_archive ]]; then
    [[ $peer_physical_npu == 3 && $(<"$probe_archive/physical_npu.txt") == 3 &&
       $(<"$probe_archive/native.txt") == af52a02b97daddadd5d93b9d100f2ff1b03b499d ]] ||
        { echo "Expected the fixed Q4 measurement archive for physical NPU3" >&2; exit 1; }
fi
area="$work/peer-baseline"
sources="$work/fused-scan-sources"
mkdir -p "$area" "$sources"
prefix=largebs-q4
if (( fixed )); then prefix=fixed-q4; fi
if [[ -n $probe_archive ]]; then prefix=fixed-q4-li-tail; fi
out=$(mktemp -d "$area/$prefix.XXXXXX")
echo "RESULT_ROOT=$out"
log="$out/run.log"
# A single transcript for setup/driver errors, with compact results on stdout.
exec 3>&1
exec >"$log" 2>&1
trap 'rc=$?; if ((rc)); then tail -25 "$log" >&3; printf "FAILED rc=%s LOG=%s\n" "$rc" "$log" >&3; fi' EXIT
export PEER_LOCAL_ONLY=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PATH="/usr/local/python3.12.13/bin:$PATH"
unset PEER_RANKED_INPUT PEER_RANKED_HASH_FILE
if [[ -r /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
else
    source /usr/local/Ascend/cann-9.2.0/set_env.sh
fi
if [[ -r /usr/local/Ascend/nnal/atb/set_env.sh ]]; then source /usr/local/Ascend/nnal/atb/set_env.sh; fi
set -u
support_rev=e34e8e0411d6f3decb3fd6aabff877e3a419b4f9
sdk_rev=813f1750a50b7e6c92d5f5397f2fbc05befef6da
sdk="$sources/unified-sdk-813f1750-d6cfd392"
workers="$root/tests/operators/lifetch/support"
host_tools="$out/host-tools"
mkdir "$host_tools"
if ! git -C "$root" cat-file -e "$support_rev^{commit}" 2>/dev/null; then
    git -C "$root" fetch -q --no-tags https://gitcode.com/CacheOS/KVIP.git refs/heads/test/lifetch-pr30-fast-accounting
fi
git -C "$root" archive "$support_rev" tests/operators/lifetch/{peer_g2_input.sh,support/{workloads.py,peer_thp.py,query_owner_input.py,peer_node_prepare.sh}} | tar -x -C "$host_tools"
helpers="$host_tools/tests/operators/lifetch"
source "$helpers/support/peer_node_prepare.sh"
# Protect this run against a second invocation using the same physical card.
exec 9>"$sources/largebs-$(</proc/sys/kernel/random/boot_id)-npu$peer_physical_npu.lock"
flock -n 9 || { echo "A large-batch replay is already running on NPU$peer_physical_npu"; exit 1; }
exec 8>"$sdk.lock"
flock 8
baseline="$work/adaptive-sources/peer-serial-da318bdb"
clean="$area/KVAF-de4fd37e"
peer_prepare_node "$root" "$baseline" "$clean" "$work" "$out"
"$PYTHON" -B -c 'import os; print(os.uname().nodename)' > "$out/host.txt"
if [[ ! -e "$sdk" ]]; then
    if ! git -C "$clean" cat-file -e "$sdk_rev^{commit}" 2>/dev/null; then
        git -C "$clean" fetch -q --no-tags https://gitcode.com/CacheOS/KVAF.git refs/heads/main
    fi
    git -C "$clean" worktree add -q --detach "$sdk" "$sdk_rev"
fi
"$PYTHON" -B "$root/scripts/build.py" --prepare-only --transport urma --kvaf-root "$sdk"
index=$(mktemp -d "$out/sdk-index.XXXXXX")
sdk_git=(env "GIT_INDEX_FILE=$index/index" git -C "$sdk")
"${sdk_git[@]}" read-tree HEAD
"${sdk_git[@]}" add -u
"${sdk_git[@]}" add access_operator/algorithms/get/direct/urma_aiv/include/kvaf_batch_get_device.h
actual=$("${sdk_git[@]}" write-tree)
[[ $actual == 1de18e8315137d7a0227c3ae4c82afd45b5fe743 ]] || { echo "Unexpected KVAF source tree $actual"; exit 1; }
rm -rf -- "$index"
cache="$clean/build-cpu/CMakeCache.txt"
headers=$(sed -n 's/^KVAF_URMA_ROOT:[^=]*=//p' "$cache")
lib=$(sed -n 's/^KVAF_URMA_LIBRARY:[^=]*=//p' "$cache")
if [[ -f "$headers/include/urma_api.h" ]]; then export KVAF_UMDK_ROOT="$headers"; fi
if [[ -f "$lib" ]]; then export KVAF_URMA_LIBRARY="$lib"; fi
export PATH="${PYTHON%/*}:$PATH"
for target in cpu npu; do
    options=(--jobs "${KVAF_BUILD_JOBS:-2}")
    if [[ $target == npu ]]; then options+=(--without-aicpu); fi
    bash "$sdk/build.sh" "$target" "${options[@]}"
done
flock -u 8
exec 8>&-
lidu="$baseline/csrc/operators/lidu"
vendors=("$lidu"/_custom_opp/vendors/*/op_api/lib/libcust_opapi.so)
[[ ${#vendors[@]} == 1 && -s ${vendors[0]} ]]
export ASCEND_CUSTOM_OPP_PATH="${vendors[0]%/op_api/lib/libcust_opapi.so}${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export PYTHONPATH="$root/python:$sdk/build-npu/python:$lidu/torch_extension${PYTHONPATH:+:$PYTHONPATH}"
export KVAF_BATCH_GET_KERNEL="$sdk/build-npu/bin/kvaf_execute_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$sdk/build-npu/bin/kvaf_execute_batch_put_kernels.o"
cpu="$sdk/build-cpu/runtime/kvaf_cpu_service"
if [[ $ASCEND_RT_VISIBLE_DEVICES != "$peer_physical_npu" || $PEER_PHYSICAL_NPU != "$peer_physical_npu" ]]; then
    echo "Dependency setup changed the selected physical card" >&2
    exit 1
fi
printf '%s\n' "$(git -C "$root" rev-parse HEAD)" > "$out/native.txt"
if [[ -r /etc/machine-id ]]; then cat /etc/machine-id > "$out/machine_id.txt"; fi
printf '%s\n' "$peer_physical_npu" > "$out/physical_npu.txt"
cat /proc/sys/kernel/random/boot_id > "$out/boot_id.txt"
printf '%s\n' "$actual" > "$out/kvaf_tree.txt"
git -C "$sdk" rev-parse HEAD > "$out/kvaf.txt"
if (( fixed )) && [[ -z $probe_archive ]]; then
    args=("$out")
    if [[ -n $fixed_archive ]]; then args+=("$fixed_archive"); fi
    fixed_archive=$("$PYTHON" -B "$workers/large_batch_placement.py" archive "${args[@]}")
    printf 'CONFIG physical_NPU=%s logical_NPU=0 large_serial=%s\n' "$peer_physical_npu" "${fixed_archive:-fresh_same_card}" >&3
fi
# Use the same registered CPU memory placement for the serial and fused arms.
placement="auto auto"
if [[ -z $archive && -z $fixed_archive && -z $probe_archive ]]; then
    placement=$("$PYTHON" -B "$workers/large_batch_placement.py" find "$out")
fi
read -r planned_node planned_adapter <<<"$placement"
set_cpu_placement() {
    local node=$1 adapter=$2 ub_device cpus node_dir
    ub_device=${KVAF_CPU_UB_DEVICE:-$adapter}
    command=("$cpu")
    if [[ $node == auto ]]; then
        for node_dir in /sys/devices/system/node/node[0-9]*; do
            [[ -r $node_dir/cpulist ]] || continue
            cpus=$(<"$node_dir/cpulist")
            if [[ -n $cpus ]] && taskset -c "$cpus" true >/dev/null 2>&1; then
                node=${node_dir##*node}
                break
            fi
        done
    fi
    if [[ $node != auto ]]; then
        if command -v numactl >/dev/null && numactl "--membind=$node" true >/dev/null 2>&1; then
            command=(numactl "--membind=$node" "$cpu")
        elif [[ -r /sys/devices/system/node/node$node/cpulist ]]; then
            cpus=$(<"/sys/devices/system/node/node$node/cpulist")
            if taskset -c "$cpus" true >/dev/null 2>&1; then command=(taskset -c "$cpus" "$cpu"); fi
        fi
    fi
    if [[ ${3:-} == archived || ${3:-} == published ]] && {
        [[ $ub_device != "$adapter" || ${#command[@]} == 1 ]] ||
        [[ ${3:-} == archived && $planned_node != auto && $planned_node != "$node" ]] ||
        [[ ${3:-} == archived && $planned_adapter != auto && $planned_adapter != "$adapter" ]]
    }; then
        echo "Archived NUMA=$node UB=$adapter conflicts with current placement or binding permissions" >&2
        return 1
    fi
    if [[ -n $ub_device && $ub_device != auto ]]; then command+=(--ub-device "$ub_device"); fi
}
if [[ -n $fixed_archive ]]; then
    read -r planned_node planned_adapter < <("$PYTHON" -B "$workers/large_batch_placement.py" get "$fixed_archive/g10/bs20/fused/thp.json")
fi
set_cpu_placement "$planned_node" "$planned_adapter"
export PYTHON
run_arm() (
    set -euo pipefail
    local group=$1 batch=$2 trace=$3 arm=$4 pair=$5 cohort=${6:-0}
    dir="$pair/$arm"
    mkdir "$dir"
    port=$("$PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
    "${command[@]}" --listen 127.0.0.1 --port "$port" --memory-page-mode thp >"$dir/cpu.log" 2>&1 &
    server=$!
    monitor=
    cleanup() {
        if [[ -n $monitor ]]; then kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true; fi
        kill "$server" 2>/dev/null || true
        wait "$server" 2>/dev/null || true
    }
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    for ((i=0; i<100; i++)); do
        if grep -Fq "LISTEN address=127.0.0.1 port=$port" "$dir/cpu.log"; then break; fi
        kill -0 "$server" 2>/dev/null || break
        sleep .1
    done
    grep -Fq "LISTEN address=127.0.0.1 port=$port" "$dir/cpu.log" || { tail -20 "$dir/cpu.log"; exit 1; }
    "$PYTHON" -B "$helpers/support/peer_thp.py" --unified-sdk --pid "$server" --log "$dir/cpu.log" --output "$dir/thp.json" --timeout 7200 &
    monitor=$!
    if [[ $arm == candidate || $arm == pr30_reference ]]; then
        (cd "$dir" && timeout --kill-after=30s 7200s bash "$root/tests/operators/lifetch/fixed_bs16_pr30.sh" \
            "$root" "$work" "$out" "$group" "$trace" "$sdk" "127.0.0.1:$port" "$dir" "$arm") >"$dir/worker.log" 2>&1 || { tail -25 "$dir/worker.log"; exit 1; }
        wait "$monitor"
        monitor=
        return
    fi
    args=(--trace "$trace" --group "$group" --batch "$batch" --cohort "$cohort" --arm "$arm" --kvaf-root "$sdk"
          --kvaf-server "127.0.0.1:$port" --output "$dir")
    if (( fixed )); then args+=(--prepared "$out/g$group/donors"); fi
    if [[ $arm == fused ]]; then
        if [[ -n $probe_archive ]]; then
            args+=(--serial-result "$probe_archive/${pair#"$out/"}"/serial --fixed-probe "$probe_kind")
        elif [[ -n $archive ]]; then
            args+=(--serial-result "$archive/g$group/bs$batch/serial" --compare-archived-pair)
        else
            args+=(--serial-result "$pair/serial")
        fi
    fi
    (cd "$dir" && timeout --kill-after=30s 7200s "$PYTHON" -B "$workers/large_batch_pair.py" "${args[@]}") >"$dir/worker.log" 2>&1 || { tail -20 "$dir/worker.log"; exit 1; }
    wait "$monitor"
    monitor=
)
run_pair() {
    local group=$1 batch=$2 trace=$3 cohort=$4 pair=$5
    mkdir -p "$pair"
    echo "CASE_BEGIN G$group BS$batch cohort=$cohort" >&3
    if [[ -n $archive ]]; then
        ref="$archive/g$group/bs$batch"
        "$PYTHON" -B "$workers/large_batch_placement.py" compare "$ref/serial/thp.json" "$ref/fused/thp.json"
        read -r node adapter < <("$PYTHON" -B "$workers/large_batch_placement.py" get "$ref/fused/thp.json")
        set_cpu_placement "$node" "$adapter" archived
        run_arm "$group" "$batch" "$trace" fused "$pair" "$cohort"
        "$PYTHON" -B "$workers/large_batch_placement.py" compare "$ref/fused/thp.json" "$pair/fused/thp.json"
        grep -E '^(CHECK|TIMING|TRAFFIC|DIAGNOSTIC|SAVED) ' "$pair/fused/worker.log" | tee -a "$out/summary.txt" >&3
        rm -f -- "$pair/fused/worker.log" "$pair/fused/cpu.log"
        return
    fi
    if [[ -n $reuse_serial && $group == 10 && $batch == 20 ]]; then
        staged=$(mktemp -d "$pair/serial-import.XXXXXX")
        if "$PYTHON" -B "$workers/large_batch_pair.py" --arm serial --trace "$trace" --group 10 --batch 20 \
            --kvaf-root "$sdk" --output "$staged" --reuse-serial-archive "$reuse_serial" >"$staged/worker.log" 2>&1 &&
            read -r archived_node archived_adapter < <("$PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r["node"],r["adapter"])' "$staged/serial_reference.json") &&
            set_cpu_placement "$archived_node" "$archived_adapter" archived >>"$staged/worker.log" 2>&1; then
            mv "$staged" "$pair/serial"
            echo "REUSED G10 BS20 serial source=$reuse_serial NUMA=$archived_node UB=$archived_adapter" >&3
        else
            printf 'REMEASURE G10 BS20 serial reason=%s\n' "$(tail -1 "$staged/worker.log")" >&3
            rm -rf -- "$staged"
            set_cpu_placement "$planned_node" "$planned_adapter"
        fi
    fi
    if [[ ! -d $pair/serial ]]; then run_arm "$group" "$batch" "$trace" serial "$pair" "$cohort"; fi
    if (( fixed )) && [[ $planned_node == auto || $planned_adapter == auto ]]; then
        read -r planned_node planned_adapter < <("$PYTHON" -B "$workers/large_batch_placement.py" get "$pair/serial/thp.json")
        set_cpu_placement "$planned_node" "$planned_adapter" archived
    fi
    run_arm "$group" "$batch" "$trace" fused "$pair" "$cohort"
    "$PYTHON" -B "$workers/large_batch_placement.py" compare "$pair/serial/thp.json" "$pair/fused/thp.json"
    for arm in serial fused; do
        grep -E '^(CHECK|TIMING|SAVED) ' "$pair/$arm/worker.log" | tee -a "$out/summary.txt" >&3
        rm -f -- "$pair/$arm/worker.log" "$pair/$arm/cpu.log"
    done
}

if [[ -n $probe_archive ]]; then
    # These cases all have their own original two-worker serial outputs.
    # Fail before opening a queue if any input, placement or oracle is missing.
    for marker in host machine_id physical_npu kvaf kvaf_tree; do
        cmp -s "$out/$marker.txt" "$probe_archive/$marker.txt" ||
            { echo "Archive mismatch in $marker: $probe_archive"; exit 1; }
    done
    rows=(1 3 4 8 12 14 0 2 5 6 7 9 10 11 13 15)
    cells=(g2/bs32 g10/bs32)
    for r in "${rows[@]}"; do cells+=("g17/bs1/cohort$r"); done
    evidence=(summary.txt archive.txt native.txt physical_npu.txt)
    for cell in "${cells[@]}"; do
        evidence+=("$cell/fused/PROBE_RESULT.json")
        for file in serial/{ARM_RESULT.json,selected.npy,thp.json,path/path_result.json} fused/{ARM_RESULT.json,thp.json}; do
            [[ -s $probe_archive/$cell/$file ]] || { echo "Missing archive file: $probe_archive/$cell/$file"; exit 1; }
        done
    done
    echo "CONFIG physical_NPU=3 logical_device=0 serial=archive_only source=$probe_archive" >&3
    printf '%s\n' "$probe_archive" > "$out/archive.txt"
    last_group=
    for cell in "${cells[@]}"; do
        IFS=/ read -r group batch cohort <<<"$cell"
        group=${group#g}
        batch=${batch#bs}
        cohort=${cohort#cohort}
        if [[ $group != "$last_group" ]]; then
            if [[ -n $last_group ]]; then rm -rf -- "$out/g$last_group/donors"; fi
            trace=$(bash "$helpers/peer_g2_input.sh" "$work" "$group")
            mkdir -p "$out/g$group"
            "$PYTHON" -B "$workers/large_batch_input.py" "$trace" "$group" "$out/g$group/donors"
            last_group=$group
        fi
        pair="$out/$cell"
        mkdir -p "$pair"
        old="$probe_archive/$cell"
        "$PYTHON" -B "$workers/large_batch_placement.py" compare "$old/serial/thp.json" "$old/fused/thp.json"
        read -r node adapter < <("$PYTHON" -B "$workers/large_batch_placement.py" get "$old/fused/thp.json")
        set_cpu_placement "$node" "$adapter" archived
        probe_kind=cold
        if [[ $batch == 1 ]]; then probe_kind=post-pair; fi
        echo "CASE_BEGIN G$group BS$batch donor=${cohort:-all} probe=$probe_kind" >&3
        run_arm "$group" "$batch" "$trace" fused "$pair" "${cohort:-0}"
        "$PYTHON" -B "$workers/large_batch_placement.py" compare "$old/fused/thp.json" "$pair/fused/thp.json"
        "$PYTHON" -B "$workers/fixed_q4_probe_report.py" case "$pair/fused/PROBE_RESULT.json" | tee -a "$out/summary.txt" >&3
    done
    rm -rf -- "$out/g$last_group/donors"
    "$PYTHON" -B "$workers/fixed_q4_probe_report.py" summary "$out" | tee -a "$out/summary.txt" >&3
    tar -czf "$out/li-tail-probes.tar.gz" -C "$out" "${evidence[@]}"
    echo "RETURN_FILE=$out/li-tail-probes.tar.gz" >&3
    exit 0
fi

for group in "${groups[@]}"; do
    trace=$(bash "$helpers/peer_g2_input.sh" "$work" "$group")
    if (( fixed )); then
        mkdir -p "$out/g$group"
        "$PYTHON" -B "$workers/large_batch_input.py" "$trace" "$group" "$out/g$group/donors"
    fi
    for batch in "${batches[@]}"; do
        cohorts=1
        if (( fixed )); then
            archive=
            if (( batch < 16 )); then
                cohorts=$((16 / batch))
            elif (( batch > 16 )); then
                archive=$fixed_archive
            fi
        fi
        for ((cohort=0; cohort<cohorts; cohort++)); do
            pair="$out/g$group/bs$batch"
            if (( cohorts > 1 )); then pair="$pair/cohort$cohort"; fi
            run_pair "$group" "$batch" "$trace" "$cohort" "$pair"
        done
        if (( fixed && batch == 16 )); then
            for block in 1 2; do
                d="$out/pr30/block$block/g$group"
                mkdir -p "$d"
                published=${PEER_PR30_PUBLISHED_ARCHIVE:-$work/peer-baseline/pr30-feature-live-main-ten.aYyvMS}
                thp="$published/block$block/g$group/paced/thp.json"
                if [[ $peer_physical_npu == 4 && -f $thp ]] && cmp -s "$published/host.txt" "$out/host.txt" &&
                    cmp -s "$published/physical_npu.txt" "$out/physical_npu.txt" &&
                    cmp -s "$published/machine_id.txt" "$out/machine_id.txt"; then
                    read -r node adapter < <("$PYTHON" -B "$workers/large_batch_placement.py" get "$thp")
                    set_cpu_placement "$node" "$adapter" published
                fi
                packed_arms=(candidate)
                if [[ $peer_physical_npu == 3 ]]; then
                    if (( block == 1 )); then packed_arms=(pr30_reference candidate); else packed_arms=(candidate pr30_reference); fi
                fi
                for packed_arm in "${packed_arms[@]}"; do
                    echo "CASE_BEGIN G$group BS16 PR30_PROTOCOL block=$block arm=$packed_arm physical_NPU=$peer_physical_npu" >&3
                    run_arm "$group" 16 "$trace" "$packed_arm" "$d"
                done
                if [[ $peer_physical_npu == 3 ]]; then
                    "$PYTHON" -B "$workers/large_batch_placement.py" compare "$d/pr30_reference/thp.json" "$d/candidate/thp.json"
                fi
                set_cpu_placement "$planned_node" "$planned_adapter"
            done
        fi
    done
    if (( fixed )); then rm -rf -- "$out/g$group/donors"; fi
done
if (( fixed )); then
    "$PYTHON" -B "$workers/fixed_q4_summary.py" "$out" --batches "${batches[@]}" --groups "${groups[@]}" | tee -a "$out/summary.txt" >&3
    target_rc=0
    if (( ${#batches[@]} == 9 && ${#groups[@]} == 5 )); then
        "$PYTHON" -B "$workers/fixed_bs16_report.py" "$out" | tee -a "$out/summary.txt" >&3 || target_rc=$?
        if (( target_rc != 0 && target_rc != 3 )); then exit "$target_rc"; fi
    fi
fi
if [[ -n $archive && $fixed == 0 ]]; then
    "$PYTHON" -B - "$out" <<'PY' | tee -a "$out/summary.txt" >&3
from pathlib import Path
from statistics import mean
import json
import sys
root = Path(sys.argv[1])
gains, incremental_gains, trace_means = [], [], []
for group in (10, 4, 17, 7, 2):
    cases = [json.loads((root / f"g{group}/bs{bs}/fused/ARM_RESULT.json").read_text())
             for bs in (20, 24, 28, 32)]
    row = [c["vs_strong_serial_graph_gain_pct"][0] for c in cases]
    since_previous = [c["vs_previous_fused_graph_gain_pct"][0] for c in cases]
    gains.extend(row)
    incremental_gains.extend(since_previous)
    trace_means.append(mean(row))
    print(f"GROUP G{group} mean_graph_gain_vs_serial_pct={mean(row):+.2f} "
          f"mean_graph_gain_vs_previous_fused_pct={mean(since_previous):+.2f} "
          f"cases_over_20pct_vs_serial={sum(x > 20 for x in row)}/4")
print(f"ALL equal_weight_mean_graph_gain_vs_serial_pct={mean(gains):+.2f} "
      f"mean_graph_gain_vs_previous_fused_pct={mean(incremental_gains):+.2f} "
      f"trace_means_over_20pct_vs_serial={sum(x > 20 for x in trace_means)}/5 "
      f"cases_over_20pct_vs_serial={sum(x > 20 for x in gains)}/20")
PY
fi
rm -rf -- "$host_tools"
# The per-arm JSON and placement records contain the measurement. Setup and
# import transcripts in this new output directory are only needed on failure.
rm -f -- "$out"/{run,venv,pybind11,lidu-import,build-lidu,build-cpu}.log
if (( fixed )); then
    echo "REPLAY_COMPLETE cells=$((${#groups[@]} * ${#batches[@]})) RESULT_ROOT=$out" >&3
    trap - EXIT
    exit "$target_rc"
fi
echo "REPLAY_COMPLETE pairs=20 RESULT_ROOT=$out" >&3
