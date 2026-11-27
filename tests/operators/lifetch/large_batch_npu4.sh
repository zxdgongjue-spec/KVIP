#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Five real Q4 traces paired with the strong two-worker serial path.
set -eo pipefail
controller_root=$(cd "$(dirname "$0")/../../.." && pwd -P)
root=$(cd "${PEER_TIMED_SOURCE:-$controller_root}" && pwd -P)
work=${PEER_G2_WORK:-/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold}
reuse_serial=
archive=
fixed=0
fixed_archive=
packed=0
standalone=0
packed_archive=
ctl=
packed_candidate=0
large_cold_windows=${PEER_LARGE_COLD_WINDOWS:-2}
if [[ $large_cold_windows != 2 && $large_cold_windows != 3 ]]; then
    echo "PEER_LARGE_COLD_WINDOWS must be 2 or 3" >&2; exit 2
fi
four_ranks=${PEER_FOUR_WINDOW_RANKS:-0}
bs32_li=${PEER_BS32_LI:-}
if [[ -n $bs32_li && $bs32_li != split-role && $bs32_li != shared-final ]]; then
    echo "Unknown PEER_BS32_LI=$bs32_li" >&2; exit 2
fi
batches=(20 24 28 32)
groups=(10 4 17 7 2)
if (( $# )); then
    if [[ $# == 2 && $1 == --candidate-only ]]; then
        archive=$(cd "$2" && pwd)
    elif [[ $# == 2 && $1 == --reuse-g10-bs20-serial ]]; then
        reuse_serial=$2
    elif [[ $1 == --four-ranks-candidate-only && $# == 1 || $1 == --fixed-q4 && $# -le 2 || ( $1 == --fixed-q4-packed || $1 == --packed-aiv ) && $# == 2 ]]; then
        fixed=1
        default_batches="1 2 4 8 16 20 24 28 32"
        if [[ $1 == --four-ranks-candidate-only ]]; then
            packed=1
            standalone=1
            packed_candidate=1
            four_ranks=1
            large_cold_windows=3
            default_batches="32 28 24 20"
        elif [[ $1 == --fixed-q4-packed || $1 == --packed-aiv ]]; then
            packed=1
            if [[ $1 == --packed-aiv ]]; then packed_candidate=1; fi
            default_batches="16 32 1 2 4 8 20 24 28"
            if (( packed_candidate )); then default_batches="32 28 24 20 16 8 4 2 1"; fi
            packed_archive=$(cd "$2" && pwd)
        elif (( $# == 2 )); then
            fixed_archive=$2
        fi
        read -ra batches <<<"${FIXED_Q4_BATCHES:-$default_batches}"
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
        echo "Usage: $0 [--four-ranks-candidate-only | --packed-aiv FIXED_ARCHIVE | --fixed-q4-packed FIXED_ARCHIVE | --fixed-q4 [LARGE_ARCHIVE] | --candidate-only RESULT_ROOT | --reuse-g10-bs20-serial RESULT_ROOT]" >&2
        exit 2
    fi
fi
packed_blocks=2
if (( packed_candidate )); then
    if (( standalone )); then packed_blocks=1; else packed_blocks=${PEER_PACKED_BLOCKS:-2}; fi
    case $packed_blocks in 1|2) ;; *) echo "Expected PEER_PACKED_BLOCKS=1 or 2" >&2; exit 2 ;; esac
fi
if [[ -n $bs32_li ]]; then
    if (( !packed_candidate )); then echo "$bs32_li requires --packed-aiv" >&2; exit 2; fi
    if [[ ! -s $root/csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer_large_dualrole.cpp ]]; then
        echo "Missing BS32 split-role device implementation" >&2; exit 2
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
area="$work/peer-baseline"
sources="$work/fused-scan-sources"
mkdir -p "$area" "$sources"
prefix=largebs-q4
if (( fixed )); then prefix=fixed-q4; fi
if [[ -n $packed_archive ]]; then prefix=fixed-q4-packed; fi
if (( packed_candidate )); then prefix="fixed-q4-aiv-${large_cold_windows}cold"; fi
carry_credit=${PEER_FOUR_WINDOW_CARRY:-0}
if [[ $carry_credit != 0 ]]; then
    [[ $carry_credit == 1 && $packed_candidate == 1 && $large_cold_windows == 3 && -z $bs32_li ]] ||
        { echo "Four-window carry requires the packed regular four-window candidate" >&2; exit 2; }
    for batch in "${batches[@]}"; do (( batch > 16 )) || { echo "Four-window carry requires BS>16" >&2; exit 2; }; done
    prefix=fixed-q4-four-window-carry
fi
if [[ $four_ranks != 0 ]]; then
    [[ $four_ranks == 1 && $packed_candidate == 1 && $large_cold_windows == 3 && $carry_credit == 0 && -z $bs32_li ]] ||
        { echo "Four ranks requires the fixed, non-carry four-window candidate" >&2; exit 2; }
    for batch in "${batches[@]}"; do (( batch > 16 )) || { echo "Four ranks requires BS>16" >&2; exit 2; }; done
    prefix=fixed-q4-four-ranks
fi
if [[ -n $bs32_li ]]; then prefix="$prefix-bs32-$bs32_li"; fi
if (( packed_blocks == 1 )); then prefix="$prefix-one-block"; fi
sdk_rev=813f1750a50b7e6c92d5f5397f2fbc05befef6da
sdk_tree=1de18e8315137d7a0227c3ae4c82afd45b5fe743
follow_ref=${PEER_FOUR_RANK_FOLLOWUP:-}
if [[ -n $follow_ref ]]; then
    (( standalone && four_ranks )) && [[ $peer_physical_npu == 4 && -z ${PEER_RANK_DIAG_SOURCE:-} ]] ||
        { echo 'Follow-up requires the NPU4 four-rank entry' >&2; exit 2; }
    prefix=fixed-q4-rank-followup
fi
audit_mode=${PEER_FOUR_RANK_AUDIT:-}
if [[ -n $audit_mode ]]; then
    case $audit_mode in serial|journal|frozen|team-schedule|two-row-natural|two-row-frozen|window-x[2348]-linear|window-x[2348]-early) ;; *) echo "Unknown audit mode: $audit_mode" >&2; exit 2 ;; esac
    (( standalone )) &&
        [[ -n ${PEER_AUDIT_REFERENCE:-} && -z $follow_ref && -z ${PEER_PACKED_RESUME:-} && $peer_physical_npu == 4 ]] ||
        { echo "Invalid four-rank audit configuration" >&2; exit 2; }
    prefix="fixed-q4-audit-$audit_mode"
fi
diag_source=${PEER_RANK_DIAG_SOURCE:-}
if [[ -n $diag_source ]]; then
    (( standalone )) && [[ $peer_physical_npu == 4 && -z ${PEER_PACKED_RESUME:-} ]] ||
        { echo 'Single-call diagnostics require the standalone NPU4 candidate' >&2; exit 2; }
    prefix=four-rank-diagnostic
fi
native=$(git -C "$root" rev-parse HEAD)
resuming=0
if [[ -n ${PEER_PACKED_RESUME:-} ]]; then
    (( packed )) || { echo "Resume needs a packed measurement" >&2; exit 2; }
    resuming=1
    out=$(cd "$PEER_PACKED_RESUME" && pwd -P)
    [[ ${out%/*} == "$(cd "$area" && pwd -P)" && ${out##*/} == "$prefix."* &&
       -f $out/packed_source.txt && $(<"$out/packed_source.txt") == "$packed_archive" &&
       $(<"$out/native.txt") == "$native" && $(<"$out/physical_npu.txt") == "$peer_physical_npu" &&
       $(<"$out/host.txt") == "$(uname -n)" &&
       $(<"$out/kvaf.txt") == "$sdk_rev" && $(<"$out/kvaf_tree.txt") == "$sdk_tree" ]] ||
        { echo "Resume source, host, card or SDK mismatch: $out" >&2; exit 2; }
    if [[ -n $follow_ref && ( ! -s $out/followup_reference.txt || $(<"$out/followup_reference.txt") != "$follow_ref" ) ]]; then
        echo "Interrupted candidate used another comparison archive: $out" >&2
        exit 2
    fi
    if [[ -s $out/machine_id.txt && -s /etc/machine-id ]]; then
        cmp -s "$out/machine_id.txt" /etc/machine-id || { echo "Resume machine ID mismatch: $out" >&2; exit 2; }
    elif [[ ! -s $out/boot_id.txt || $(<"$out/boot_id.txt") != "$(</proc/sys/kernel/random/boot_id)" ]]; then
        echo "Cannot identify resume host by machine or boot ID: $out" >&2; exit 2
    fi
    git -C "$root" diff --quiet HEAD -- || { echo "Tracked changes in timed worktree: $root" >&2; exit 2; }
    if (( standalone )); then
        [[ $(<"$out/selection_oracle.txt") == captured_logical_scores && ! -e $out/reference.txt ]] ||
            { echo "Resume oracle mismatch: $out" >&2; exit 2; }
    fi
    if (( packed_candidate )); then
        saved_blocks=2
        if [[ -f $out/candidate_blocks.txt ]]; then saved_blocks=$(<"$out/candidate_blocks.txt"); fi
        [[ $saved_blocks == "$packed_blocks" ]] || { echo "Candidate block count differs from archived run" >&2; exit 2; }
    fi
else
    out=$(mktemp -d "$area/$prefix.XXXXXX")
fi
echo "RESULT_ROOT=$out"
log="$out/run.log"
# A single transcript for setup/driver errors, with compact results on stdout.
exec 3>&1
exec >>"$log" 2>&1
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
sdk="$sources/unified-sdk-813f1750-d6cfd392"
workers="$root/tests/operators/lifetch/support"
# Protect this run against a second invocation using the same physical card.
card_lock="$sources/largebs-$(</proc/sys/kernel/random/boot_id)-npu$peer_physical_npu.lock"
exec 9>"$card_lock"
if (( packed_candidate )); then
    if ! flock -n 9; then
        echo "WAITING_FOR_CARD physical_NPU=$peer_physical_npu lock=$card_lock" >&3
        if command -v lslocks >/dev/null; then lslocks -n -o PID,COMMAND,PATH | grep -F -- "$card_lock" >&3 || true; fi
        flock 9
    fi
else
    flock -n 9 || { echo "A large-batch replay is already running on NPU$peer_physical_npu"; exit 1; }
fi
host_tools="$out/host-tools"
mkdir -p "$host_tools"
if ! git -C "$root" cat-file -e "$support_rev^{commit}" 2>/dev/null; then
    git -C "$root" fetch -q --no-tags https://gitcode.com/CacheOS/KVIP.git refs/heads/test/lifetch-pr30-fast-accounting
fi
git -C "$root" archive "$support_rev" tests/operators/lifetch/{peer_g2_input.sh,support/{workloads.py,peer_thp.py,query_owner_input.py,peer_node_prepare.sh}} | tar -x -C "$host_tools"
helpers="$host_tools/tests/operators/lifetch"
source "$helpers/support/peer_node_prepare.sh"
exec 8>"$sdk.lock"
flock 8
baseline="$work/adaptive-sources/peer-serial-da318bdb"
clean="$area/KVAF-de4fd37e"
if (( standalone )) && [[ $audit_mode != serial ]]; then
    peer_prepare_node "$root" "$baseline" "$clean" "$work" "$out" fused-only
else
    peer_prepare_node "$root" "$baseline" "$clean" "$work" "$out"
fi
if (( !resuming )); then uname -n > "$out/host.txt"; fi
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
[[ $actual == "$sdk_tree" && $(git -C "$sdk" rev-parse HEAD) == "$sdk_rev" ]] || { echo "Unexpected KVAF source tree $actual"; exit 1; }
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
if (( !standalone )) || [[ $audit_mode == serial ]]; then
    lidu="$baseline/csrc/operators/lidu"
    vendors=("$lidu"/_custom_opp/vendors/*/op_api/lib/libcust_opapi.so)
    [[ ${#vendors[@]} == 1 && -s ${vendors[0]} ]]
    export ASCEND_CUSTOM_OPP_PATH="${vendors[0]%/op_api/lib/libcust_opapi.so}${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
    export PYTHONPATH="$lidu/torch_extension${PYTHONPATH:+:$PYTHONPATH}"
fi
export PYTHONPATH="$root/python:$sdk/build-npu/python${PYTHONPATH:+:$PYTHONPATH}"
export KVAF_BATCH_GET_KERNEL="$sdk/build-npu/bin/kvaf_execute_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$sdk/build-npu/bin/kvaf_execute_batch_put_kernels.o"
cpu="$sdk/build-cpu/runtime/kvaf_cpu_service"
if [[ $ASCEND_RT_VISIBLE_DEVICES != "$peer_physical_npu" || $PEER_PHYSICAL_NPU != "$peer_physical_npu" ]]; then
    echo "Dependency setup changed the selected physical card" >&2
    exit 1
fi
if (( !resuming )); then
    printf '%s\n' "$native" > "$out/native.txt"
    if [[ -r /etc/machine-id ]]; then cat /etc/machine-id > "$out/machine_id.txt"; fi
    printf '%s\n' "$peer_physical_npu" > "$out/physical_npu.txt"
    cat /proc/sys/kernel/random/boot_id > "$out/boot_id.txt"
    printf '%s\n' "$actual" > "$out/kvaf_tree.txt"
    printf '%s\n' "$sdk_rev" > "$out/kvaf.txt"
fi
if [[ -n $audit_mode ]]; then
    git -C "$controller_root" rev-parse HEAD > "$out/audit_reader.txt"
    printf '%s\n' "$PEER_AUDIT_REFERENCE" > "$out/audit_reference.txt"
    echo "AUDIT physical_NPU=4 mode=$audit_mode source=$PEER_AUDIT_REFERENCE" >&3
elif [[ -n $diag_source ]]; then
    echo "CONFIG physical_NPU=$peer_physical_npu mode=$PEER_RANK_DIAG_MODE source=$diag_source timed_calls=0 diagnostic_calls_per_cell=1" >&3
elif (( packed )); then
    if (( !resuming )); then printf '%s\n' "$packed_archive" > "$out/packed_source.txt"; fi
    if (( standalone )); then
        if (( !resuming )); then
            printf '1\n' > "$out/candidate_blocks.txt"
            printf 'captured_logical_scores\n' > "$out/selection_oracle.txt"
        fi
        if [[ -n $follow_ref ]]; then
            printf '%s\n' "${PEER_FOUR_RANK_NEW_DEVICE_BUILD:?}" > "$out/followup_new_device_build.txt"
            printf '%s\n' "${PEER_FOUR_RANK_SAME_PACING:?}" > "$out/followup_same_pacing.txt"
            follow_report="$controller_root/tests/operators/lifetch/support/four_rank_followup.py"
            "$PYTHON" -B "$follow_report" verify "$out" "$follow_ref"
            echo "FOLLOWUP source=$follow_ref candidate=$native scope=$PEER_FOUR_RANK_SCOPE fresh_arms=fused archived_baseline=1" >&3
        else
            echo "CANDIDATE_ONLY oracle=captured_logical_scores fresh_arms=fused comparisons=unavailable" >&3
        fi
    elif (( packed_candidate )); then
        printf '%s\n' "$packed_blocks" > "$out/candidate_blocks.txt"
        ctl=$("$PYTHON" -B "$workers/fixed_q4_packed_report.py" reference "$out" "$packed_archive" "$root" "${PEER_PACKED_REFERENCE:-}")
        if [[ -f $out/reference.txt && $(<"$out/reference.txt") != "$ctl" ]]; then
            echo "Cannot resume against a different packed reference" >&2; exit 1
        fi
        printf '%s\n' "$ctl" > "$out/reference.txt"
        printf 'PACKED_REFERENCE=%s fresh_arms=fused target_pct=15\n' "$ctl" >&3
    else
        "$PYTHON" -B "$workers/fixed_q4_packed_report.py" verify-source "$out" "$packed_archive" "$root"
    fi
    if (( carry_credit )); then
        echo "WINDOW_SCHEDULE four-window-carry earned_per_row=256/512/768/1024 independent_private_CQ=1" >&3
    fi
    if (( packed_candidate && large_cold_windows == 2 )); then
        large_batches=20/24/28/32
        if [[ -n $bs32_li ]]; then large_batches=20/24/28; fi
        echo "WINDOW_SCHEDULE BS$large_batches first=256 cold=384,384 row_limit=1024" >&3
    fi
    if [[ -n $bs32_li ]]; then
        echo "WINDOW_SCHEDULE BS32 policy=$bs32_li first=256 cold=256,256,256 row_limit=1024" >&3
    fi
    echo "CONFIG packed_calls=10 blocks=$packed_blocks groups=${groups[*]} batches=${batches[*]} physical_NPU=$peer_physical_npu archive=$packed_archive" >&3
elif (( fixed )); then
    args=("$out")
    if [[ -n $fixed_archive ]]; then args+=("$fixed_archive"); fi
    fixed_archive=$("$PYTHON" -B "$workers/large_batch_placement.py" archive "${args[@]}")
    printf 'CONFIG physical_NPU=%s logical_NPU=0 large_serial=%s\n' "$peer_physical_npu" "${fixed_archive:-fresh_same_card}" >&3
fi
# Use the same registered CPU memory placement for the serial and fused arms.
placement="auto auto"
if [[ -n $audit_mode ]]; then
    placement=$("$PYTHON" -B "$controller_root/tests/operators/lifetch/support/large_batch_placement.py" get "$PEER_AUDIT_REFERENCE/g7/bs32/block1/fused/thp.json")
elif [[ -n $follow_ref ]]; then
    placement=$("$PYTHON" -B "$workers/large_batch_placement.py" get "$follow_ref/g7/bs32/block1/fused/thp.json")
elif [[ -z $archive && -z $fixed_archive && -z $packed_archive ]]; then
    placement=$("$PYTHON" -B "$workers/large_batch_placement.py" find "$out")
fi
trace_paths=()
if (( standalone && resuming )) && [[ -n $follow_ref ]]; then
    "$PYTHON" -B "$controller_root/tests/operators/lifetch/support/four_rank_accept.py" validate-partial \
        "$out" "$follow_ref" "$native" >&3
elif (( standalone && resuming )); then
    verified=$("$PYTHON" -B "$controller_root/tests/operators/lifetch/support/fixed_q4_resume.py" \
        "$out" "$root" "$helpers" "$work" "$KVAF_BATCH_GET_KERNEL" 2>&3)
    while IFS=$'\t' read -r key value; do
        if [[ $key == placement ]]; then placement=$value; elif [[ -n $key ]]; then trace_paths[$key]=$value; fi
    done <<<"$verified"
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
    "${command[@]}" --listen 127.0.0.1 --port "$port" --memory-page-mode thp 9>&- >"$dir/cpu.log" 2>&1 &
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
    "$PYTHON" -B "$helpers/support/peer_thp.py" --unified-sdk --pid "$server" --log "$dir/cpu.log" --output "$dir/thp.json" --timeout 7200 9>&- &
    monitor=$!
    if [[ -n $audit_mode ]]; then
        local tool="$controller_root/tests/operators/lifetch/support/four_rank_accept_worker.py"
        (cd "$dir" && timeout --kill-after=30s 7200s "$PYTHON" -B "$tool" "$audit_mode" "$root"             "$PEER_AUDIT_REFERENCE" "$group" "$batch" "$trace" "$sdk" "127.0.0.1:$port" "$prepared_bank" "$dir") >"$dir/worker.log" 2>&1 || {
            rc=$?; tail -25 "$dir/worker.log"; exit "$rc"
        }
        wait "$monitor"; monitor=
        return
    fi
    if [[ -n $diag_source ]]; then
        local tool="$controller_root/tests/operators/lifetch/support/four_rank_diagnostic.py"
        (cd "$dir" && timeout --kill-after=30s 7200s "$PYTHON" -B "$tool" "$root" "$diag_source" "$group" "$batch" "$trace" "$sdk" "127.0.0.1:$port" "$prepared_bank" "$dir") >"$dir/worker.log" 2>&1 || {
            rc=$?; tail -25 "$dir/worker.log"; exit "$rc"
        }
        wait "$monitor"
        monitor=
        return
    fi
    if [[ $arm == candidate || $arm == pr30_reference ]]; then
        (cd "$dir" && timeout --kill-after=30s 7200s bash "$root/tests/operators/lifetch/fixed_bs16_pr30.sh" \
            "$root" "$work" "$out" "$group" "$trace" "$sdk" "127.0.0.1:$port" "$dir" "$arm") >"$dir/worker.log" 2>&1 || { tail -25 "$dir/worker.log"; exit 1; }
        wait "$monitor"
        monitor=
        return
    fi
    args=(--trace "$trace" --group "$group" --batch "$batch" --cohort "$cohort" --arm "$arm" --kvaf-root "$sdk"
          --kvaf-server "127.0.0.1:$port" --output "$dir")
    if (( fixed )); then args+=(--prepared "$prepared_bank"); fi
    if (( packed )); then
        args+=(--packed-ten)
        if (( standalone )); then args+=(--captured-score-oracle); else args+=(--serial-result "$packed_oracle"); fi
        if (( packed_candidate )); then args+=(--compare-packed-candidate); fi
        if (( carry_credit )); then args+=(--carry-credit); fi
        if (( packed_candidate && (batch == 32 || four_ranks) )) && [[ ${pair##*/} == block1 ]]; then args+=(--window-density); fi
        if (( four_ranks )); then
            args+=(--large-window-policy four-ranks)
        elif (( packed_candidate && batch == 32 )) && [[ -n $bs32_li ]]; then
            args+=(--large-window-policy "$bs32_li")
        elif (( packed_candidate && batch > 16 && large_cold_windows == 2 )); then
            args+=(--large-two-cold)
        fi
    elif [[ $arm == fused ]]; then
        if [[ -n $archive ]]; then
            args+=(--serial-result "$archive/g$group/bs$batch/serial" --compare-archived-pair)
        else
            args+=(--serial-result "$pair/serial")
        fi
    fi
    (cd "$dir" && timeout --kill-after=30s 7200s "$PYTHON" -B "$workers/large_batch_pair.py" "${args[@]}") >"$dir/worker.log" 2>&1 || {
        worker_rc=$?
        printf 'WORKER_FAILED G%s BS%s arm=%s rc=%s worker_log=%s\n' "$group" "$batch" "$arm" "$worker_rc" "$dir/worker.log" >&3
        tail -20 "$dir/worker.log"
        exit "$worker_rc"
    }
    wait "$monitor"
    monitor=
)
if [[ -n $audit_mode ]]; then
    source "$controller_root/tests/operators/lifetch/support/four_rank_accept_loop.sh"
    exit 0
fi
if [[ -n $diag_source ]]; then
    count=0
    for group in 7 2 10; do
        case $group:${PEER_RANK_DIAG_MODE:?} in 7:li|2:tail|10:tail|2:prefetch-ready|10:prefetch-ready) continue ;; esac
        trace=
        mkdir -p "$out/g$group"
        prepared_bank="$out/g$group/donors"
        shapes=(32)
        if [[ $group == 7 ]]; then shapes+=(28); fi
        if [[ $PEER_RANK_DIAG_MODE == prefetch-ready ]]; then shapes=(28); fi
        for batch in "${shapes[@]}"; do
            pair="$out/g$group/bs$batch"
            mkdir -p "$pair"
            ref="$diag_source/g$group/bs$batch/block1/fused"
            [[ -f $ref/placement_verified && -s $ref/packed_result.json ]]
            cached=$("$PYTHON" -B "$controller_root/tests/operators/lifetch/support/four_rank_diagnostic.py" \
                --cached "$diag_source" "$group" "$batch" "$controller_root")
            if [[ -n $cached ]]; then
                printf '%s\n' "$cached" > "$pair/diagnostic_source.txt"
                echo "DIAGNOSTIC_REUSED G$group BS$batch source=$cached" >&3
                grep -E "^(SINGLE_DIAGNOSTIC |BATCH_CLOCK |LAST_TEAM |LI_COUNTER|PREFETCH_CQ )" "$cached/worker.log" | tee -a "$out/summary.txt" >&3
                count=$((count+1))
                continue
            fi
            if [[ -z $trace ]]; then
                trace=$(bash "$helpers/peer_g2_input.sh" "$work" "$group")
                "$PYTHON" -B "$controller_root/tests/operators/lifetch/support/large_batch_input.py" "$trace" "$group" "$prepared_bank"
            fi
            placement=$("$PYTHON" -B "$workers/large_batch_placement.py" get "$ref/thp.json")
            read -r planned_node planned_adapter <<<"$placement"
            set_cpu_placement "$planned_node" "$planned_adapter" archived
            printf 'DIAGNOSTIC_BEGIN G%s BS%s mode=%s\n' "$group" "$batch" "$PEER_RANK_DIAG_MODE" >&3
            run_arm "$group" "$batch" "$trace" fused "$pair"
            "$PYTHON" -B "$workers/large_batch_placement.py" compare "$ref/thp.json" "$pair/fused/thp.json"
            touch "$pair/fused/placement_verified"
            printf '%s\n' "$pair/fused" > "$pair/diagnostic_source.txt"
            grep -E "^(SINGLE_DIAGNOSTIC |BATCH_CLOCK |LAST_TEAM |LI_COUNTER|PREFETCH_CQ )" "$pair/fused/worker.log" | tee -a "$out/summary.txt" >&3
            count=$((count+1))
        done
        rm -rf -- "$out/g$group/donors"
    done
    "$PYTHON" -B "$controller_root/tests/operators/lifetch/support/four_rank_diagnostic.py" --package "$out" >&3
    echo "DIAGNOSTICS_COMPLETE cells=$count RESULT_ROOT=$out" >&3
    exit 0
fi
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

run_packed_cohort() {
    local group=$1 batch=$2 trace=$3 cohort=$4 pair=$5
    local block arm ref dir backup packed_oracle placement node adapter result
    local -a order
    if [[ -n $follow_ref ]]; then
        ref="$follow_ref/g$group/bs$batch/block1/fused/thp.json"
        placement=$("$PYTHON" -B "$workers/large_batch_placement.py" get "$ref")
        read -r planned_node planned_adapter <<<"$placement"
        set_cpu_placement "$planned_node" "$planned_adapter" archived
    elif (( standalone )); then
        ref=
        set_cpu_placement "$planned_node" "$planned_adapter"
    else
        packed_oracle=$("$PYTHON" -B "$workers/fixed_q4_packed_archive.py" oracle-dir "$packed_archive" "$group" "$batch" "$cohort")
        ref="$packed_oracle/thp.json"
        placement=$("$PYTHON" -B "$workers/large_batch_placement.py" get "$ref")
        read -r node adapter <<<"$placement"
        set_cpu_placement "$node" "$adapter" archived
    fi
    for ((block=1; block<=packed_blocks; block++)); do
        dir="$pair/block$block"
        mkdir -p "$dir"
        order=(serial fused)
        if (( block == 2 )); then order=(fused serial); fi
        if (( packed_candidate )); then order=(fused); fi
        for arm in "${order[@]}"; do
            result="$dir/$arm"
            if [[ -f $result/placement_verified && -s $result/packed_result.json ]]; then
                if (( standalone )); then
                    (( resuming )) || { echo "Unexpected existing cell: $result" >&2; return 1; }
                    echo "REUSED G$group BS$batch block=$block arm=$arm" >&3
                else
                    "$PYTHON" -B "$workers/large_batch_placement.py" compare "$ref" "$result/thp.json"
                fi
                continue
            fi
            if [[ -e $result ]]; then
                backup=$(mktemp -d "$dir/incomplete.XXXXXX")
                mv -- "$result" "$backup/"
                echo "PARTIAL_SAVED G$group BS$batch path=$backup/$arm" >&3
            fi
            printf 'CASE_BEGIN G%s BS%s cohort=%s block=%s arm=%s\n' "$group" "$batch" "$cohort" "$block" "$arm" >&3
            run_arm "$group" "$batch" "$trace" "$arm" "$dir" "$cohort"
            [[ -s $result/packed_result.json ]]
            if (( packed_candidate && (batch == 32 || four_ranks) && block == 1 )); then
                grep ' WINDOW_DIAG ' "$result/worker.log" | tee -a "$out/summary.txt" >&3
                if [[ -n $bs32_li ]]; then
                    grep ' SPLIT_ROLE_DIAG ' "$result/worker.log" | tee -a "$out/summary.txt" >&3
                    if [[ $bs32_li == shared-final ]]; then
                        grep ' FINAL_PAIR_DIAG ' "$result/worker.log" | tee -a "$out/summary.txt" >&3
                    fi
                fi
            fi
            if (( carry_credit )); then
                grep '^CREDIT ' "$result/worker.log" | tee -a "$out/summary.txt" >&3
            fi
            if (( four_ranks || (packed_candidate && batch > 16 && large_cold_windows == 2) )) ||
                { (( packed_candidate && batch == 32 )) && [[ -n $bs32_li ]]; }; then
                grep '^WINDOWS ' "$result/worker.log" | tee -a "$out/summary.txt" >&3
            fi
            if [[ -n $follow_ref ]]; then
                "$PYTHON" -B "$workers/large_batch_placement.py" compare "$ref" "$result/thp.json"
            elif (( standalone )); then
                placement=$("$PYTHON" -B "$workers/large_batch_placement.py" get "$result/thp.json")
                read -r node adapter <<<"$placement"
                [[ $planned_node == auto || $planned_node == "$node" ]] &&
                    [[ $planned_adapter == auto || $planned_adapter == "$adapter" ]] ||
                    { echo "Unexpected measured NUMA/UB: $placement, requested: $planned_node $planned_adapter" >&2; return 1; }
                planned_node=$node
                planned_adapter=$adapter
            else
                "$PYTHON" -B "$workers/large_batch_placement.py" compare "$ref" "$result/thp.json"
            fi
            touch "$result/placement_verified"
        done
    done
}

run_group_batches() {
    local group=$1 batch trace
    shift
    if [[ -z ${trace_paths[$group]:-} ]]; then
        trace_paths[$group]=$(bash "$helpers/peer_g2_input.sh" "$work" "$group")
    fi
    trace=${trace_paths[$group]}
    if (( fixed )); then
        mkdir -p "$out/g$group"
        prepared_bank="$out/g$group/donors"
        if [[ -n $follow_ref && -s $follow_ref/g$group/donors/source.json ]]; then
            prepared_bank="$follow_ref/g$group/donors"
        elif (( packed_candidate )) && [[ -n $ctl && -s $ctl/g$group/donors/source.json ]]; then
            prepared_bank="$ctl/g$group/donors"
        elif (( standalone && resuming )) && [[ -f $out/g$group/bs$1/block1/fused/placement_verified ]]; then
            # Completed cells had their input fingerprints checked before any worker started.
            :
        elif [[ ! -f $prepared_bank/source.json ]]; then
            if [[ -e $prepared_bank ]]; then
                donor_backup=$(mktemp -d "$out/g$group/incomplete-donors.XXXXXX")
                mv -- "$prepared_bank" "$donor_backup/"
            fi
            donor_stage=$(mktemp -d "$out/g$group/donors-prepare.XXXXXX")
            "$PYTHON" -B "$controller_root/tests/operators/lifetch/support/large_batch_input.py" "$trace" "$group" "$donor_stage/bank"
            mv "$donor_stage/bank" "$prepared_bank"
            rmdir "$donor_stage"
        fi
    fi
    for batch in "$@"; do
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
            if (( packed )); then
                run_packed_cohort "$group" "$batch" "$trace" "$cohort" "$pair"
            else
                run_pair "$group" "$batch" "$trace" "$cohort" "$pair"
            fi
        done
        if [[ -n $follow_ref ]]; then
            "$PYTHON" -B "$follow_report" cell "$out" "$follow_ref" "$group" "$batch" | tee -a "$out/summary.txt" >&3
        elif (( standalone )); then
            "$PYTHON" -B "$workers/fixed_q4_candidate_only.py" cell "$out" "$group" "$batch" | tee -a "$out/summary.txt" >&3
        elif (( packed )); then
            "$PYTHON" -B "$workers/fixed_q4_packed_report.py" cell "$out" "$group" "$batch" | tee -a "$out/summary.txt" >&3
            if (( four_ranks )); then
                "$PYTHON" -B "$workers/fixed_q4_window_ranks.py" "$out" "$group" "$batch" | tee -a "$out/summary.txt" >&3
            fi
        elif (( fixed && batch == 16 )); then
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
    if (( fixed )) && [[ $batch == "${batches[${#batches[@]} - 1]}" ]]; then
        rm -rf -- "$out/g$group/donors"
    fi
}
if [[ -n $follow_ref ]]; then
    for group in "${groups[@]}"; do
        if [[ $PEER_FOUR_RANK_SCOPE == focus && $group != 7 ]]; then
            run_group_batches "$group" 32
        else
            run_group_batches "$group" "${batches[@]}"
        fi
        rm -rf -- "$out/g$group/donors"
    done
    "$PYTHON" -B "$follow_report" summary "$out" "$follow_ref" "$PEER_FOUR_RANK_SCOPE" >&3
    rm -rf -- "$host_tools"
    echo "REPLAY_COMPLETE RESULT_ROOT=$out" >&3
    exit 0
elif (( packed_candidate )); then
    # Finish each large shape for all five traces before entering smaller
    # shapes, which have many more donor cohorts. Keep each donor bank once.
    for batch in "${batches[@]}"; do
        for group in "${groups[@]}"; do run_group_batches "$group" "$batch"; done
    done
else
    for group in "${groups[@]}"; do run_group_batches "$group" "${batches[@]}"; done
fi
if (( standalone )); then
    "$PYTHON" -B "$workers/fixed_q4_candidate_only.py" summary "$out" --groups "${groups[@]}" --batches "${batches[@]}" | tee -a "$out/summary.txt" >&3
    target_rc=0
elif (( packed )); then
    "$PYTHON" -B "$workers/fixed_q4_packed_report.py" summary "$out" --groups "${groups[@]}" --batches "${batches[@]}" | tee -a "$out/summary.txt" >&3
    target_rc=0
elif (( fixed )); then
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
if (( !resuming )); then rm -f -- "$out"/{run,venv,pybind11,lidu-import,build-lidu,build-cpu}.log; fi
if (( fixed )); then
    echo "REPLAY_COMPLETE cells=$((${#groups[@]} * ${#batches[@]})) RESULT_ROOT=$out" >&3
    trap - EXIT
    exit "$target_rc"
fi
echo "REPLAY_COMPLETE pairs=20 RESULT_ROOT=$out" >&3
