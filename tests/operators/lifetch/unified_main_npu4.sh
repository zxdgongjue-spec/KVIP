#!/usr/bin/env bash
# Five captured BS16/Q4 groups on one physical NPU4 with the unified KVAF SDK.
set -eo pipefail
if (( $# < 3 || $# > 5 )) || [[ ! $2 =~ ^[a-f0-9]{40}$ || ! $3 =~ ^[a-f0-9]{40}$ ||
    ( $# == 4 && $4 != cap-pair ) || ( $# == 5 && $4 != active-history && $4 != pr30-parity && $4 != near512 && $4 != near512-five && $4 != near512-locality && $4 != pr30-feature-steps && $4 != pr30-feature-live-main && $4 != pr-head-parity ) ]]; then
    echo "Usage: bash unified_main_npu4.sh KVIP_REPO READER_SHA NATIVE_SHA [cap-pair | active-history ARCHIVED_CAP_PAIR | pr30-parity ARCHIVED_ACTIVE_HISTORY | {near512,near512-five} ARCHIVED_PR30_PARITY | near512-locality ARCHIVED_ACTIVE_HISTORY | {pr30-feature-steps,pr30-feature-live-main} {ARCHIVED_UNIFIED_MAIN,auto-local} | pr-head-parity auto-local]" >&2
    exit 2
fi
repo=$1
reader=$2
main=$3
mode=${4:-single}
live_feature_main=0
head_parity=0
if [[ $mode == pr-head-parity ]]; then
    head_parity=1
    live_feature_main=1
    mode=pr30-feature-steps
elif [[ $mode == pr30-feature-live-main ]]; then
    live_feature_main=1
    mode=pr30-feature-steps
fi
upstream=d6cfd392f722dd3f96ef6e05175f503aac511dbb
name=unified_main
prefix=unified-main-five
auto_feature_reference=0
feature_prior=
if [[ $main != "$upstream" && $head_parity == 0 ]]; then
    git -C "$repo" merge-base --is-ancestor 54acdb9c54853441d08898a2623407f4492a83a4 "$main" || {
        echo "Expected a native build containing the window-rank and paced-GET PR" >&2
        exit 2
    }
    name=long_history_candidate
    prefix=long-history-five
fi
if (( head_parity )); then
    pr30=4bd5d90eaffe9db71ffc64778f9a0cf81bc2e872
    pr31=58155892a4f3b7a3bf9789d6055fa4a0f1e95c07
    pr31_table=cfe2b52f6403d5edd612c314bc7110525a5f5789
    if [[ $main != "$pr30" || ${5:-} != auto-local ]]; then
        echo 'Expected the pinned PR30 head and auto-local for the 128K PR comparison' >&2
        exit 2
    fi
    prefix=pr-head-cap128-ten
    auto_feature_reference=1
    archive=auto-local
elif [[ $mode == pr30-feature-steps ]]; then
    dualcut=bba4372065e549d0f2291bc08458d6250d4fa161
    paced=b261b4885edd79dabac822a2e6e3790c7f226ad6
    if [[ $5 == auto-local ]]; then
        auto_feature_reference=1
        archive=auto-local
    else
        archive=$(cd "$5" && pwd)
        if [[ $(<"$archive/main.txt") != "$upstream" || $(<"$archive/physical_npu.txt") != 4 ]]; then
            echo "Expected a matching mainline archive on physical NPU4: $archive" >&2
            exit 2
        fi
    fi
    if [[ $main != "$upstream" ||
          $(git -C "$repo" rev-parse "$dualcut^") != "$main" ||
          $(git -C "$repo" rev-parse "$paced^") != "$dualcut" ]]; then
        echo "Expected the NPU4 main archive and the matching main -> dualcut -> paced experiment commits" >&2
        exit 2
    fi
    prefix=pr30-feature-fast
    if (( live_feature_main )); then prefix=pr30-feature-live-main-ten; fi
    if [[ -n ${PEER_FEATURE_PRIOR:-} ]]; then feature_prior=$(cd "$PEER_FEATURE_PRIOR" && pwd -P); fi
elif [[ $mode == cap-pair ]]; then
    if [[ $main != cfe2b52f6403d5edd612c314bc7110525a5f5789 ]]; then
        echo "The capacity comparison requires the candidate with 512K support" >&2
        exit 2
    fi
    prefix=cap-pair-five
elif [[ $mode == active-history ]]; then
    archive=$(cd "$5" && pwd)
    prior=cfe2b52f6403d5edd612c314bc7110525a5f5789
    if [[ $(<"$archive/main.txt") != "$prior" || $(<"$archive/physical_npu.txt") != 4 || $prior == "$main" ]] ||
       ! git -C "$repo" merge-base --is-ancestor "$prior" "$main"; then
        echo "Expected a newer 512K candidate and the completed NPU4 capacity pair" >&2
        exit 2
    fi
    prefix=active-history-five
elif [[ $mode == pr30-parity ]]; then
    archive=$(cd "$5" && pwd)
    pr30=54acdb9c54853441d08898a2623407f4492a83a4
    if [[ $(<"$archive/main.txt") != "$main" || $(<"$archive/physical_npu.txt") != 4 || $pr30 == "$main" ]]; then
        echo "Expected the completed single-call capacity archive for this candidate on NPU4" >&2
        exit 2
    fi
    prefix=pr30-parity-five
elif [[ $mode == near512* ]]; then
    archive=$(cd "$5" && pwd)
    if [[ $(<"$archive/main.txt") != "$main" || $(<"$archive/physical_npu.txt") != 4 ]]; then
        echo "Expected the local NPU4 parity archive for the same native build" >&2
        exit 2
    fi
    prefix=$mode
    if [[ $mode == near512 ]]; then prefix=near512-ten; fi
fi
sdk=813f1750a50b7e6c92d5f5397f2fbc05befef6da
patched_tree=1de18e8315137d7a0227c3ae4c82afd45b5fe743
work=${PEER_G2_WORK:-/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold}
area="$work/peer-baseline"
sources="$work/fused-scan-sources"
mkdir -p "$area" "$sources"
out=$(mktemp -d "$area/$prefix.XXXXXX")
echo "RESULT_ROOT=$out"
exec > >(tee "$out/run.log") 2>&1
export PEER_PHYSICAL_NPU=4 ASCEND_RT_VISIBLE_DEVICES=4 PEER_LOCAL_ONLY=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PATH="/usr/local/python3.12.13/bin:$PATH"
unset PEER_RANKED_INPUT PEER_RANKED_HASH_FILE
if [[ -r /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
else
    source /usr/local/Ascend/cann-9.2.0/set_env.sh
fi
if [[ -r /usr/local/Ascend/nnal/atb/set_env.sh ]]; then source /usr/local/Ascend/nnal/atb/set_env.sh; fi
set -euo pipefail
# Serialize checkout/preparation of the shared build cache, but not NPU replay.
if (( head_parity )); then
    # The BS20..32 launcher uses this same card. Wait before taking the SDK lock.
    exec 7>"$sources/largebs-$(</proc/sys/kernel/random/boot_id)-npu4.lock"
    if ! flock -n 7; then
        echo "WAITING_FOR_PHYSICAL_NPU4"
        flock 7
    fi
fi
exec 8>"$sources/unified-sdk-${sdk:0:8}-${upstream:0:8}.lock"
flock 8
checkout() {
    local rev=$1 dest="$sources/unified-$1"
    git -C "$repo" cat-file -e "$rev^{commit}" || return
    if [[ -d "$dest" ]]; then
        if [[ $(git -C "$dest" rev-parse HEAD) == "$rev" && -z $(git -C "$dest" status --porcelain) ]]; then
            printf '%s\n' "$dest"
            return
        fi
        dest=$(mktemp -d "$dest.XXXXXX") || return
    fi
    git -C "$repo" worktree add -q --detach "$dest" "$rev" || return
    printf '%s\n' "$dest"
}
reader_src=$(checkout "$reader")
main_src=$(checkout "$main")
if (( head_parity )); then
    pr30_src=$main_src
    pr31_src=$(checkout "$pr31")
    pr31_table_src=$(checkout "$pr31_table")
elif [[ $mode == pr30-feature-steps ]]; then
    dualcut_src=$(checkout "$dualcut")
    paced_src=$(checkout "$paced")
fi
if [[ $mode == pr30-parity ]]; then pr30_src=$(checkout "$pr30"); fi
entry="$reader_src/tests/operators/lifetch"
support="$entry/support"
if [[ -n $feature_prior ]]; then
    active_rc=0
    python3 -B "$support/pr30_reuse_block.py" --active "$feature_prior" || active_rc=$?
    if (( active_rc == 0 )); then
        echo "PREVIOUS_REPLAY_ACTIVE=$feature_prior Stop the earlier run in its original terminal before resuming." >&2
        exit 1
    elif (( active_rc != 3 )); then
        exit "$active_rc"
    fi
    printf '%s\n' "$feature_prior" > "$out/prior_result_root.txt"
fi
if [[ $mode == pr30-feature-steps ]]; then
    # Also serialize two invocations of this launcher on the same machine and card.
    exec 9>"$sources/pr30-test-$(</proc/sys/kernel/random/boot_id)-npu4.lock"
    flock -n 9 || { echo "A feature replay is already using physical NPU4" >&2; exit 1; }
fi
printf '%s\n' "$main" >"$out/main.txt"
printf '%s\n' "$reader" >"$out/reader.txt"
printf '%s\n' "$sdk" >"$out/kvaf.txt"
printf '4\n' >"$out/physical_npu.txt"
cat /proc/sys/kernel/hostname >"$out/host.txt"
cat /proc/sys/kernel/random/boot_id >"$out/boot_id.txt"
if [[ -r /etc/machine-id ]]; then cat /etc/machine-id >"$out/machine_id.txt"; fi
if [[ $mode == active-history || $mode == pr30-parity || $mode == near512* || $mode == pr30-feature-steps ]]; then
    printf '%s\n' "$archive" >"$out/reference.txt"
    for field in host machine_id; do
        if [[ $auto_feature_reference == 0 && -f "$archive/$field.txt" ]] && ! cmp -s "$archive/$field.txt" "$out/$field.txt"; then
            echo "Capacity archive belongs to a different host ($field): $archive" >&2
            exit 1
        fi
    done
fi

# Reuse this machine's production Python, LIDU and URMA toolchain. The old SDK
# is only the local Git object source and supplies no runtime objects to this arm.
clean="$area/KVAF-de4fd37e"
baseline="$work/adaptive-sources/peer-serial-da318bdb"
peer_physical_npu=4
source "$support/peer_node_prepare.sh"
dependency_mode=full
# The published PR30 code requests LIDU registration even on its fused path.
if [[ $head_parity == 0 && ( $name == long_history_candidate || $mode == pr30-feature-steps ) ]]; then dependency_mode=fused-only; fi
peer_prepare_node "$repo" "$baseline" "$clean" "$work" "$out" "$dependency_mode"
sdk_src="$sources/unified-sdk-${sdk:0:8}-${upstream:0:8}"
if ! git -C "$clean" cat-file -e "$sdk^{commit}" 2>/dev/null; then
    git -C "$clean" fetch -q --no-tags https://gitcode.com/CacheOS/KVAF.git refs/heads/main
fi
if [[ ! -e "$sdk_src" ]]; then git -C "$clean" worktree add -q --detach "$sdk_src" "$sdk"; fi
[[ $(git -C "$sdk_src" rev-parse HEAD) == "$sdk" ]]
peer_node_logged "$out/prepare-sdk.log" "$PYTHON" -B "$main_src/scripts/build.py" --prepare-only --transport urma --kvaf-root "$sdk_src"
# Verify the four official main patches, including the previously untracked header.
git_env=(env "GIT_INDEX_FILE=$out/sdk-source.index" git -C "$sdk_src")
"${git_env[@]}" read-tree HEAD
"${git_env[@]}" add -u
"${git_env[@]}" add access_operator/algorithms/get/direct/urma_aiv/include/kvaf_batch_get_device.h
actual_tree=$("${git_env[@]}" write-tree)
if [[ $actual_tree != "$patched_tree" ]]; then echo "SDK_SOURCE_MISMATCH actual=$actual_tree expected=$patched_tree root=$sdk_src" >&2; exit 1; fi
printf '%s\n' "$actual_tree" >"$out/kvaf_tree.txt"

# Reuse the on-host liburma/header pair previously built for this CANN host.
cache="$clean/build-cpu/CMakeCache.txt"
headers=$(sed -n 's/^KVAF_URMA_ROOT:[^=]*=//p' "$cache")
urma=$(sed -n 's/^KVAF_URMA_LIBRARY:[^=]*=//p' "$cache")
if [[ -f "$headers/include/urma_api.h" ]]; then export KVAF_UMDK_ROOT="$headers"; fi
if [[ -f "$urma" ]]; then export KVAF_URMA_LIBRARY="$urma"; fi
export PATH="${PYTHON%/*}:$PATH"
jobs=${KVAF_BUILD_JOBS:-2}
echo "BUILDING_UNIFIED_SDK revision=$sdk patched_tree=$actual_tree jobs=$jobs logs=$out/build-sdk-{cpu,npu}.log"
peer_node_logged "$out/build-sdk-cpu.log" bash "$sdk_src/build.sh" cpu --jobs "$jobs"
peer_node_logged "$out/build-sdk-npu.log" bash "$sdk_src/build.sh" npu --without-aicpu --jobs "$jobs"
cpu="$sdk_src/build-cpu/runtime/kvaf_cpu_service"
if [[ $dependency_mode == full ]]; then
    lidu="$baseline/csrc/operators/lidu"
    vendors=("$lidu"/_custom_opp/vendors/*/op_api/lib/libcust_opapi.so)
    [[ ${#vendors[@]} == 1 && -s ${vendors[0]} ]]
    export ASCEND_CUSTOM_OPP_PATH="${vendors[0]%/op_api/lib/libcust_opapi.so}${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
    export PYTHONPATH="$lidu/torch_extension${PYTHONPATH:+:$PYTHONPATH}"
fi
export PYTHONPATH="$main_src/python:$sdk_src/build-npu/python${PYTHONPATH:+:$PYTHONPATH}"
export KVAF_BATCH_GET_KERNEL="$sdk_src/build-npu/bin/kvaf_execute_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$sdk_src/build-npu/bin/kvaf_execute_batch_put_kernels.o"
[[ -x $cpu && -s $KVAF_BATCH_GET_KERNEL && -s $KVAF_BATCH_PUT_KERNEL ]]
"$PYTHON" -B - "$sdk_src" <<'PY'
import sys
from pathlib import Path
import kvaf
assert Path(kvaf.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve()/'build-npu/python')
for method in ('begin_get_graph', 'end_get_graph', 'get_graph_urma_resources'):
    getattr(kvaf.NpuRuntime, method)
PY
flock -u 8
exec 8>&-
if [[ $mode == pr30-parity ]]; then
    echo "CONFIG reference=main_plus_PR30 candidate=active_512k NPU=4 BS=16 Q=4 warmup=4 measured_graphs=1 calls_per_graph=20"
    compare="$support/pr30_archive_compare.py"
    declare -A reference
    missing=()
    for g in 10 4 17 7 2; do
        if path=$("$PYTHON" -B "$compare" --candidate "$archive" --repo "$repo" --find-exact-group "$g"); then
            reference[$g]=$path
            echo "G$g PR30_REFERENCE=archive root=$path"
        else
            rc=$?
            if (( rc != 3 )); then exit "$rc"; fi
            missing+=("$g")
        fi
    done
    if (( ${#missing[@]} )); then
        ref_out=$(mktemp -d "$area/pr30-reference-five.XXXXXX")
        printf '%s\n' "$pr30" > "$ref_out/main.txt"
        for field in reader kvaf kvaf_tree host physical_npu boot_id machine_id; do
            if [[ -f $out/$field.txt ]]; then cp "$out/$field.txt" "$ref_out/$field.txt"; fi
        done
        printf 'PR30_MISSING_GROUPS=%s REFERENCE_ROOT=%s\n' "${missing[*]}" "$ref_out"
        for g in "${missing[@]}"; do reference[$g]=$ref_out; done
    fi
elif [[ $mode == pr30-feature-steps ]]; then
    if (( head_parity )); then
        arm_names=(pr30 pr31 pr31_table)
        printf '%s\n' "$pr30" >"$out/pr30.txt"
        printf '%s\n' "$pr31" >"$out/pr31.txt"
        printf '%s\n' "$pr31_table" >"$out/pr31_table.txt"
    else
        arm_names=(mainline dualcut paced)
        printf '%s\n' "$main" >"$out/mainline.txt"
        printf '%s\n' "$dualcut" >"$out/dualcut.txt"
        printf '%s\n' "$paced" >"$out/paced.txt"
        if (( live_feature_main )); then printf '1\n' >"$out/live_main.txt"; fi
    fi
    for block in 1 2; do
        mkdir -p "$out/block$block"
        for field in live_main "${arm_names[@]}" reader physical_npu host boot_id machine_id kvaf kvaf_tree reference; do
            if [[ -f "$out/$field.txt" ]]; then cp "$out/$field.txt" "$out/block$block/"; fi
        done
    done
    if (( head_parity )); then
        echo "CONFIG measured=pr30,pr31,pr31_table model_limit=131072 physical_slots=131328 input=original_trace blocks=2 reverse_order=1 groups=10,4,17,7,2 physical_NPU=4 BS16_Q4 warmup=4 calls_per_arm_per_block=10"
    elif (( live_feature_main )); then
        echo "CONFIG measured=mainline,dualcut,paced blocks=2 reverse_order=1 groups=10,4,17,7,2 physical_NPU=4 BS16_Q4 warmup=4 calls_per_arm_per_block=10 placement_and_oracle_archive=$archive"
    else
        echo "CONFIG baseline=$archive measured=dualcut,paced blocks=2 reverse_order=1 groups=10,4,17,7,2 physical_NPU=4 BS16_Q4 warmup=4 calls_per_arm_per_block=10 archived_main_calls=20"
        if [[ -n $feature_prior ]]; then echo "REUSE_COMPLETE_PAIRS_FROM=$feature_prior"; fi
    fi
elif [[ $mode == active-history ]]; then
    echo "CONFIG measured=active_512k archive=$archive native=$main sdk=$sdk NPU=4 BS=16 Q=4 warmup=4 calls=20"
elif [[ $mode == cap-pair ]]; then
    echo "CONFIG measured=capacity_pair model_limit=131072/524288 bound_history=131328/524544 input=original_capture native=$main sdk=$sdk NPU=4 BS=16 Q=4 warmup=4 measured_per_arm=20"
elif [[ $mode == near512-locality ]]; then
    echo "CONFIG groups=10,4,17,7,2 input=rank_preserving_short_capture model_limit=524288 BS=16 Q=4 physical_NPU=4 measured_calls_per_group=10"
elif [[ $mode == near512-five ]]; then
    echo "CONFIG model_limit=524288 physical_slots=524544 groups=10,4,17,7,2 input=stitched_per_group NPU=4 BS=16 Q=4 measured_calls_per_group=10"
elif [[ $mode == near512 ]]; then
    echo "CONFIG model_limit=524288 bound_history=524544 NPU=4 BS=16 Q=4 measured_calls=10 placement_archive=$archive"
else
    echo "CONFIG measured=$name native=$main sdk=$sdk NPU=4 BS=16 Q=4 token_bytes=656 warmup=4 calls=20"
fi
select_owner() {
    local g=$1 owner existing candidate_root
    owner=${PEER_OWNER_REFERENCE:-$area/fused.N7n8ZP}
    if [[ $g == 2 ]]; then owner=${PEER_G2_OWNER_REFERENCE:-$area/fused.gbxKC9}; fi
    if [[ $name == long_history_candidate ]] &&
       ! compgen -G "$owner/g$g/logical-ties/results/*/cases/diagnose/fused/score_topk_check.json" >/dev/null; then
        for existing in "$area"/fused.*/g"$g"/logical-ties; do
            candidate_root=${existing%/g"$g"/logical-ties}
            if [[ -f "$candidate_root/logical-ties.txt" && -f "$candidate_root/physical_npu.txt" ]] &&
               [[ $(<"$candidate_root/logical-ties.txt") == 888db6b1603d22f4142bd2ccffe0518611856a31 && $(<"$candidate_root/physical_npu.txt") == 4 ]] &&
               compgen -G "$existing/results/*/cases/diagnose/fused/score_topk_check.json" >/dev/null; then
                owner=$candidate_root
                break
            fi
        done
    fi
    if [[ ! -f "$owner/logical-ties.txt" ]] ||
       ! compgen -G "$owner/g$g/logical-ties/results/*/cases/diagnose/fused/score_topk_check.json" >/dev/null; then
        echo "Missing archived strong-serial oracle for G$g under $area" >&2
        exit 1
    fi
    printf '%s\n' "$owner"
}
run_arm() (
    set -euo pipefail
    local g=$1 label=$2 trace=$3 owner=$4 paired_placement=${5:-}
    arm="$out/g$g/$label"
    mkdir -p "$arm"
    placement="${PEER_MAIN_PLACEMENT_ROOT:-$area/main-five.hF6qQ8}/g$g/mainline/thp.json"
    # Select a reference from this host. Containers on different machines
    # may have identical generic hostnames and a shared results directory.
    here=$(cat "$out/host.txt")
    on_this_card() {
        local p=$1
        [[ -f $p/host.txt && -f $p/physical_npu.txt ]] &&
            [[ $(<"$p/host.txt") == "$here" && $(<"$p/physical_npu.txt") == 4 ]] &&
            [[ ! -s "$out/machine_id.txt" || ! -s "$p/machine_id.txt" || $(<"$out/machine_id.txt") == $(<"$p/machine_id.txt") ]]
    }
    if [[ $auto_feature_reference == 1 && -z $paired_placement ]]; then
        placement=
    elif [[ -n $paired_placement ]]; then
        placement=$paired_placement
        [[ -f $placement ]]
    elif [[ $mode == cap-pair ]] && ! on_this_card "${PEER_MAIN_PLACEMENT_ROOT:-$area/main-five.hF6qQ8}"; then
        placement=
    fi
    if [[ ! -f "$placement" && $name == long_history_candidate ]]; then
        for previous in "$area"/long-history-five.* "$area"/unified-main-five.* "$area"/main-five.*; do
            if on_this_card "$previous" && [[ -f "$previous/g$g/mainline/thp.json" ]]; then
                placement="$previous/g$g/mainline/thp.json"
                break
            fi
        done
    fi
    # The new SDK opens separate Put and Get sessions over one allocation.
    # Select the actual archive HCA for both before measuring that allocation.
    node=
    ub_device=${KVAF_CPU_UB_DEVICE:-}
    if [[ -f "$placement" ]]; then
        node=$("$PYTHON" -B "$support/li_window_report.py" numa "$placement")
        if [[ -z "$ub_device" ]]; then
            ub_device=$("$PYTHON" -B - "$placement" <<'PY_DEVICE'
import json, sys
with open(sys.argv[1]) as src:
    info = json.load(src)
name = info.get('ub_device')
if info.get('status') != 'measured' or not isinstance(name, str) or not name:
    raise SystemExit(f'Archive lacks a measured CPU URMA device: {sys.argv[1]}')
print(name)
PY_DEVICE
    )
        fi
    elif [[ $name == unified_main && $auto_feature_reference == 0 ]]; then
        echo "Missing verified mainline CPU placement: $placement" >&2
        exit 1
    elif (( auto_feature_reference )); then
        # No local placement receipt yet. Keep the CPU allocation on one
        # allowed NUMA node; reuse the observed adapter and node for its peer.
        for node_path in /sys/devices/system/node/node[0-9]*; do
            [[ -r "$node_path/cpulist" ]] || continue
            cpus=$(<"$node_path/cpulist")
            if [[ -n $cpus ]] && command -v taskset >/dev/null && taskset -c "$cpus" true >/dev/null 2>&1; then
                node=${node_path##*node}
                break
            fi
        done
    fi
    command=("$cpu")
    if [[ -n "$node" ]]; then
        if command -v numactl >/dev/null && numactl "--membind=$node" true >/dev/null 2>&1; then
            command=(numactl "--membind=$node" "$cpu")
        elif [[ -r /sys/devices/system/node/node${node}/cpulist ]] && command -v taskset >/dev/null; then
            cpus=$(<"/sys/devices/system/node/node${node}/cpulist")
            if taskset -c "$cpus" true >/dev/null 2>&1; then command=(taskset -c "$cpus" "$cpu"); fi
        fi
    fi
    port=$("$PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
    if [[ -n "$ub_device" ]]; then command+=(--ub-device "$ub_device"); fi
    "${command[@]}" --listen 127.0.0.1 --port "$port" --memory-page-mode thp >"$arm/cpu.log" 2>&1 &
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
    for ((i=0; i<30; i++)); do
        if grep -Fq "LISTEN address=127.0.0.1 port=$port" "$arm/cpu.log"; then break; fi
        kill -0 "$server" 2>/dev/null || break
        sleep 1
    done
    if ! grep -Fq "LISTEN address=127.0.0.1 port=$port" "$arm/cpu.log"; then tail -20 "$arm/cpu.log" >&2; exit 1; fi
    worker_timeout=1800
    # The production-graph audit restores and validates 20 independent wide
    # Index states outside the timed event. Allow for these host transfers.
    if [[ $mode == pr30-parity || $mode == near512* || $mode == pr30-feature-steps ]]; then worker_timeout=7200; fi
    "$PYTHON" -B "$support/peer_thp.py" --unified-sdk --pid "$server" --log "$arm/cpu.log" --output "$arm/thp.json" --timeout "$worker_timeout" >"$arm/thp.log" 2>&1 &
    monitor=$!
    args=()
    case $label in
        cap128k) args+=(--history-cap 131328);;
        cap512k) args+=(--history-cap 524544);;
    esac
    if [[ $mode == pr30-parity ]]; then args+=(--production-graph); fi
    if [[ $mode == pr30-feature-steps ]]; then
        case $label in
            pr30|pr31|pr31_table) args+=(--ablation-policy paced --original-128k);;
            mainline|dualcut) args+=(--ablation-policy single);;
            paced) args+=(--ablation-policy paced);;
            *) exit 2;;
        esac
    fi
    display=$label
    if [[ $label == mainline && $live_feature_main == 0 ]]; then display=$name; fi
    if [[ $mode == near512 ]]; then echo "CASE_BEGIN near512"; else echo "CASE_BEGIN G$g $display"; fi
    worker="$support/main_five_run.py"
    if [[ $mode == near512* ]]; then
        worker="$support/near512_worker.py"
        args+=(--work "$work")
        if [[ $mode == near512-five || $mode == near512-locality ]]; then
            args+=(--stitch-source --source-proof "$archive/g$g/cap512k/results/mainline/cases/diagnose/fused/mainline_checks.json")
            if [[ $mode == near512-locality ]]; then args+=(--preserve-workload); fi
        fi
    else
        args+=(--trace "$trace" --owner-archive "$owner")
    fi
    (cd "$arm" && timeout "$worker_timeout" "$PYTHON" -B "$worker" --main "$main_src" --revision "$main" \
         --group "$g" --kvaf-root "$sdk_src" \
         --kvaf-server "127.0.0.1:$port" --output "$arm" "${args[@]}") >"$arm/worker.log" 2>&1 || {
        echo "FAILED_ARM=G$g/$display WORKER_LOG=$arm/worker.log" >&2
        tail -30 "$arm/worker.log" >&2
        exit 1
    }
    if ! wait "$monitor"; then cat "$arm/thp.log" >&2; exit 1; fi
    monitor=
    if [[ $mode == near512* ]]; then
        grep -E '^(INPUT|CONFIG|WORKLOAD|CHECK|TIMING|LI_DETAIL|ARCHIVE_DELTA|TRAFFIC|REPLAY_COMPLETE)( |$)' "$arm/worker.log"
        exit 0
    fi
    if [[ $mode != single ]]; then exit 0; fi
    PYTHONPATH="$support:$PYTHONPATH" "$PYTHON" -B - "$out" "$owner" "$g" "$main" "$name" <<'PY_REPORT'
import sys
from pathlib import Path
from main_five_report import archived_oracle, case_folder, measured_parallel, read

root, owner = map(Path, sys.argv[1:3])
group, revision, name = int(sys.argv[3]), sys.argv[4], sys.argv[5]
ids, receipt = archived_oracle(owner, group)
source = read(case_folder(owner, group, 'logical-ties').parent / 'replay_input.json')
timing, exact, _, _, _ = measured_parallel(root, group, 'mainline', revision, owner,
                                          ids, receipt, source, verify_library=True)
print(f'G{group} CHECK arm={name} TopK_score_vs_strong_serial=PASS calls=20/20 '
      f'queries=1280/1280 exact_ID_sets={exact.sum()}/1280 KV=PASS ordered_Index=PASS rows=320/320')
print(f'G{group} {name}', *(f'{key}_mean/P50_us={value[0]:.2f}/{value[1]:.2f}'
                                  for key, value in timing.items()))
PY_REPORT
)
run_group() (
    set -euo pipefail
    g=$1
    if [[ $mode == near512-five || $mode == near512-locality ]]; then
        run_arm "$g" near512 "" "" "$archive/g$g/cap512k/thp.json"
        exit
    fi
    trace=$(bash "$entry/peer_g2_input.sh" "$work" "$g")
    if [[ $mode == pr30-feature-steps ]]; then
        group_reference=$archive
        if (( auto_feature_reference )); then
            group_reference=
            if (( !live_feature_main )); then
                group_reference=$("$PYTHON" -B "$support/feature_host_reference.py" "$area" "$g" baseline)
            fi
        fi
        anchor=
        if (( head_parity )) && [[ -d ${PEER_PR30_TIMING_ARCHIVE:-} ]]; then
            if anchor=$("$PYTHON" -B "$support/pr_head_anchor.py" "$PEER_PR30_TIMING_ARCHIVE" "$out" "$g"); then
                anchor_paths=()
                while IFS= read -r path; do anchor_paths+=("$path"); done <<< "$anchor"
                if [[ ${#anchor_paths[@]} != 2 || ! -d ${anchor_paths[0]} || ! -f ${anchor_paths[1]} ]]; then
                    echo "Invalid PR30 archive paths for G$g" >&2
                    exit 1
                fi
            else
                rc=$?
                if (( rc != 3 )); then exit "$rc"; fi
                anchor=
            fi
        fi
        if [[ -n $anchor ]]; then
            owner=${anchor_paths[0]}
            placement=${anchor_paths[1]}
            group_reference=$(cd "$PEER_PR30_TIMING_ARCHIVE" && pwd -P)
            echo "G$g PLACEMENT_AND_ORACLE_REFERENCE=$group_reference"
        elif [[ -n $group_reference ]]; then
            owner=$("$PYTHON" -B - "$support" "$group_reference" "$g" <<'PY_OWNER'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from main_five_report import case_folder, read
folder = case_folder(Path(sys.argv[2]), int(sys.argv[3]), 'mainline')
print(read(folder / 'score_topk_check.json')['native_reference']['archive_path'])
PY_OWNER
            )
            placement="$group_reference/g$g/mainline/thp.json"
            if (( live_feature_main )); then
                echo "G$g PLACEMENT_AND_ORACLE_REFERENCE=$group_reference"
            else
                echo "G$g MAIN_REFERENCE=$group_reference"
            fi
        else
            owner=$("$PYTHON" -B "$support/feature_host_reference.py" "$area" "$g" owner)
            if [[ -z $owner ]]; then
                owner="$out/oracle/g$g"
                KVIP_Q4_NATIVE_LI_CACHE="${KVIP_Q4_NATIVE_LI_CACHE:-$area/native-li-reference-g2}" \
                    "$PYTHON" -B "$support/feature_score_oracle.py" "$trace" "$g" "$owner"
            fi
            placement=$("$PYTHON" -B "$support/feature_host_reference.py" "$area" "$g" placement)
            if (( live_feature_main )); then
                echo "G$g ORACLE_REFERENCE=$owner"
            else
                echo "G$g MAIN_REFERENCE=unavailable measured=dualcut,paced oracle=$owner"
            fi
        fi
        result_root=$out
        for block in 1 2; do
            out="$result_root/block$block"
            if (( live_feature_main )); then
                case $g in
                    10|2) order=("${arm_names[@]}");;
                    4|7)  order=("${arm_names[1]}" "${arm_names[2]}" "${arm_names[0]}");;
                    17)   order=("${arm_names[2]}" "${arm_names[0]}" "${arm_names[1]}");;
                esac
                if [[ $block == 2 ]]; then order=("${order[2]}" "${order[1]}" "${order[0]}"); fi
            else
                order=(dualcut paced)
                if [[ $g == 4 || $g == 7 ]]; then order=(paced dualcut); fi
                if [[ $block == 2 ]]; then order=("${order[1]}" "${order[0]}"); fi
            fi
            mkdir -p "$out/g$g"
            printf '%s\n' "${group_reference:-unavailable}" >"$out/g$g/reference.txt"
            printf '%s\n' "${order[*]}" >"$out/g$g/order.txt"
            if [[ -n $feature_prior && $live_feature_main == 0 ]]; then
                reuse_rc=0
                "$PYTHON" -B "$support/pr30_reuse_block.py" "$feature_prior" "$out" "$g" "$owner" || reuse_rc=$?
                case $reuse_rc in
                    0) continue;;
                    3) ;;
                    *) exit "$reuse_rc";;
                esac
            fi
            echo "BLOCK_BEGIN G$g block=$block order=${order[*]}"
            for label in "${order[@]}"; do
                (
                    case $label in
                        pr30) main=$pr30; main_src=$pr30_src;;
                        pr31) main=$pr31; main_src=$pr31_src;;
                        pr31_table) main=$pr31_table; main_src=$pr31_table_src;;
                        dualcut) main=$dualcut; main_src=$dualcut_src;;
                        paced) main=$paced; main_src=$paced_src;;
                    esac
                    export PYTHONPATH="$main_src/python:$PYTHONPATH"
                    started=$SECONDS
                    run_arm "$g" "$label" "$trace" "$owner" "$placement"
                    echo "ARM_COMPLETE G$g $label block=$block wall_s=$((SECONDS-started))"
                    grep -E '^FEATURE_(INPUT_RESTORE|VALIDATION|SCORE_CHECK)( |$)' "$out/g$g/$label/worker.log" || true
                )
                if [[ -z $placement ]]; then placement="$out/g$g/$label/thp.json"; fi
            done
            if (( head_parity )); then
                "$PYTHON" -B "$support/pr_head_parity_report.py" "$out" "$g" "$owner"
            else
                "$PYTHON" -B "$support/pr30_feature_steps.py" "$out" "$g" "$owner"
            fi
        done
        exit
    fi
    owner=$(select_owner "$g")
    if [[ $mode == pr30-parity ]]; then
        local_ref=${reference[$g]}
        placement="$archive/g$g/cap512k/thp.json"
        # Keep each arm's production checkout and receipt under its own root.
        measure_pr30() (
            if [[ $local_ref == "${ref_out:-}" ]]; then
                out=$ref_out
                main=$pr30
                main_src=$pr30_src
                export PYTHONPATH="$main_src/python:$sdk_src/build-npu/python${PYTHONPATH:+:$PYTHONPATH}"
                run_arm "$g" cap128k "$trace" "$owner" "$placement"
            fi
            exit 0
        )
        # Alternate first arm when both measurements are missing.
        if [[ $g != 4 && $g != 7 ]]; then measure_pr30; fi
        run_arm "$g" cap512k "$trace" "$owner" "$placement"
        if [[ $g == 4 || $g == 7 ]]; then measure_pr30; fi
        printf '%s\n' "$local_ref" > "$out/g$g/pr30_reference.txt"
        "$PYTHON" -B "$support/pr30_parity_group.py" "$out" "$g" "$local_ref" "$repo" "$archive"
        exit
    fi
    if [[ $mode == single ]]; then
        run_arm "$g" mainline "$trace" "$owner"
        exit
    fi
    if [[ $mode == active-history ]]; then
        run_arm "$g" cap512k "$trace" "$owner" "$archive/g$g/cap512k/thp.json"
        "$PYTHON" -B "$support/cap_pair_report.py" "$out" "$g" "$owner" --reference "$archive"
        exit
    fi
    # Both shapes use the same binary, input, native score oracle, fresh
    # worker and peer. Alternate which shape goes first across captures.
    case $g in
        4|7) order=(cap512k cap128k);;
        *)   order=(cap128k cap512k);;
    esac
    mkdir -p "$out/g$g"
    printf '%s\n' "${order[*]}" >"$out/g$g/order.txt"
    run_arm "$g" "${order[0]}" "$trace" "$owner"
    run_arm "$g" "${order[1]}" "$trace" "$owner" "$out/g$g/${order[0]}/thp.json"
    "$PYTHON" -B "$support/cap_pair_report.py" "$out" "$g" "$owner"
)
if [[ $mode == near512 ]]; then
    # Only reuse the 128K source operands if a native long trace isn't available.
    # The input builder distinguishes the source group from the measured case.
    run_arm 4 near512 "" "" "$archive/g4/cap512k/thp.json"
    echo "REPLAY_COMPLETE RESULT_ROOT=$out"
    exit 0
fi
failed=()
trap 'exit 130' INT
trap 'exit 143' TERM
for g in 10 4 17 7 2; do
    set +e
    run_group "$g"
    code=$?
    set -e
    if (( code == 130 || code == 143 )); then exit "$code"; fi
    if (( code )); then failed+=("$g"); fi
done
if ((${#failed[@]})); then echo "REPLAY_FAILED groups=${failed[*]} RESULT_ROOT=$out" >&2; exit 1; fi
if (( head_parity )); then
    "$PYTHON" -B "$support/pr_head_parity_report.py" "$out" --summary
elif [[ $mode == pr30-feature-steps ]]; then
    for block in 1 2; do
        echo "BLOCK_SUMMARY block=$block"
        "$PYTHON" -B "$support/pr30_feature_steps.py" "$out/block$block" --summary
    done
    "$PYTHON" -B "$support/pr30_feature_repeats.py" "$out"
elif [[ $mode == cap-pair ]]; then
    "$PYTHON" -B "$support/cap_pair_report.py" "$out" --summary
elif [[ $mode == active-history ]]; then
    "$PYTHON" -B "$support/cap_pair_report.py" "$out" --reference "$archive" --summary
elif [[ $mode == pr30-parity ]]; then
    "$PYTHON" -B - "$out" <<'PY_MAP'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
refs = {g: (root/f'g{g}/pr30_reference.txt').read_text().strip() for g in (10, 4, 17, 7, 2)}
(root/'pr30_references.json').write_text(json.dumps(refs, indent=2))
PY_MAP
    "$PYTHON" -B "$compare" --candidate "$out" --reference-map "$out/pr30_references.json" --repo "$repo"
elif [[ $mode == near512-five || $mode == near512-locality ]]; then
    "$PYTHON" -B - "$out" "$mode" <<'PY_NEAR512'
import json
from pathlib import Path
from statistics import mean
import sys

root = Path(sys.argv[1])
kind = 'rank_preserving_short_capture' if sys.argv[2] == 'near512-locality' else 'stitched_short_capture_pages'
groups = (10, 4, 17, 7, 2)
rows = []
for group in groups:
    path = root / f'g{group}/near512/near512_result.json'
    row = json.loads(path.read_text())
    if ((row['group'], row['donor']['group'], row['source_kind']) != (group, group, kind)
            or row['measured_calls'] != 10 or row['scores']['queries'] != 640):
        raise ValueError(f'G{group}: incomplete or wrong-trace result in {path}')
    rows.append(row)
print('FIVE_GROUPS_COMPLETE groups=5 measured_calls=50 TopK_queries=3200',
      *(f'{key}_mean_us={mean(row["mean_p50_us"][key][0] for row in rows):.2f}'
        for key in ('graph', 'LI1_to_LI3', 'final_topk_through_tail')))
PY_NEAR512
fi
echo "REPLAY_COMPLETE RESULT_ROOT=$out"
