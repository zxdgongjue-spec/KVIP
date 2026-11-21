#!/usr/bin/env bash
# Replay the captured inputs with a single transport build and timing protocol.
set -eo pipefail
peer_physical_npu=${PEER_PHYSICAL_NPU-4}
if [[ ! $peer_physical_npu =~ ^(0|[1-9][0-9]*)$ ]]; then
    printf 'PEER_PHYSICAL_NPU must be a single physical device ordinal, got "%s"\n' "$peer_physical_npu" >&2
    exit 2
fi
root=$(cd "$(dirname "$0")/../../.." && pwd)
tools="$root/tests/operators/lifetch"
work=${PEER_G2_WORK:-/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold}
area="$work/peer-baseline"
src="$work/adaptive-sources"
old=cdfc4da4b53b80255740183c7a0a8db8bf350dcb
new=4262f8e6be86c7da65cc3771b97d30467bb14e4e
serial_revision=da318bdb81344672b5d9013e04d517669d3eb142
baseline="$src/peer-serial-${serial_revision:0:8}"
clean="$area/KVAF-de4fd37e"
transport=a2857b460a74704114fd30a849ae38666cc2c372
kvaf="$area/KVAF-${transport:0:8}"
snapshot="$work/skip-results.po80Nw/g2-input"
ranked=
ranked_sha=1d2486a76151ae37662c47aab835d06df50be866
adaptive=
help_reference=
joint_reference=
joint=false
joint_local=false
joint_archive=false
ranked_auto=false
query_owner=false
local_resolver=false
offset_resolver=false
row_deps=false
plan_candidate=
packet_reference=
plan_control=
current_fixed=6bca602ad9e548e9f12c0142fea25f2bc5c6877d
ranked_q_workers=${PEER_RANKED_Q_WORKERS:-2}
ranked_groups=(7 2 10 4 17)
resume=
anchor=
has_anchor=false
four_field=5072f2c907f0fbd3bf1c1fbe499dd979c8c1dfca
if [[ ${1:-} == --ranked-help || ${1:-} == --ranked-joint ]]; then
    if [[ $1 == --ranked-joint ]]; then joint=true; fi
    ranked=${2:?supply full helper candidate revision}
    [[ $ranked =~ ^[0-9a-f]{40}$ && $ranked != "$four_field" ]] || { echo 'Expected the full helper candidate SHA' >&2; exit 2; }
    help_reference="$area/fused.rs7TG7"
    shift 2
    while (( $# )); do
        case "$1" in
            --previous) joint_reference=${2:?supply the completed 56dcd74a reference root} ;;
            --reference) help_reference=${2:?supply a completed ranked reference root} ;;
            --resume) resume=${2:?supply the helper run root} ;;
            --anchor) anchor=${2:?supply group number}
                case "$anchor" in 7|2|10|4|17) ;; *) echo 'Unsupported anchor group' >&2; exit 2 ;; esac ;;
            *) echo "Unknown ranked helper option: $1" >&2; exit 2 ;;
        esac
        shift 2
    done
    if [[ ! -s "$help_reference/baseline.txt" || ! -s "$help_reference/candidate.txt" ]] ||
       [[ $(<"$help_reference/baseline.txt") != "$ranked_sha" || $(<"$help_reference/candidate.txt") != "$four_field" ]]; then
        echo "Need the completed 1d2486a7 / 5072f2c9 run at $help_reference; use --reference PATH to select it." >&2
        exit 1
    fi
    help_reference=$(cd "$help_reference" && pwd -P)
    if "$joint"; then
        [[ -n "$joint_reference" && -z "$anchor" ]] || { echo '--ranked-joint requires --previous DIR and uses archived controls' >&2; exit 2; }
        joint_reference=$(cd "$joint_reference" && pwd -P)
        for g in 2 7 10 17 4; do
            test -s "$joint_reference/g$g/comparison.json" || { echo "Missing previous G$g result in $joint_reference" >&2; exit 1; }
        done
    elif [[ -n "$joint_reference" ]]; then
        echo '--previous requires --ranked-joint' >&2; exit 2
    fi
elif [[ ${1:-} == --offset-resolver || ${1:-} == --row-deps || ${1:-} == --li-packets ]]; then
    [[ $# -ge 2 && $2 =~ ^[0-9a-f]{40}$ ]] || { echo "Usage: peer_fused.sh $1 NATIVE_SHA [GROUP ...]" >&2; exit 2; }
    query_owner=true
    offset_resolver=true
    ranked_sha=f8b185e3d042a18b24fe695c0a49a7d969ca487d
    fixed_order_rev=7b60f0dddab91a4b154043e78088842d7856cba6
    if [[ $1 != --offset-resolver ]]; then
        row_deps=true
        plan_candidate=${1#--}
        if [[ $plan_candidate == li-packets ]]; then
            plan_control=row-deps
            ranked_sha=6fe0f113205a6f7328a470a03b86d4bd2efe5dc5
        else
            plan_control=offset
            ranked_sha=2942155cad43d9876087c817e38a87275f1a9b5b
        fi
        [[ $2 != "$ranked_sha" ]] || { echo "Supply a candidate distinct from the $plan_control control" >&2; exit 2; }
    fi
    ranked=$2
    unset KVIP_URMA_ROUTE_POLICY
    export PEER_LOCAL_ONLY=1
    shift 2
    offset_groups=(7 2 10 17 4)
    if [[ $plan_candidate == li-packets ]]; then
        offset_groups=(7 2)
        if [[ ${1:-} == --reference ]]; then
            packet_reference=$(cd "${2:?missing prior row-deps result directory}" && pwd -P)
            shift 2
            if [[ ! -s "$packet_reference/row-deps.txt" || ! -s "$packet_reference/fixed-order.txt" ||
                  ! -s "$packet_reference/physical_npu.txt" ||
                  $(<"$packet_reference/row-deps.txt") != "$ranked_sha" ||
                  $(<"$packet_reference/fixed-order.txt") != "$fixed_order_rev" ||
                  $(<"$packet_reference/physical_npu.txt") != "$peer_physical_npu" ]]; then
                echo "Incompatible row-deps reference: $packet_reference" >&2
                exit 1
            fi
        fi
    fi
    if (($#)); then
        offset_groups=()
        for group in "$@"; do
            case "$group" in 7|2|10|17|4) ;; *) echo "Unsupported group: $group" >&2; exit 2 ;; esac
            [[ " ${offset_groups[*]} " != *" $group "* ]] || { echo "Duplicate group: $group" >&2; exit 2; }
            offset_groups+=("$group")
        done
        shift "$#"
    fi
elif [[ ${1:-} == --local-resolver ]]; then
    [[ $# == 1 ]] || { echo 'Usage: peer_fused.sh --local-resolver' >&2; exit 2; }
    query_owner=true
    local_resolver=true
    ranked_sha=f8b185e3d042a18b24fe695c0a49a7d969ca487d
    fixed_order_rev=7b60f0dddab91a4b154043e78088842d7856cba6
    spill_rev=b0404c2760f82049920a0d43571e8a7aac904d6e
    ranked=99b4d02b1b451117a73356e0099ee79276635a6a
    unset KVIP_URMA_ROUTE_POLICY
    export PEER_LOCAL_ONLY=1
    shift
elif [[ ${1:-} == --query-owner ]]; then
    query_owner=true
    ranked=${2:?supply query-owner commit}
    ranked_sha=$current_fixed
    unset KVIP_URMA_ROUTE_POLICY
    export PEER_LOCAL_ONLY=1
    [[ $ranked =~ ^[0-9a-f]{40}$ && $ranked != "$ranked_sha" ]] || exit 2
    shift 2
elif [[ ${1:-} == --ranked-auto ]]; then
    ranked_auto=true
    ranked_groups=(2 7 10 4 17)
    ranked=${2:?supply full adaptive candidate commit}
    [[ $ranked =~ ^[0-9a-f]{40}$ && $ranked != "$current_fixed" ]] || { echo 'Expected the adaptive candidate SHA' >&2; exit 2; }
    ranked_groups=(2 7 10 17 4)
    help_reference="$area/help-772293d4-fecf79e510598355a1ee"
    shift 2
    if [[ ${1:-} == --reference ]]; then help_reference=${2:?supply same-node ranked reference}; shift 2; fi
    help_reference=$(cd "$help_reference" && pwd -P)
    if [[ $(<"$help_reference/baseline.txt") != "$ranked_sha" ||
          $(<"$help_reference/physical_npu.txt") != "$peer_physical_npu" ||
          $(<"$help_reference/host.txt") != "$(</proc/sys/kernel/hostname)" ]]; then
        echo "Archived original revision, host or NPU differs: $help_reference" >&2; exit 1
    fi
elif [[ ${1:-} == --ranked-joint-local ]]; then
    ranked=${2:?supply candidate commit}
    [[ $ranked =~ ^[0-9a-f]{40}$ ]] || { echo 'Expected the full candidate SHA' >&2; exit 2; }
    joint=true
    joint_local=true
    shift 2
    if [[ ${1:-} == --reference ]]; then
        help_reference=$(cd "${2:?supply the prior same-node ranked-joint-local run directory}" && pwd -P)
        joint_local=false
        joint_archive=true
        shift 2
        case "$ranked_q_workers" in 1|2|4) ;; *) echo 'PEER_RANKED_Q_WORKERS expects 1, 2 or 4' >&2; exit 2 ;; esac
        if [[ -n ${PEER_RANKED_GROUPS:-} ]]; then
            read -ra ranked_groups <<<"$PEER_RANKED_GROUPS"
            (( ${#ranked_groups[@]} > 0 )) || { echo 'Empty group selection' >&2; exit 2; }
            for g in "${ranked_groups[@]}"; do
                case "$g" in 2|4|7|10|17) ;; *) echo "Unknown group: $g" >&2; exit 2 ;; esac
            done
        fi
        joint_control=772293d4f0ccb002a1d5210016619af534008e88
        if [[ $(<"$help_reference/baseline.txt") != "$ranked_sha" ||
              $(<"$help_reference/candidate.txt") != "$joint_control" ||
              $(<"$help_reference/physical_npu.txt") != "$peer_physical_npu" ||
              $(<"$help_reference/host.txt") != "$(</proc/sys/kernel/hostname)" ]]; then
            echo "Archived ranked joint control, host, or physical NPU mismatch: $help_reference" >&2
            exit 1
        fi
        [[ $ranked != "$joint_control" ]] || { echo 'Candidate equals the archived control' >&2; exit 2; }
    fi
elif [[ ${1:-} == --ranked-index ]]; then
    ranked=${2:?supply candidate commit}
    shift 2
elif [[ ${1:-} == --main-adaptive ]]; then
    adaptive=${2:?supply candidate commit}
    shift 2
fi
reuse=
if [[ -z "$ranked" && $# -eq 2 && $1 == --reuse-serial ]]; then
    if [[ -d "$2/serial" ]]; then
        reuse=$(cd "$2/serial" && pwd -P)
    elif [[ -n "$adaptive" ]]; then
        printf 'SERIAL_BASELINE missing=%s measuring_locally=true\n' "$2/serial" >&2
    else
        printf 'Missing serial result: %s/serial\n' "$2" >&2
        exit 1
    fi
elif [[ $# -ne 0 ]]; then
    echo 'Usage: peer_fused.sh --li-packets NATIVE_SHA [--reference DIR] [GROUP ...] | --row-deps NATIVE_SHA [GROUP ...] | --offset-resolver NATIVE_SHA [GROUP ...] | --ranked-auto FULL_SHA [--reference DIR] | --ranked-joint-local FULL_SHA [--reference DIR] | --ranked-joint FULL_SHA --previous DIR [--reference DIR] | --ranked-help FULL_SHA [--reference DIR] [--resume DIR] [--anchor GROUP] | --ranked-index CANDIDATE_SHA | [--main-adaptive CANDIDATE_SHA] [--reuse-serial PREVIOUS_RESULT_ROOT]' >&2
    exit 2
fi
export PATH=/usr/local/python3.12.13/bin:$PATH
mkdir -p "$area"
if [[ -n "$help_reference" ]] || "$joint_local"; then
    # Repeating the command resumes the same machine, harness and exact revisions.
    harness=$(git -C "$root" rev-parse HEAD)
    host_id=$(</proc/sys/kernel/hostname)
    boot_id=$(</proc/sys/kernel/random/boot_id)
    key_parts=("$ranked" "$help_reference" "$host_id" "$boot_id" "$harness" "$peer_physical_npu")
    if "$joint"; then key_parts+=("$joint_reference"); fi
    if "$joint_archive"; then key_parts+=("joint-control" "$ranked_q_workers"); fi
    run_key=$(printf '%s\n' "${key_parts[@]}" |
        python3 -B -c 'import hashlib, sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest()[:20])')
    if [[ -n "$resume" ]]; then
        out=$(cd "$resume" && pwd -P)
        [[ -f "$out/run.key" && $(<"$out/run.key") == "$run_key" ]] || { echo "Incompatible --resume directory: $out" >&2; exit 1; }
    else
        out="$area/help-${ranked:0:8}-$run_key"
        mkdir -p "$out"
    fi
    exec 9>"$out/run.lock"
    # FD 9 shares the shell's open file description, which holds the lock
    # through all groups after this Python process exits.
    python3 -B - "$out" <<'LOCK'
import fcntl
import sys

try:
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit(f"A replay is already running in {sys.argv[1]}")
LOCK
    printf '%s\n' "$run_key" >"$out/run.key"
    if [[ -n "$help_reference" ]]; then printf '%s\n' "$help_reference" >"$out/reference.txt"; fi
    if [[ -n "$joint_reference" ]]; then printf '%s\n' "$joint_reference" >"$out/previous.txt"; fi
    printf '%s\n' "$harness" >"$out/harness.txt"
    printf '%s\n' "$host_id" >"$out/host.txt"
    if [[ -n "$anchor" ]]; then
        mkdir -p "$out/g$anchor"
        touch "$out/g$anchor/anchor.requested"
    fi
    for marker in "$out"/g*/anchor.requested; do
        if [[ -f "$marker" ]]; then has_anchor=true; break; fi
    done
else
    out=$(mktemp -d "$area/fused.XXXXXX")
fi
printf 'RESULT_ROOT=%s\n' "$out"
if [[ -n "$packet_reference" ]]; then
    printf '%s\n' "$packet_reference" >"$out/packet-reference.txt"
    for group in "${offset_groups[@]}"; do
        for label in fixed-order row-deps; do
            test -s "$packet_reference/g$group/$label/thp.json" || {
                echo "Missing archived G$group $label result in $packet_reference" >&2; exit 1;
            }
        done
    done
fi
if [[ -n "$ranked" ]]; then
    printf '%s\n' "$peer_physical_npu" >"$out/physical_npu.txt"
fi
if "$query_owner"; then
    if [[ -x "$area/python-${serial_revision:0:8}/bin/python3" ]]; then
        export PATH="$area/python-${serial_revision:0:8}/bin:$PATH"
    fi
    declare -A owner_traces
    input_groups=(2 7 10 17 4)
    if "$offset_resolver"; then input_groups=("${offset_groups[@]}"); fi
    for group in "${input_groups[@]}"; do
        owner_traces[$group]=$(bash "$tools/peer_g2_input.sh" "$work" "$group")
    done
    printf 'INPUTS_READY local_groups=%s\n' "${input_groups[*]}"
fi
if [[ -r /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
else
    source /usr/local/Ascend/cann-9.2.0/set_env.sh
fi
if [[ -r /usr/local/Ascend/nnal/atb/set_env.sh ]]; then source /usr/local/Ascend/nnal/atb/set_env.sh; fi
set -euo pipefail
pin() {
    local repo=$1 checkout=$2 sha=$3
    if [[ ! -e "$checkout" ]]; then git -C "$repo" worktree add -q --detach "$checkout" "$sha"; fi
    [[ $(git -C "$checkout" rev-parse HEAD) == "$sha" ]] || { echo "Unexpected revision in $checkout" >&2; exit 1; }
    git -C "$checkout" diff --quiet && git -C "$checkout" diff --cached --quiet || { echo "Modified source: $checkout" >&2; exit 1; }
}
if [[ -z "$ranked" ]]; then trace=$(bash "$tools/peer_g2_input.sh" "$work"); fi
local_baseline=false
if [[ -n "$ranked" || ( -n "$adaptive" && -z "$reuse" ) ]]; then
    local_baseline=true
    source "$tools/support/peer_node_prepare.sh"
    preparation=${ranked:+fused-only}
    if "$ranked_auto"; then preparation=full; fi
    peer_prepare_node "$root" "$baseline" "$clean" "$work" "$out" "$preparation"
fi
if [[ -z "$ranked" ]]; then
    pin "$root" "$baseline" "$serial_revision"
    git -C "$root" diff --exit-code --quiet "$serial_revision" HEAD -- csrc/operators/lidu
fi
if [[ -n "$ranked" ]]; then
    if "$query_owner"; then
        if ! git -C "$root" cat-file -e "$ranked^{commit}" 2>/dev/null; then
            if "$row_deps"; then git -C "$root" fetch -q origin "refs/heads/perf/urma-q4-$plan_candidate"
            elif "$offset_resolver"; then git -C "$root" fetch -q origin refs/heads/perf/urma-q4-offset-slots
            elif "$local_resolver"; then git -C "$root" fetch -q origin refs/heads/perf/urma-q4-local-resolver
            else git -C "$root" fetch -q origin refs/heads/perf/urma-q4-query-owner; fi
        fi
        ranked_base="$src/ranked-${ranked_sha:0:8}"
        pin "$root" "$ranked_base" "$ranked_sha"
        if "$local_resolver" || "$offset_resolver"; then
            ranked_fixed_order="$src/ranked-${fixed_order_rev:0:8}"
            if ! git -C "$root" cat-file -e "$fixed_order_rev^{commit}" 2>/dev/null; then
                git -C "$root" fetch -q origin refs/heads/perf/urma-q4-tie-control
            fi
            pin "$root" "$ranked_fixed_order" "$fixed_order_rev"
            compare_rev=$ranked
            if "$row_deps"; then compare_rev=$ranked_sha; fi
            # The row-deps control already changed the final classifier in this header.
            fixed_exclusions=()
            if [[ $plan_control == row-deps ]]; then
                fixed_exclusions+=(':(exclude)csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer/indexer_vector.h')
            fi
            git -C "$root" diff --quiet "$fixed_order_rev" "$compare_rev" -- \
                csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer "${fixed_exclusions[@]}"
            printf '%s\n' "$fixed_order_rev" >"$out/fixed-order.txt"
            if "$row_deps"; then
                printf '%s\n' "$ranked_sha" >"$out/$plan_control.txt"
                printf '%s\n' "$ranked" >"$out/$plan_candidate.txt"
            else
                printf '%s\n' "$ranked_sha" >"$out/chain.txt"
                if "$offset_resolver"; then
                    printf '%s\n' "$ranked" >"$out/offset.txt"
                else
                    ranked_spill="$src/ranked-${spill_rev:0:8}"
                    pin "$root" "$ranked_spill" "$spill_rev"
                    printf '%s\n' "$spill_rev" >"$out/spill.txt"
                    printf '%s\n' "$ranked" >"$out/local.txt"
                fi
            fi
        fi
    elif "$ranked_auto"; then
        git -C "$root" merge-base --is-ancestor "$current_fixed" "$ranked"
        ranked_base="$src/ranked-${current_fixed:0:8}"
        pin "$root" "$ranked_base" "$current_fixed"
    elif [[ -n "$help_reference" ]]; then
        if ! git -C "$root" cat-file -e "$ranked^{commit}" 2>/dev/null; then
            if "$joint"; then git -C "$root" fetch -q origin perf/ranked-index-post-tail; else git -C "$root" fetch -q origin perf/ranked-index-help4; fi
        fi
        if "$joint_archive"; then git -C "$root" merge-base --is-ancestor "$joint_control" "$ranked"; fi
        git -C "$root" merge-base --is-ancestor "$four_field" "$ranked"
        ranked_base="$src/ranked-${four_field:0:8}"
        if "$has_anchor"; then pin "$root" "$ranked_base" "$four_field"; fi
    else
        if "$joint_local"; then
            if ! git -C "$root" cat-file -e "$ranked^{commit}" 2>/dev/null; then
                git -C "$root" fetch -q origin perf/ranked-index-post-tail
            fi
        else
            git -C "$root" fetch -q origin codex/lifetch-ranked-integration perf/ranked-all-aiv-index
        fi
        ranked_base="$src/ranked-${ranked_sha:0:8}"
        pin "$root" "$ranked_base" "$ranked_sha"
    fi
    git -C "$root" merge-base --is-ancestor "$ranked_sha" "$ranked"
    ranked_candidate="$src/ranked-${ranked:0:8}"
    pin "$root" "$ranked_candidate" "$ranked"
    printf '%s\n' "$ranked_sha" >"$out/baseline.txt"
    if "$ranked_auto"; then printf '%s\n' "$current_fixed" >"$out/fixed.txt"; fi
    printf '%s\n' "$ranked" >"$out/candidate.txt"
elif [[ -n "$adaptive" ]]; then
    main_sha=a56154c37bebb0928da4a4095e9670898e6b21c7
    git -C "$root" diff --exit-code --quiet "$main_sha" HEAD -- csrc python
    git -C "$root" fetch -q origin perf/urma-main-adaptive
    git -C "$root" merge-base --is-ancestor "$main_sha" "$adaptive"
    adaptive_src="$src/urma-main-${adaptive:0:8}"
    pin "$root" "$adaptive_src" "$adaptive"
    printf '%s\n' "$adaptive" >"$out/candidate.txt"
else
    git -C "$root" fetch -q origin test/peer-g2-low-cq
    pin "$root" "$src/auto-${old:0:8}" "$old"
    pin "$root" "$src/auto-${new:0:8}" "$new"
fi
if ! git -C "$clean" cat-file -e "$transport^{commit}" 2>/dev/null; then
    git -C "$clean" fetch -q origin test/peer-main-urma-fused-context
fi
pin "$clean" "$kvaf" "$transport"
pin "$clean" "$clean" de4fd37e02fa86357f71fe4fcbbbecc3e75d1d58
[[ -n "$ranked" || -n "$adaptive" || -s "$snapshot/manifest.json" ]] || { echo "Missing G2 snapshot $snapshot" >&2; exit 1; }
if "$local_baseline"; then
    cann="$ASCEND_CANN_PACKAGE_PATH"
else
    cann=$(sed -n 's/^ASCEND_CANN_PACKAGE_PATH:[^=]*=//p' "$clean/build-npu/CMakeCache.txt")
fi
test -d "$cann/include/acl"
export ASCEND_HOME_PATH="$cann" ASCEND_CANN_PACKAGE_PATH="$cann"
export ASCEND_RT_VISIBLE_DEVICES="$peer_physical_npu" SOC_VERSION=Ascend950DT_9582 ASCEND_LAUNCH_BLOCKING=0
export MAX_JOBS=2 OMP_NUM_THREADS=1
venv="$area/python-${serial_revision:0:8}"
export PATH="$venv/bin:$PATH" PYTHON="$venv/bin/python"
test -x "$PYTHON"
lidu="$baseline/csrc/operators/lidu"
export PYTHONPATH="$root/python:$kvaf/build-npu/python${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "$ranked" ]] || "$ranked_auto"; then
    vendors=("$lidu"/_custom_opp/vendors/*/op_api/lib/libcust_opapi.so)
    [[ ${#vendors[@]} == 1 && -f ${vendors[0]} ]]
    export ASCEND_CUSTOM_OPP_PATH="${vendors[0]%/op_api/lib/libcust_opapi.so}${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
    export PYTHONPATH="$lidu/torch_extension:$PYTHONPATH"
fi
export KVAF_BATCH_GET_KERNEL="$kvaf/build-npu/bin/kvaf_execute_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$kvaf/build-npu/bin/kvaf_execute_batch_put_kernels.o"
extension=$("$PYTHON" -c 'import sysconfig; print("_kvaf"+sysconfig.get_config_var("EXT_SUFFIX"))')
if [[ ! -s "$KVAF_BATCH_GET_KERNEL" || ! -s "$KVAF_BATCH_PUT_KERNEL" ||
      ! -s "$kvaf/build-npu/python/kvaf/$extension" ]]; then
    echo 'BUILD KVAF URMA NPU'
    if ! bash "$kvaf/build.sh" npu --without-aicpu --jobs 2 >"$out/build.log" 2>&1; then
        tail -15 "$out/build.log" >&2; exit 1
    fi
fi
cpu="$clean/build-cpu/runtime/kvaf_cpu_service"
test -x "$cpu"
if "$query_owner"; then
    allowed=()
    if "$row_deps"; then
        allowed+=(':(exclude)csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer/indexer_vector.h')
    fi
    git -C "$ranked_candidate" diff --exit-code --quiet "$ranked_sha" HEAD -- \
        csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer \
        csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer.cpp \
        csrc/operators/common/lifetch "${allowed[@]}"
fi
if [[ -n "$ranked" ]]; then
    printf 'CONFIG URMA physical_NPU=%s logical_device=0 BS16 Q4 656B ranked=%.8s candidate=%.8s KVAF=%.8s\n' "$peer_physical_npu" "$ranked_sha" "$ranked" "$transport"
    if "$row_deps"; then
        echo "CONFIG arms=fixed-order,$plan_control,$plan_candidate query_owners=16x4 on=$plan_control,$plan_candidate boundaries=4 budget=256/1024 fixed_route warmup=4 measured=20"
    elif "$offset_resolver"; then
        echo 'CONFIG arms=fixed-order,chain,offset owners=16x4 boundaries=4 budget=256/1024 fixed_route warmup=4 measured=20'
    elif "$local_resolver"; then
        echo 'CONFIG arms=chain,spill,local owners=16x4 boundaries=4 budget=256/1024 fixed_route warmup=4 measured=20 final_tie=ascending_physical_position'
    elif "$query_owner"; then
        echo 'CONFIG baseline=main_thread_fixed candidate_final_owners=16x4 prefetch_boundaries=4 window_budget=256 total_budget=1024 route=fixed warmup=4 samples=20'
    fi
    # Compile through the same cache key used by run_path. Fail before copying
    # five traces or running the baseline if the toolchain rejects either arm.
    if "$ranked_auto"; then
        printf 'CONTROLS current_fixed=%.8s original=%.8s G2_native_workers=2,32 final_communication_cores=8\n' "$current_fixed" "$ranked_sha"
    elif [[ -n "$help_reference" ]]; then
        printf 'REFERENCE root=%s original=%.8s previous=%.8s profile=4_warmup_20_calls\n' "$help_reference" "$ranked_sha" "${joint_control:-$four_field}"
    fi
    build_sources=("$ranked_candidate")
    if [[ -z "$packet_reference" ]]; then
        if [[ -z "$help_reference" ]] || "$has_anchor" || "$ranked_auto"; then build_sources+=("$ranked_base"); fi
        if "$offset_resolver"; then build_sources+=("$ranked_fixed_order"); fi
    fi
    if "$local_resolver"; then build_sources+=("$ranked_spill" "$ranked_fixed_order"); fi
    for source_root in "${build_sources[@]}"; do
        PYTHONPATH="$source_root/python:$PYTHONPATH" "$PYTHON" -B - "$source_root" "$kvaf/build-npu" "$SOC_VERSION" "$out" <<'BUILD'
from pathlib import Path
import hashlib
import sys
from kvip.runtime import build

source, runtime = map(Path, sys.argv[1:3])
if not Path(build.__file__).resolve().is_relative_to(source.resolve()):
    raise RuntimeError(f"Unexpected build module: {build.__file__}")
cann, headers = build.kvaf_build_paths(runtime)
library, = build.build_libraries(cache_dir=runtime / 'lifetch', cann=cann, headers=headers, soc=sys.argv[3])
with library.open('rb') as handle:
    digest = hashlib.file_digest(handle, 'sha256').hexdigest()
(Path(sys.argv[4]) / f'{source.name}.binary.sha256').write_text(digest + '\n')
print(f"BUILD_READY source={source.name} cache={library.parent.name}", flush=True)
BUILD
    done
elif [[ -n "$adaptive" ]]; then
    printf 'CONFIG G2 physical_NPU=%s logical_device=0 BS16 Q4 656B main=%.8s candidate=%.8s\n' "$peer_physical_npu" "$main_sha" "$adaptive"
    echo 'UNITS us  DETAIL=p50/max[observed/eligible] auxiliary=true  DRIFT=within_graph_last5_minus_first5'
else
    printf 'CONFIG G2 BS16 Q4 656B physical_NPU=%s logical_device=0 after=%s before=%s main=a56154c3 KVAF=%s\n' "$peer_physical_npu" "${new:0:8}" "${old:0:8}" "${transport:0:8}"
fi
run_arm() (
    set -euo pipefail
    label=$1
    group=${2:-2}
    arm="${3:-$out}/$label"; mkdir -p "$arm"
    port=$("$PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
    "${cpu_command[@]}" --listen 127.0.0.1 --port "$port" --memory-page-mode thp >"$arm/cpu.log" 2>&1 &
    pid=$!
    monitor=
    cleanup() {
        if [[ -n "$monitor" ]]; then kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true; fi
        kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
    }
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    for ((i=0; i<30; i++)); do
        kill -0 "$pid" 2>/dev/null || break
        if grep -Fq "LISTEN address=127.0.0.1 port=$port" "$arm/cpu.log"; then break; fi
        sleep 1
    done
    if ! kill -0 "$pid" 2>/dev/null || ! grep -Fq "LISTEN address=127.0.0.1 port=$port" "$arm/cpu.log"; then
        tail -6 "$arm/cpu.log" >&2; exit 1
    fi
    "$PYTHON" -B "$tools/support/peer_thp.py" --timeout 1800 --pid "$pid" --log "$arm/cpu.log" --output "$arm/thp.json" >"$arm/thp.log" 2>&1 &
    monitor=$!
    level=1
    case "$label" in
        fixed-order) mode=(--fused-only --production-root "$ranked_fixed_order") ;;
        chain) mode=(--fused-only --production-root "$ranked_base") ;;
        spill) mode=(--fused-only --production-root "$ranked_spill") ;;
        offset|row-deps|li-packets)
            source_root=$ranked_candidate
            if [[ $label == "$plan_control" ]]; then source_root=$ranked_base; fi
            mode=(--fused-only --production-root "$source_root") ;;
        local) mode=(--fused-only --production-root "$ranked_candidate") ;;
        fixed) mode=(--fused-only --production-root "$ranked_base" --ranked-q-workers 2); level=2 ;;
        auto) mode=(--fused-only --production-root "$ranked_candidate" --ranked-q-workers 2 --ranked-auto-route); level=2 ;;
        serial2|serial32) mode=(--serial-only --serial-workers "${label#serial}") ;;
        baseline|anchor) mode=(--fused-only --production-root "$ranked_base"); level=2 ;;
        candidate) mode=(--fused-only --production-root "$ranked_candidate"); level=2
                   if "$joint_archive"; then mode+=(--ranked-q-workers "$ranked_q_workers"); fi ;;
        serial*) mode=(--serial-only) ;;
        main) mode=(--fused-only)
              if [[ -n "$adaptive" ]]; then mode+=(--route-policy main); level=2; fi ;;
        zero|early) mode=(--fused-only --route-policy "$label" --candidate-root "$adaptive_src"); level=2 ;;
        before*) candidate=$old ;;
        after) candidate=$new ;;
    esac
    limits=(--prefetch-tokens 1024 --intermediate-fetches 4)
    if "$query_owner"; then
        mode+=(--ranked-q-workers 2 --query-owner-arm "$label")
        level=1
    fi
    if [[ "$label" == before* || "$label" == after ]]; then
        limits=(--prefetch-tokens 512 --intermediate-fetches 8)
        mode=(--fused-only --candidate-root "$src/auto-${candidate:0:8}" --candidate-snapshot "$snapshot")
    fi
    printf 'CASE_BEGIN G%s %s\n' "$group" "$label"
    if ! "$PYTHON" -B "$tools/test_lifetch.py" diagnose "${mode[@]}" "${limits[@]}" \
        --transport urma --input real --trace-root "$trace" --group "$group" --round-range 1:1 \
        --device 0 --kvaf-root "$kvaf" --kvaf-server "127.0.0.1:$port" \
        --prefetch-rank-start 1024 --warmup 4 --iterations 20 --level "$level" \
        --output "$arm/results" --timeout-seconds 1800 >"$arm/replay.log" 2>&1; then
        if "$query_owner"; then
            printf 'FAILED_ARM=G%s/%s WORKER_DIR=%s\n' "$group" "$label" "$arm/results" >&2
            awk '/^FAIL:|^REPLAY CHECK FAILURE|^STREAMING FETCH KV AUDIT:/ {
                print substr($0,1,700); if (++n == 4) exit
            }' "$arm/replay.log" >&2
            find "$arm/results" -name worker.log -exec awk '
                /^EAGER_STAGE / {stage=$0}
                /^DEVICE_FAILURE / {failure=$0}
                /^(AssertionError|ValueError|RuntimeError|IndexError|KeyError|TypeError):/ {exception=$0}
                /fault kernel|faultKernel|kernel_name|507014|Physical cutoff tie rule|Replay TopK ranking differs/ && ++n<=3 {print substr($0,1,600)}
                END {if(stage) print stage; if(failure) print failure; if(exception) print substr(exception,1,700)}
            ' {} \; >&2
            tail -5 "$arm/replay.log" >&2
        else
            tail -5 "$arm/replay.log" >&2
            find "$arm/results" -name worker.log -exec tail -10 {} \; >&2
        fi
        exit 1
    fi
    if ! wait "$monitor"; then cat "$arm/thp.log" >&2; exit 1; fi
    monitor=
    if [[ -n "$ranked" ]]; then exit 0; fi
    if [[ -z "$adaptive" ]]; then
        "$PYTHON" -B "$tools/support/peer_delta.py" "$out" "$label"
    else
        if [[ "$label" == serial ]]; then
            "$PYTHON" -B "$tools/support/peer_delta.py" "$out" serial --local-baseline >"$out/serial-screen.txt"
        fi
        "$PYTHON" -B "$tools/support/main_route_report.py" "$out" --arm "$label"
    fi
)
bind_cpu_service() {
    local node=$1 cpus allowed
    policy=inherited
    cpu_command=("$cpu")
    [[ -n "$node" ]] || return 0
    if command -v numactl >/dev/null && numactl "--membind=$node" true >/dev/null 2>&1; then
        cpu_command=(numactl "--membind=$node" "$cpu")
        policy=membind
    elif command -v taskset >/dev/null && [[ -r /sys/devices/system/node/node${node}/cpulist ]]; then
        # Prefer local first touch if the container forbids set_mempolicy.
        cpus=$(<"/sys/devices/system/node/node${node}/cpulist")
        allowed=$("$PYTHON" -c 'import os,sys
allowed = os.sched_getaffinity(0)
ids = set()
for item in sys.argv[1].split(","):
    lo, _, hi = item.partition("-")
    ids.update(range(int(lo), int(hi or lo) + 1))
print(",".join(str(n) for n in sorted(ids & allowed)))' "$cpus")
        if [[ -n "$allowed" ]] && taskset -c "$allowed" true >/dev/null 2>&1; then
            cpu_command=(taskset -c "$allowed" "$cpu")
            policy=local_first_touch
        fi
    fi
}
cd "$root"
node=
if "$local_baseline" && [[ -z "$help_reference" ]]; then
    # Keep even the first arm on one NUMA node when the container allows it.
    node=$("$PYTHON" -c 'import ctypes
from pathlib import Path
cpu = ctypes.CDLL(None).sched_getcpu()
nodes = list(Path(f"/sys/devices/system/cpu/cpu{cpu}").glob("node[0-9]*"))
print(nodes[0].name[4:] if len(nodes) == 1 else "")')
fi
if "$joint_local"; then
    if [[ -f "$out/numa_target.txt" ]]; then
        node=$(<"$out/numa_target.txt")
    else
        printf '%s\n' "$node" >"$out/numa_target.txt"
    fi
fi
bind_cpu_service "$node"
if [[ -n "$help_reference" ]] || "$joint_local"; then
    source "$tools/support/ranked_help_run.sh"
    if "$ranked_auto"; then
        auto_order=(fixed auto)
        for group in "${ranked_groups[@]}"; do
            case_dir="$out/g$group"
            mkdir -p "$case_dir"
            if [[ -s "$case_dir/summary.txt" ]]; then cat "$case_dir/summary.txt"; continue; fi
            params=$(ranked_reference_case "$group")
            preferred=${params%%$'\n'*}
            reference_trace=${params#*$'\n'}
            bind_cpu_service "$preferred"
            trace=$(PEER_RANKED_INPUT="$reference_trace" PEER_RANKED_HASH_FILE="$case_dir/input.sha256" \
                bash "$tools/peer_g2_input.sh" "$work" "$group")
            cmp -s "$case_dir/input.sha256" "$help_reference/g$group/input.sha256" || { echo "Changed G$group input" >&2; exit 1; }
            arms=("${auto_order[@]}")
            if [[ $group == 2 ]]; then arms=(serial2 "${arms[@]}" serial32); fi
            for arm in "${arms[@]}"; do
                if [[ ! -f "$case_dir/$arm/pass" ]]; then
                    if [[ -e "$case_dir/$arm" ]]; then
                        prior=$(mktemp -d "$case_dir/interrupted.XXXXXX")
                        mv "$case_dir/$arm" "$prior/"
                    fi
                    run_arm "$arm" "$group" "$case_dir"
                    touch "$case_dir/$arm/pass"
                fi
            done
            "$PYTHON" -B "$tools/support/ranked_auto_report.py" "$out" --reference "$help_reference" --group "$group" >"$case_dir/summary.tmp"
            mv "$case_dir/summary.tmp" "$case_dir/summary.txt"
            cat "$case_dir/summary.txt"
            auto_order=("${auto_order[1]}" "${auto_order[0]}")
        done
        echo "REPLAY_COMPLETE RESULT_ROOT=$out"
    else
        ranked_help_groups
    fi
    exit 0
fi
if [[ -n "$ranked" ]]; then
    printf 'CPU_PLACEMENT target_node=%s policy=%s actual=measured_per_arm\n' "${node:-unknown}" "$policy"
    order=(baseline candidate)
    groups=(2 7 10 17 4)
    if "$offset_resolver"; then order=(chain offset fixed-order); groups=("${offset_groups[@]}"); fi
    if "$row_deps"; then order=("$plan_control" "$plan_candidate" fixed-order); fi
    if [[ -n "$packet_reference" ]]; then
        order=("$plan_candidate")
        echo "CONFIG archived_controls=fixed-order,row-deps measured=$plan_candidate reference=$packet_reference"
    fi
    if "$local_resolver"; then order=(chain spill local); groups=(7 2 10 17 4); fi
    for group in "${groups[@]}"; do
        case_dir="$out/g$group"
        if "$query_owner"; then trace=${owner_traces[$group]}
        else trace=$(bash "$tools/peer_g2_input.sh" "$work" "$group"); fi
        # The two historically ambiguous groups also check the fixed fused tail.
        if "$local_resolver" && [[ $group == 7 || $group == 4 ]]; then
            run_arm fixed-order "$group" "$case_dir"
        fi
        for label in "${order[@]}"; do
            run_arm "$label" "$group" "$case_dir"
        done
        if "$query_owner"; then
            "$PYTHON" -B "$tools/support/query_owner_report.py" "$out" "$group" | tee -a "$out/summary.txt"
        else
            "$PYTHON" -B "$tools/support/ranked_five_report.py" "$out" --group "$group" | tee -a "$out/summary.txt"
        fi
        order=("${order[@]:1}" "${order[0]}")
    done
    printf 'REPLAY_COMPLETE RESULT_ROOT=%s\n' "$out"
    exit 0
fi
if [[ -n "$reuse" ]]; then
    # Keep the original profile, timestamps and logs available in both reports.
    ln -s "$reuse" "$out/serial"
    if [[ -z "$adaptive" ]]; then
        "$PYTHON" -B "$tools/support/peer_delta.py" "$out" serial
    else
        "$PYTHON" -B "$tools/support/peer_delta.py" "$out" serial >"$out/serial-screen.txt"
        "$PYTHON" -B "$tools/support/main_route_report.py" "$out" --arm serial
    fi
    printf 'REUSED serial=%s\n' "$reuse"
else
    run_arm serial
fi
if [[ "$policy" == inherited ]]; then
    node=$("$PYTHON" -B "$tools/support/peer_delta.py" "$out" --serial-node)
    bind_cpu_service "$node"
fi
printf 'CPU_PLACEMENT target_node=%s policy=%s actual=measured_per_arm\n' "${node:-unknown}" "$policy"
if [[ -n "$adaptive" ]]; then
    for label in main zero early; do run_arm "$label"; done
    "$PYTHON" -B "$tools/support/main_route_report.py" "$out" --gains-only
else
    for label in main before after before-end serial-end; do run_arm "$label"; done
    "$PYTHON" -B "$tools/support/peer_delta.py" "$out"
fi
