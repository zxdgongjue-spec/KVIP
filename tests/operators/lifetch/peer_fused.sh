#!/usr/bin/env bash
# Replay the captured inputs with a single transport build and timing protocol.
set -eo pipefail
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
if [[ ${1:-} == --ranked-index ]]; then
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
    echo 'Usage: peer_fused.sh --ranked-index CANDIDATE_SHA | [--main-adaptive CANDIDATE_SHA] [--reuse-serial PREVIOUS_RESULT_ROOT]' >&2
    exit 2
fi
mkdir -p "$area"
out=$(mktemp -d "$area/fused.XXXXXX")
printf 'RESULT_ROOT=%s\n' "$out"
export PATH=/usr/local/python3.12.13/bin:$PATH
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
    peer_prepare_node "$root" "$baseline" "$clean" "$work" "$out" "${ranked:+fused-only}"
fi
if [[ -z "$ranked" ]]; then
    pin "$root" "$baseline" "$serial_revision"
    git -C "$root" diff --exit-code --quiet "$serial_revision" HEAD -- csrc/operators/lidu
fi
if [[ -n "$ranked" ]]; then
    git -C "$root" fetch -q origin codex/lifetch-ranked-integration perf/ranked-all-aiv-index
    git -C "$root" merge-base --is-ancestor "$ranked_sha" "$ranked"
    ranked_base="$src/ranked-${ranked_sha:0:8}"
    ranked_candidate="$src/ranked-${ranked:0:8}"
    pin "$root" "$ranked_base" "$ranked_sha"
    pin "$root" "$ranked_candidate" "$ranked"
    printf '%s\n' "$ranked_sha" >"$out/baseline.txt"
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
git -C "$clean" fetch -q origin test/peer-main-urma-fused-context
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
export ASCEND_RT_VISIBLE_DEVICES=4 SOC_VERSION=Ascend950DT_9582 ASCEND_LAUNCH_BLOCKING=0
export MAX_JOBS=2 OMP_NUM_THREADS=1
venv="$area/python-${serial_revision:0:8}"
export PATH="$venv/bin:$PATH" PYTHON="$venv/bin/python"
test -x "$PYTHON"
lidu="$baseline/csrc/operators/lidu"
export PYTHONPATH="$root/python:$kvaf/build-npu/python${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "$ranked" ]]; then
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
if [[ -n "$ranked" ]]; then
    printf 'CONFIG URMA NPU4 BS16 Q4 656B ranked=%.8s candidate=%.8s KVAF=%.8s\n' "$ranked_sha" "$ranked" "$transport"
elif [[ -n "$adaptive" ]]; then
    printf 'CONFIG G2 NPU4 BS16 Q4 656B main=%.8s candidate=%.8s\n' "$main_sha" "$adaptive"
    echo 'UNITS us  DETAIL=p50/max[observed/eligible] auxiliary=true  DRIFT=within_graph_last5_minus_first5'
else
    printf 'CONFIG G2 BS16 Q4 656B NPU4 after=%s before=%s main=a56154c3 KVAF=%s\n' "${new:0:8}" "${old:0:8}" "${transport:0:8}"
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
        baseline) mode=(--fused-only --production-root "$ranked_base"); level=2 ;;
        candidate) mode=(--fused-only --production-root "$ranked_candidate"); level=2 ;;
        serial*) mode=(--serial-only) ;;
        main) mode=(--fused-only)
              if [[ -n "$adaptive" ]]; then mode+=(--route-policy main); level=2; fi ;;
        zero|early) mode=(--fused-only --route-policy "$label" --candidate-root "$adaptive_src"); level=2 ;;
        before*) candidate=$old ;;
        after) candidate=$new ;;
    esac
    limits=(--prefetch-tokens 1024 --intermediate-fetches 4)
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
        tail -5 "$arm/replay.log" >&2
        find "$arm/results" -name worker.log -exec tail -10 {} \; >&2
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
if "$local_baseline"; then
    # Keep even the first arm on one NUMA node when the container allows it.
    node=$("$PYTHON" -c 'import ctypes
from pathlib import Path
cpu = ctypes.CDLL(None).sched_getcpu()
nodes = list(Path(f"/sys/devices/system/cpu/cpu{cpu}").glob("node[0-9]*"))
print(nodes[0].name[4:] if len(nodes) == 1 else "")')
fi
bind_cpu_service "$node"
if [[ -n "$ranked" ]]; then
    printf 'CPU_PLACEMENT target_node=%s policy=%s actual=measured_per_arm\n' "${node:-unknown}" "$policy"
    order=(baseline candidate)
    for group in 2 7 10 17 4; do
        case_dir="$out/g$group"
        trace=$(bash "$tools/peer_g2_input.sh" "$work" "$group")
        for label in "${order[@]}"; do
            run_arm "$label" "$group" "$case_dir"
        done
        "$PYTHON" -B "$tools/support/ranked_five_report.py" "$out" --group "$group" | tee -a "$out/summary.txt"
        order=("${order[1]}" "${order[0]}")
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
