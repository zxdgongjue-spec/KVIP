#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Five real Q4 traces, four native batch sizes, paired with the strong two-worker serial path.
set -eo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
work=${PEER_G2_WORK:-/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold}
area="$work/peer-baseline"
sources="$work/fused-scan-sources"
mkdir -p "$area" "$sources"
out=$(mktemp -d "$area/largebs-q4.XXXXXX")
echo "RESULT_ROOT=$out"
log="$out/run.log"
# A single transcript for setup/driver errors, with compact results on stdout.
exec 3>&1
exec >"$log" 2>&1
trap 'rc=$?; if ((rc)); then tail -25 "$log" >&3; printf "FAILED rc=%s LOG=%s\n" "$rc" "$log" >&3; fi' EXIT
export PEER_LOCAL_ONLY=1 ASCEND_RT_VISIBLE_DEVICES=4 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
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
exec 9>"$sources/largebs-$(</proc/sys/kernel/random/boot_id)-npu4.lock"
flock -n 9 || { echo 'A large-batch replay is already running on NPU4'; exit 1; }
exec 8>"$sdk.lock"
flock 8
peer_physical_npu=4
baseline="$work/adaptive-sources/peer-serial-da318bdb"
clean="$area/KVAF-de4fd37e"
peer_prepare_node "$root" "$baseline" "$clean" "$work" "$out"
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
export PEER_PHYSICAL_NPU=4
printf '%s\n' "$(git -C "$root" rev-parse HEAD)" > "$out/native.txt"
hostname > "$out/host.txt"
if [[ -r /etc/machine-id ]]; then cat /etc/machine-id > "$out/machine_id.txt"; fi
printf '4\n' > "$out/physical_npu.txt"
# Use the same registered CPU memory placement for the serial and fused arms.
placement=$("$PYTHON" -B "$workers/large_batch_placement.py" find "$out")
read -r node adapter <<<"$placement"
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
if [[ -n $ub_device && $ub_device != auto ]]; then command+=(--ub-device "$ub_device"); fi
run_arm() (
    set -euo pipefail
    local group=$1 batch=$2 trace=$3 arm=$4 pair=$5
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
    args=(--trace "$trace" --group "$group" --batch "$batch" --arm "$arm" --kvaf-root "$sdk"
          --kvaf-server "127.0.0.1:$port" --output "$dir")
    if [[ $arm == fused ]]; then args+=(--serial-result "$pair/serial"); fi
    (cd "$dir" && timeout --kill-after=30s 7200s "$PYTHON" -B "$workers/large_batch_pair.py" "${args[@]}") >"$dir/worker.log" 2>&1 || { tail -20 "$dir/worker.log"; exit 1; }
    wait "$monitor"
    monitor=
)
for group in 10 4 17 7 2; do
    trace=$(bash "$helpers/peer_g2_input.sh" "$work" "$group")
    for batch in 20 24 28 32; do
        pair="$out/g$group/bs$batch"
        mkdir -p "$pair"
        echo "CASE_BEGIN G$group BS$batch" >&3
        run_arm "$group" "$batch" "$trace" serial "$pair"
        run_arm "$group" "$batch" "$trace" fused "$pair"
        "$PYTHON" -B "$workers/large_batch_placement.py" compare "$pair/serial/thp.json" "$pair/fused/thp.json"
        for arm in serial fused; do
            grep -E '^(CHECK|TIMING|SAVED) ' "$pair/$arm/worker.log" | tee -a "$out/summary.txt" >&3
            rm -f -- "$pair/$arm/worker.log" "$pair/$arm/cpu.log"
        done
    done
done
rm -rf -- "$host_tools"
# The per-arm JSON and placement records contain the measurement. Setup and
# import transcripts in this new output directory are only needed on failure.
rm -f -- "$out"/{run,venv,pybind11,lidu-import,build-lidu,build-cpu}.log
echo "REPLAY_COMPLETE pairs=20 RESULT_ROOT=$out" >&3
