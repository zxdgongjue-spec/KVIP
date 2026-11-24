#!/usr/bin/env bash
# Fresh five-case replay of the unified-KVAF KVIP main branch on physical NPU4.
set -eo pipefail
if (( $# != 3 )) || [[ ! $2 =~ ^[a-f0-9]{40}$ || $3 != d6cfd392f722dd3f96ef6e05175f503aac511dbb ]]; then
    echo "Usage: bash unified_main_npu4.sh KVIP_REPO READER_SHA d6cfd392f722dd3f96ef6e05175f503aac511dbb" >&2
    exit 2
fi
repo=$1
reader=$2
main=$3
sdk=813f1750a50b7e6c92d5f5397f2fbc05befef6da
patched_tree=1de18e8315137d7a0227c3ae4c82afd45b5fe743
work=${PEER_G2_WORK:-/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold}
area="$work/peer-baseline"
sources="$work/fused-scan-sources"
mkdir -p "$area" "$sources"
out=$(mktemp -d "$area/unified-main-five.XXXXXX")
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
exec 8>"$sources/unified-sdk-$main.lock"
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
entry="$reader_src/tests/operators/lifetch"
support="$entry/support"
printf '%s\n' "$main" >"$out/main.txt"
printf '%s\n' "$reader" >"$out/reader.txt"
printf '%s\n' "$sdk" >"$out/kvaf.txt"
printf '4\n' >"$out/physical_npu.txt"
cat /proc/sys/kernel/hostname >"$out/host.txt"
cat /proc/sys/kernel/random/boot_id >"$out/boot_id.txt"
if [[ -r /etc/machine-id ]]; then cat /etc/machine-id >"$out/machine_id.txt"; fi

# Reuse this machine's production Python, LIDU and URMA toolchain. The old SDK
# is only the local Git object source and supplies no runtime objects to this arm.
clean="$area/KVAF-de4fd37e"
baseline="$work/adaptive-sources/peer-serial-da318bdb"
peer_physical_npu=4
source "$support/peer_node_prepare.sh"
peer_prepare_node "$repo" "$baseline" "$clean" "$work" "$out" full
sdk_src="$sources/unified-sdk-${sdk:0:8}-${main:0:8}"
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
lidu="$baseline/csrc/operators/lidu"
vendors=("$lidu"/_custom_opp/vendors/*/op_api/lib/libcust_opapi.so)
[[ ${#vendors[@]} == 1 && -s ${vendors[0]} ]]
export ASCEND_CUSTOM_OPP_PATH="${vendors[0]%/op_api/lib/libcust_opapi.so}${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export PYTHONPATH="$main_src/python:$sdk_src/build-npu/python:$lidu/torch_extension${PYTHONPATH:+:$PYTHONPATH}"
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
echo "CONFIG measured=unified_main native=$main sdk=$sdk NPU=4 BS=16 Q=4 token_bytes=656 warmup=4 calls=20"
run_group() (
    set -euo pipefail
    g=$1
    owner="$area/fused.N7n8ZP"
    if [[ $g == 2 ]]; then owner="$area/fused.gbxKC9"; fi
    arm="$out/g$g/mainline"
    mkdir -p "$arm"
    trace=$(bash "$entry/peer_g2_input.sh" "$work" "$g")
    node=$("$PYTHON" -B "$support/li_window_report.py" numa "$area/main-five.hF6qQ8/g$g/mainline/thp.json")
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
    "$PYTHON" -B "$support/peer_thp.py" --unified-sdk --pid "$server" --log "$arm/cpu.log" --output "$arm/thp.json" --timeout 1800 >"$arm/thp.log" 2>&1 &
    monitor=$!
    echo "CASE_BEGIN G$g unified_main"
    (cd "$arm" && timeout 1800 "$PYTHON" -B "$support/main_five_run.py" --main "$main_src" --revision "$main" \
         --trace "$trace" --group "$g" --owner-archive "$owner" --kvaf-root "$sdk_src" \
         --kvaf-server "127.0.0.1:$port" --output "$arm") >"$arm/worker.log" 2>&1 || {
        echo "FAILED_ARM=G$g/unified_main WORKER_LOG=$arm/worker.log" >&2
        tail -30 "$arm/worker.log" >&2
        exit 1
    }
    if ! wait "$monitor"; then cat "$arm/thp.log" >&2; exit 1; fi
    monitor=
    PYTHONPATH="$support:$PYTHONPATH" "$PYTHON" -B - "$out" "$owner" "$g" "$main" <<'PY_REPORT'
import sys
from pathlib import Path
from main_five_report import archived_oracle, case_folder, measured_parallel, read

root, owner = map(Path, sys.argv[1:3])
group, revision = int(sys.argv[3]), sys.argv[4]
ids, receipt = archived_oracle(owner, group)
source = read(case_folder(owner, group, 'logical-ties').parent / 'replay_input.json')
timing, exact, _, _, _ = measured_parallel(root, group, 'mainline', revision, owner,
                                          ids, receipt, source, verify_library=True)
print(f'G{group} CHECK arm=unified_main TopK_score_vs_strong_serial=PASS calls=20/20 '
      f'queries=1280/1280 exact_ID_sets={exact.sum()}/1280 KV=PASS ordered_Index=PASS rows=320/320')
print(f'G{group} unified_main', *(f'{key}_mean/P50_us={value[0]:.2f}/{value[1]:.2f}'
                                  for key, value in timing.items()))
PY_REPORT
)
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
echo "REPLAY_COMPLETE RESULT_ROOT=$out"
