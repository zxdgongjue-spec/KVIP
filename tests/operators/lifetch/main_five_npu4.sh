#!/usr/bin/env bash
# B24: replay upstream main and compare with the on-host production archives.
set -eo pipefail
repo=${1:?supply the local KVIP repository}
reader=${2:?supply the full replay reader revision}
main_rev=609fa26fd715f9409bbe74e1b2d3685c58dfb0cf
owner_rev=888db6b1603d22f4142bd2ccffe0518611856a31
transport=a2857b460a74704114fd30a849ae38666cc2c372
serial=da318bdb81344672b5d9013e04d517669d3eb142
peer_physical_npu=4
export PEER_PHYSICAL_NPU=4 ASCEND_RT_VISIBLE_DEVICES=4 PEER_LOCAL_ONLY=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PATH="/usr/local/python3.12.13/bin:$PATH"
unset PEER_RANKED_INPUT PEER_RANKED_HASH_FILE
work=${PEER_G2_WORK:-/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold}
area="$work/peer-baseline"
src="$work/fused-scan-sources"
mkdir -p "$src" "$area"
out=$(mktemp -d "$area/main-five.XXXXXX")
echo "RESULT_ROOT=$out"
exec > >(tee "$out/run.log") 2>&1

checkout() {
    local name=$1 rev=$2 dest="$src/$1-$2"
    git -C "$repo" cat-file -e "$rev^{commit}" || return
    if [[ -e "$dest" ]]; then
        if [[ $(git -C "$dest" rev-parse HEAD 2>/dev/null) == "$rev" &&
              -z $(git -C "$dest" status --porcelain --untracked-files=normal) ]]; then
            printf '%s\n' "$dest"
            return
        fi
        dest=$(mktemp -d "$src/$name-$rev.XXXXXX") || return
    fi
    git -C "$repo" worktree add -q --detach "$dest" "$rev" || return
    printf '%s\n' "$dest"
}
reader_src=$(checkout reader "$reader")
main_src=$(checkout upstream "$main_rev")
support="$reader_src/tests/operators/lifetch/support"
printf '%s\n' "$main_rev" >"$out/main.txt"
printf '%s\n' "$reader" >"$out/reader.txt"
printf '%s\n' "$transport" >"$out/kvaf.txt"
printf '4\n' >"$out/physical_npu.txt"
cat /proc/sys/kernel/hostname >"$out/host.txt"
cat /proc/sys/kernel/random/boot_id >"$out/boot_id.txt"
if [[ -r /etc/machine-id ]]; then cat /etc/machine-id >"$out/machine_id.txt"; fi
echo "CONFIG main=$main_rev reader=$reader NPU=4 transport=urma BS=16 Q=4 token_bytes=656 warmup=4 calls=20"
groups=(10 4 17 7 2)
archive_for() {
    if [[ $1 == 2 ]]; then printf '%s\n' "$area/fused.gbxKC9"
    else printf '%s\n' "$area/fused.N7n8ZP"; fi
}
for group in "${groups[@]}"; do
    archive=$(archive_for "$group")
    if [[ ! -r "$archive/logical-ties.txt" || $(<"$archive/logical-ties.txt") != "$owner_rev" ||
          ! -r "$archive/physical_npu.txt" || $(<"$archive/physical_npu.txt") != 4 ||
          ! -s "$archive/g$group/logical-ties/thp.json" ]]; then
        echo "Archived B24 owner TopK result missing or incompatible: G$group $archive" >&2
        exit 1
    fi
done
if [[ -r /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
else
    source /usr/local/Ascend/cann-9.2.0/set_env.sh
fi
if [[ -r /usr/local/Ascend/nnal/atb/set_env.sh ]]; then source /usr/local/Ascend/nnal/atb/set_env.sh; fi
set -euo pipefail
clean="$area/KVAF-de4fd37e"
kvaf="$area/KVAF-${transport:0:8}"
baseline="$work/adaptive-sources/peer-serial-${serial:0:8}"
source "$support/peer_node_prepare.sh"
# The upstream test driver imports the registered LIDU extension during startup.
peer_prepare_node "$repo" "$baseline" "$clean" "$work" "$out" full
peer_node_pin "$clean" "$kvaf" "$transport" test/peer-main-urma-fused-context
export PATH="${PYTHON%/*}:$PATH"
lidu="$baseline/csrc/operators/lidu"
vendors=("$lidu"/_custom_opp/vendors/*/op_api/lib/libcust_opapi.so)
[[ ${#vendors[@]} == 1 && -s ${vendors[0]} ]]
export ASCEND_CUSTOM_OPP_PATH="${vendors[0]%/op_api/lib/libcust_opapi.so}${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export PYTHONPATH="$main_src/python:$kvaf/build-npu/python:$lidu/torch_extension${PYTHONPATH:+:$PYTHONPATH}"
export KVAF_BATCH_GET_KERNEL="$kvaf/build-npu/bin/kvaf_execute_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$kvaf/build-npu/bin/kvaf_execute_batch_put_kernels.o"
suffix=$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
if [[ ! -s "$KVAF_BATCH_GET_KERNEL" || ! -s "$KVAF_BATCH_PUT_KERNEL" ||
      ! -s "$kvaf/build-npu/python/kvaf/_kvaf$suffix" ]]; then
    peer_node_logged "$out/build-kvaf.log" bash "$kvaf/build.sh" npu --without-aicpu --jobs 2
fi
cpu="$clean/build-cpu/runtime/kvaf_cpu_service"

run_group() (
    set -euo pipefail
    group=$1
    archive=$(archive_for "$group")
    arm="$out/g$group/mainline"
    mkdir -p "$arm"
    trace=$(bash "$reader_src/tests/operators/lifetch/peer_g2_input.sh" "$work" "$group")
    node=$("$PYTHON" -B "$support/owner_pack_tail.py" numa "$archive" "$group")
    command=("$cpu")
    if [[ -n "$node" ]]; then
        if command -v numactl >/dev/null && numactl "--membind=$node" true >/dev/null 2>&1; then
            command=(numactl "--membind=$node" "$cpu")
        elif command -v taskset >/dev/null && [[ -r /sys/devices/system/node/node${node}/cpulist ]]; then
            cpus=$(<"/sys/devices/system/node/node${node}/cpulist")
            if taskset -c "$cpus" true >/dev/null 2>&1; then command=(taskset -c "$cpus" "$cpu"); fi
        fi
    fi
    port=$("$PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
    "${command[@]}" --listen 127.0.0.1 --port "$port" --memory-page-mode thp >"$arm/cpu.log" 2>&1 &
    server_pid=$!
    monitor=
    cleanup() {
        if [[ -n "$monitor" ]]; then kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true; fi
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    }
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    for ((i=0; i<30; i++)); do
        if grep -Fq "LISTEN address=127.0.0.1 port=$port" "$arm/cpu.log"; then break; fi
        kill -0 "$server_pid" 2>/dev/null || break
        sleep 1
    done
    if ! grep -Fq "LISTEN address=127.0.0.1 port=$port" "$arm/cpu.log"; then tail -15 "$arm/cpu.log"; exit 1; fi
    "$PYTHON" -B "$support/peer_thp.py" --pid "$server_pid" --log "$arm/cpu.log" --output "$arm/thp.json" --timeout 1800 >"$arm/thp.log" 2>&1 &
    monitor=$!
    echo "CASE_BEGIN G$group mainline_parallel_tail"
    if ! (cd "$arm" && timeout 1800 "$PYTHON" -B "$support/main_five_run.py" \
        --main "$main_src" --trace "$trace" --group "$group" --owner-archive "$archive" \
        --kvaf-root "$kvaf" --kvaf-server "127.0.0.1:$port" --output "$arm") >"$arm/worker.log" 2>&1; then
        echo "FAILED_ARM=G$group/mainline WORKER_LOG=$arm/worker.log" >&2
        tail -30 "$arm/worker.log" >&2
        exit 1
    fi
    if ! wait "$monitor"; then cat "$arm/thp.log" >&2; exit 1; fi
    monitor=
    "$PYTHON" -B "$support/main_five_report.py" "$out" "$group" "$archive" | tee -a "$out/summary.txt"
)
for group in "${groups[@]}"; do run_group "$group"; done
echo "REPLAY_COMPLETE RESULT_ROOT=$out"
