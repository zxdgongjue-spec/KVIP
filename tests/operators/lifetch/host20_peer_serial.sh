#!/usr/bin/env bash
# Reproduce the G2 reference under an isolated KVIP/KVAF build on physical NPU4.
set -eo pipefail
kip=$(cd "$(dirname "$0")/../../.." && pwd)
tools="$kip/tests/operators/lifetch"
work=/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold
kvaf_sha=de4fd37e02fa86357f71fe4fcbbbecc3e75d1d58
area="$work/peer-baseline"
kvaf="$area/KVAF-${kvaf_sha:0:8}"
mkdir -p "$area"
out=$(mktemp -d "$area/g2.XXXXXX")
printf 'RESULT_ROOT=%s\n' "$out"
export PATH=/usr/local/python3.12.13/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -euo pipefail
trace=$(bash "$tools/peer_g2_input.sh" "$work")
test -d "$trace" || { echo "Invalid prepared input: $trace" >&2; exit 1; }
if [[ ! -d "$kvaf" ]]; then
    if [[ "$(git -C "$work/KVAF" remote get-url origin 2>/dev/null || true)" == https://gitcode.com/CacheOS/KVAF* ]]; then
        git -C "$work/KVAF" fetch -q origin main
        git -C "$work/KVAF" worktree add -q --detach "$kvaf" "$kvaf_sha"
    else
        git clone -q --single-branch --branch main --no-checkout https://gitcode.com/CacheOS/KVAF.git "$kvaf"
        git -C "$kvaf" checkout -q --detach "$kvaf_sha"
    fi
fi
[[ $(git -C "$kvaf" rev-parse HEAD) == "$kvaf_sha" ]]
git -C "$kvaf" diff --quiet
git -C "$kvaf" diff --cached --quiet
git -C "$kip" diff --quiet
git -C "$kip" diff --cached --quiet
cache="$work/KVAF/build-npu/CMakeCache.txt"
cann=$(sed -n 's/^ASCEND_CANN_PACKAGE_PATH:[^=]*=//p' "$cache")
test -d "$cann/include/acl"
export ASCEND_HOME_PATH="$cann" ASCEND_CANN_PACKAGE_PATH="$cann"
export ASCEND_RT_VISIBLE_DEVICES=4 SOC_VERSION=Ascend950DT_9582 ASCEND_LAUNCH_BLOCKING=0
export MAX_JOBS=2 OMP_NUM_THREADS=1 NANOVLLM_A5_OPS_BUILD_JOBS=2
# Keep LIDU's editable path registration private to this reference build.
venv="$area/python-$(git -C "$kip" rev-parse --short=8 HEAD)"
if [[ ! -x "$venv/bin/python" ]]; then
    /usr/local/python3.12.13/bin/python3 -m venv --system-site-packages "$venv"
fi
export PATH="$venv/bin:$PATH" PYTHON="$venv/bin/python" NANOVLLM_A5_OPS_PYTHON="$venv/bin/python"
export KVAF_BATCH_GET_KERNEL="$kvaf/build-npu/bin/kvaf_execute_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$kvaf/build-npu/bin/kvaf_execute_batch_put_kernels.o"
lidu="$kip/csrc/operators/lidu"
export PYTHONPATH="$lidu/torch_extension:$kip/python:$kvaf/build-npu/python${PYTHONPATH:+:$PYTHONPATH}"
oldcpu="$work/KVAF/build-cpu/CMakeCache.txt"
if [[ -f "$oldcpu" ]]; then
    umdk=$(sed -n 's/^KVAF_URMA_ROOT:[^=]*=//p' "$oldcpu")
    lib=$(sed -n 's/^KVAF_URMA_LIBRARY:[^=]*=//p' "$oldcpu")
    if [[ -f "$umdk/include/urma_api.h" ]]; then export KVAF_UMDK_ROOT="$umdk"; fi
    if [[ -f "$lib" ]]; then export KVAF_URMA_LIBRARY="$lib"; fi
fi
run_logged() {
    local log=$1
    shift
    if "$@" >"$log" 2>&1; then return 0; else
        local rc=$?
        printf 'FAILED rc=%s LOG=%s\n' "$rc" "$log" >&2
        tail -12 "$log" >&2
        return "$rc"
    fi
}
printf 'SOURCE KVIP=%s KVAF=%s physical_NPU=4 logical=0\n' "$(git -C "$kip" rev-parse HEAD)" "$kvaf_sha"
run_logged "$out/build-cpu.log" bash "$kvaf/build.sh" cpu --jobs 2
run_logged "$out/build-npu.log" bash "$kvaf/build.sh" npu --without-aicpu --jobs 2
if [[ ! -f "$venv/lidu-built" ]]; then
    run_logged "$out/build-lidu.log" bash "$lidu/build_lidu_fused_batch_get_prepare_mtp.sh"
    touch "$venv/lidu-built"
fi
vendors=("$lidu"/_custom_opp/vendors/*/op_api/lib/libcust_opapi.so)
[[ ${#vendors[@]} == 1 && -f ${vendors[0]} ]]
export ASCEND_CUSTOM_OPP_PATH="${vendors[0]%/op_api/lib/libcust_opapi.so}${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
port=$("$PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
"$kvaf/build-cpu/runtime/kvaf_cpu_service" --listen 127.0.0.1 --port "$port" --memory-page-mode thp >"$out/cpu.log" 2>&1 &
pid=$!
monitor=
cleanup() {
    if [[ -n "$monitor" ]]; then kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true; fi
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for ((i=0; i<30; i++)); do
    kill -0 "$pid" 2>/dev/null || break
    if grep -Fq "LISTEN address=127.0.0.1 port=$port" "$out/cpu.log"; then break; fi
    sleep 1
done
if ! kill -0 "$pid" 2>/dev/null || ! grep -Fq "LISTEN address=127.0.0.1 port=$port" "$out/cpu.log"; then
    tail -8 "$out/cpu.log" >&2; exit 1
fi
"$PYTHON" -B "$tools/support/peer_thp.py" --pid "$pid" --log "$out/cpu.log" --output "$out/thp.json" >"$out/thp.log" 2>&1 &
monitor=$!
cd "$kip"
if ! run_logged "$out/replay.log" "$PYTHON" -B "$tools/test_lifetch.py" diagnose --serial-only \
    --transport urma --input real --trace-root "$trace" --group 2 --round-range 1:1 \
    --device 0 --kvaf-root "$kvaf" --kvaf-server "127.0.0.1:$port" \
    --prefetch-rank-start 1024 --prefetch-tokens 1024 --intermediate-fetches 4 \
    --warmup 4 --iterations 20 --level 3 --output "$out/results" --timeout-seconds 1000; then
    find "$out/results" -name worker.log -exec tail -12 {} \; >&2
    exit 1
fi
"$PYTHON" -B "$tools/support/peer_baseline_report.py" "$out/results"
if wait "$monitor"; then cat "$out/thp.log"; else cat "$out/thp.log" >&2; exit 1; fi
monitor=
