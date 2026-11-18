#!/usr/bin/env bash
# Replay the five frozen 656-byte inputs on host20. No model server.
set -Eeo pipefail
set +u
export PATH=/usr/local/python3.12.13/bin:$PATH
source /usr/local/Ascend/cann-9.2.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
reader=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
kvip=/home/c00982601/CacheOS/vLLM-DSA/third_party/KVIP
work=/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold
inputs="$work/skip-results.po80Nw"
build="$work/KVAF/build-npu"
cpu="$work/KVAF/build-cpu/runtime/kvaf_cpu_service"
native_sha=6de229cab2b877fea055cb3411d5161cc764ec81
native="$work/fused-scan-sources/native-$native_sha"
for group in 2 7 10 17 4; do
    [[ -s "$inputs/g${group}-input/manifest.json" ]] || { echo "INPUT_MISSING G$group $inputs" >&2; exit 1; }
done
[[ -x "$cpu" ]] || { echo "CPU_BINARY_MISSING $cpu" >&2; exit 1; }
for op in get put; do
    file="$build/bin/kvaf_execute_batch_${op}_kernels.o"
    [[ -s "$file" ]] || { echo "KERNEL_MISSING $file" >&2; exit 1; }
done
if [[ ! -d "$native" ]]; then
    git -C "$kvip" fetch -q origin refs/heads/perf/urma-list-segments
    mkdir -p "$(dirname "$native")"
    git -C "$kvip" worktree add -q --detach "$native" "$native_sha"
fi
[[ $(git -C "$native" rev-parse HEAD) == "$native_sha" ]] || { echo "6de229cab2b877fea055cb3411d5161cc764ec81_MISMATCH $native" >&2; exit 1; }
set +u
source "$kvip/scripts/env.sh"
set -u
export PATH=/usr/local/python3.12.13/bin:$PATH
export ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.0 ASCEND_RT_VISIBLE_DEVICES=4
export ASCEND_HOME_PATH="$ASCEND_CANN_PACKAGE_PATH" ASCEND_LAUNCH_BLOCKING=0 SOC_VERSION=ascend950dt_9582
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export MAX_JOBS=2 NANOVLLM_A5_OPS_BUILD_JOBS=2 VLLM_USE_V2_MODEL_RUNNER=0
export KVAF_BATCH_GET_KERNEL="$build/bin/kvaf_execute_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$build/bin/kvaf_execute_batch_put_kernels.o"
export PYTHONPATH="$reader/python:$build/python${PYTHONPATH:+:$PYTHONPATH}"
cd "$reader"
result=$(mktemp -d "$work/host20-segments-npu4.XXXXXX")
cpu_port=$(python3 -B -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
echo "CONFIG URMA NPU4 logical=0 BS16 MTP3 656B cold_windows=8 native=$native_sha"
echo "RESULT_ROOT=$result"
"$cpu" --listen 127.0.0.1 --port "$cpu_port" --memory-page-mode thp >"$result/.cpu.log" 2>&1 &
pid=$!
cleanup() {
    local rc=$?
    trap - EXIT
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    if ((rc)); then echo "FAILED rc=$rc RESULT_ROOT=$result" >&2; fi
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
ready=0
for ((i=0;i<30;i++)); do
    kill -0 "$pid" 2>/dev/null || break
    if grep -Fq "[KVAF_CPU] LISTEN address=127.0.0.1 port=$cpu_port" "$result/.cpu.log"; then ready=1; break; fi
    sleep 1
done
if ((!ready)); then tail -n 10 "$result/.cpu.log" >&2; exit 1; fi
variants=(forward reverse reverse-inc-register reverse-inc-segments reverse-end forward-end)
profile_variants=(reverse reverse-inc-register reverse-inc-segments)
run_case() {
    local group=$1 phase=$2 warm=5 iterations=10 args=() selected=("${variants[@]}")
    local base="$result/g${group}-$phase"
    if [[ "$phase" == kernels ]]; then
        selected=("${profile_variants[@]}")
        args=(--profile-dir "$result/profiles/g$group" --profile-metrics none)
        warm=2; iterations=1
    fi
    echo "BEGIN G$group $phase"
    python3 -B -X faulthandler -m kvip.diagnostics.replay \
        --snapshot "$inputs/g${group}-input" --build-dir "$build" --transport urma --mode fused \
        --threshold-policy recent --study scan-order --scan-variants "${selected[@]}" \
        --cold-windows 8 --isolate-variants --upstream-root "$native" --upstream-revision "$native_sha" \
        --kvaf-server "127.0.0.1:$cpu_port" --device 0 --soc Ascend950DT_9582 \
        --warmup "$warm" --iterations "$iterations" --diagnostic-replays 0 "${args[@]}" --quiet \
        --output "$base.json" >"$base.log" 2>&1 || {
            local rc=$?
            echo "REPLAY_FAILED G$group LOG=$base.log" >&2
            python3 -B -m kvip.diagnostics.console --result "$base.json" --log "$base.log" --returncode "$rc" 2>&1 |
                awk '/^BUILD_DIAGNOSTIC_BEGIN/{block=1} block || /^REPLAY_FAILED|^FAILURE_JSON|^FAILURE_LOG|^RUN_LOG/{print} /^BUILD_DIAGNOSTIC_END/{block=0}' >&2
            return "$rc"
        }
    if [[ "$phase" == graph ]]; then
        python3 -B -m kvip.diagnostics.scan_report "$base.json" --expected-native "$native_sha" \
            --snapshot "$inputs/g${group}-input" > "$base.txt"
        awk '/^CONTROL |^variant |^(forward|reverse|reverse-inc-register|reverse-inc-segments) +[0-9]|^GATE reverse-inc-segments |^CHECK incremental_variants/{print}' "$base.txt"
    else
        python3 -B "$reader/tests/diagnostics/list_segment_times.py" "$base.json" | tee "$base.txt"
    fi
}
for group in 7 2 10 17 4; do run_case "$group" graph; done
for group in 7 2; do run_case "$group" kernels; done
(cd "$result"; tar -czf results.tar.gz g*.json g*.txt)
echo "ALL_FIVE_PASS RETURN_FILE=$result/results.tar.gz"
