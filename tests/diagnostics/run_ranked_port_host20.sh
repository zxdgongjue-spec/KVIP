#!/usr/bin/env bash
# Dynamic TopK only. All controls share the corrected 8 x 8 Fetch schedule.
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
ranked_sha=7e3f73b742dd778615de9725317985b6f9313ffb
checkout() {
    local branch=$1 sha=$2 destination=$3
    git -C "$kvip" fetch --quiet origin "refs/heads/$branch"
    if [[ ! -d "$destination" ]]; then
        mkdir -p "$(dirname "$destination")"
        git -C "$kvip" worktree add --quiet --detach "$destination" "$sha"
    fi
    [[ $(git -C "$destination" rev-parse HEAD) == "$sha" ]] || { echo "SOURCE_MISMATCH $destination" >&2; exit 1; }
    [[ -z $(git -C "$destination" status --porcelain --untracked-files=all -- python/kvip csrc) ]] || {
        echo "SOURCE_MODIFIED $destination" >&2; exit 1;
    }
}
ranked="$work/fused-scan-sources/native-$ranked_sha"
checkout perf/urma-ranked-only "$ranked_sha" "$ranked"
# The fix is entirely inside KVIP_RANKED_PORT. Keep the previous, unchanged
# eight-window controls and their compiled cache instead of rebuilding them.
baseline_sha=e749e42e802febe71deed1c513cc86b10c9c150b
baseline="$work/fused-scan-sources/native-$baseline_sha"
checkout perf/urma-ranked-only "$baseline_sha" "$baseline"
for group in 7 2; do
    [[ -s "$inputs/g${group}-input/manifest.json" ]] || { echo "INPUT_MISSING G$group" >&2; exit 1; }
done
[[ -x "$cpu" ]] || { echo "CPU_BINARY_MISSING $cpu" >&2; exit 1; }
for op in get put; do
    [[ -s "$build/bin/kvaf_execute_batch_${op}_kernels.o" ]] || { echo "KERNEL_MISSING $op" >&2; exit 1; }
done
set +u
source "$kvip/scripts/env.sh"
set -u
export PATH=/usr/local/python3.12.13/bin:$PATH
export ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.0
export ASCEND_HOME_PATH="$ASCEND_CANN_PACKAGE_PATH" SOC_VERSION=ascend950dt_9582
export ASCEND_RT_VISIBLE_DEVICES=4 ASCEND_LAUNCH_BLOCKING=0 VLLM_USE_V2_MODEL_RUNNER=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=2 NANOVLLM_A5_OPS_BUILD_JOBS=2
export PYTHONUNBUFFERED=1
export KVAF_BATCH_GET_KERNEL="$build/bin/kvaf_execute_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$build/bin/kvaf_execute_batch_put_kernels.o"
export PYTHONPATH="$reader/python:$build/python${PYTHONPATH:+:$PYTHONPATH}"
unset KVIP_FETCH_AB_CHECK
cd "$reader"
result=$(mktemp -d "$work/host20-ranked-port-npu4.XXXXXX")
printf 'CONFIG URMA NPU4 logical=0 BS16 Q4 656B cold_windows=8 Fetch_AIV_per_window=8 seed_is_not_window budgets=64/512\nCONTROL=%s\nRANKED=%s\nRESULT_ROOT=%s\n' \
    "$baseline_sha" "$ranked_sha" "$result"
for group in 7 2; do
    python3 -B -m kvip.diagnostics.ranked_port cpu --snapshot "$inputs/g${group}-input"
done
echo BUILD_BEGIN
if ! (
    python3 -B -m kvip.diagnostics.ranked_port build --root "$baseline" --revision "$baseline_sha" \
        --build-dir "$build" --soc Ascend950DT_9582 --baseline &&
    python3 -B -m kvip.diagnostics.ranked_port build --root "$ranked" --revision "$ranked_sha" \
        --build-dir "$build" --soc Ascend950DT_9582
) >"$result/build.log" 2>&1; then
    printf 'BUILD_FAILED LOG=%s/build.log\n' "$result" >&2
    tail -n 40 "$result/build.log" >&2
    exit 1
fi
echo BUILD_PASS
cpu_port=$(python3 -B -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
"$cpu" --listen 127.0.0.1 --port "$cpu_port" --memory-page-mode thp >"$result/.cpu.log" 2>&1 &
pid=$!
cleanup() {
    local rc=$?
    trap - EXIT
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    if ((rc)); then printf 'FAILED rc=%s RESULT_ROOT=%s\n' "$rc" "$result" >&2; fi
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
if ((!ready)); then tail -n 15 "$result/.cpu.log" >&2; exit 1; fi
echo REPLAY_CPU_READY
run_case() {
    local group=$1 name=$2 repeat=$3 source="$baseline" sha="$baseline_sha" diagnostics=0
    local extra=(--study scan-order --scan-variants "$name")
    local out="$result/g${group}-${name}-${repeat}.json" log="$result/.g${group}-${name}-${repeat}.log"
    if [[ "$name" == ranked* ]]; then
        source="$ranked"; sha="$ranked_sha"
        extra=(--study threshold --ranked-port)
        if [[ "$repeat" == 1 ]]; then diagnostics=1; fi
    fi
    if ! python3 -B -X faulthandler -m kvip.diagnostics.replay \
        --snapshot "$inputs/g${group}-input" --build-dir "$build" --transport urma --mode fused \
        --threshold-policy recent "${extra[@]}" --cold-windows 8 \
        --upstream-root "$source" --upstream-revision "$sha" \
        --kvaf-server "127.0.0.1:$cpu_port" --device 0 --soc Ascend950DT_9582 \
        --warmup 5 --iterations 10 --diagnostic-replays "$diagnostics" --quiet --output "$out" >"$log" 2>&1; then
        printf 'REPLAY_FAILED G%s variant=%s LOG=%s\n' "$group" "$name" "$log" >&2
        python3 -B -m kvip.diagnostics.console --result "$out" --log "$log" --returncode 70 >&2 || true
        tail -n 15 "$log" >&2
        return 1
    fi
}
for group in 7 2; do
    echo "BEGIN G$group graph"
    # Fail on candidate correctness before spending time on control timings.
    for name in ranked forward reverse; do run_case "$group" "$name" 1; done
    for name in reverse forward ranked; do run_case "$group" "$name" 2; done
    python3 -B -m kvip.diagnostics.ranked_port report --root "$result" --group "$group" | tee "$result/g${group}-summary.txt"
done
(cd "$result" && tar -czf results.tar.gz g*.json g*-summary.txt)
printf 'G7_G2_COMPLETE RETURN_FILE=%s/results.tar.gz\n' "$result"
