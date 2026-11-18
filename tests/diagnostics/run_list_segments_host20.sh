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
branch=perf/urma-list-segments
study=${1:-segments}
case "$study" in
    budget)
        branch=perf/urma-consumed-prefix
        native_sha=f7e092a54f7d1ef9fb0b43982a9d4ab5d938a37d
        ;;
    seed)
        branch=perf/urma-seed-mask
        native_sha=a0956cad1327112ace20f3897431aaa797534f6c
        ;;
    parallel)
        branch=perf/urma-seed-parallel
        native_sha=501ad79ea5e5578990f13fd095d2ce5bd97ef167
        ;;
    segments) ;;
    *) echo "Unknown study $study" >&2; exit 2 ;;
esac
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
    git -C "$kvip" fetch -q origin "refs/heads/$branch"
    mkdir -p "$(dirname "$native")"
    git -C "$kvip" worktree add -q --detach "$native" "$native_sha"
fi
[[ $(git -C "$native" rev-parse HEAD) == "$native_sha" ]] || { echo "SOURCE_MISMATCH expected=$native_sha path=$native" >&2; exit 1; }
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
if [[ "$study" == budget ]]; then
    python3 -B tests/diagnostics/window_budget_report.py --prior "$work/host20-segments-npu4.0R77QO"
fi
result=$(mktemp -d "$work/host20-$study-npu4.XXXXXX")
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
variants=(forward reverse reverse-inc-register reverse-inc-light-ties reverse-inc-segments reverse-end forward-end)
profile_variants=(reverse reverse-inc-register reverse-inc-light-ties reverse-inc-segments)
if [[ "$study" == budget ]]; then
    variants=(forward reverse reverse-inc-segments reverse-consumed
        reverse-cap128 reverse-inc-segments-cap128 reverse-consumed-cap128
        reverse-cap480 reverse-inc-segments-cap480 reverse-consumed-cap480 reverse-end forward-end)
    profile_variants=(reverse reverse-inc-register reverse-inc-segments reverse-consumed)
fi
run_case() {
    local group=$1 phase=$2 warm=5 iterations=10 diagnostic=0 args=() selected=("${variants[@]}")
    local base="$result/g${group}-$phase"
    if [[ "$study" == seed || "$study" == parallel ]]; then
        selected=(forward reverse reverse-consumed reverse-seed-mask reverse-seed-mask-end
            reverse-consumed-end reverse-end forward-end)
        if [[ "$group" == 4 ]]; then
            selected=(forward reverse reverse-cap128 reverse-consumed reverse-seed-mask
                reverse-seed-mask-end reverse-consumed-end reverse-cap128-end reverse-end forward-end)
        fi
        profile_variants=(reverse reverse-consumed reverse-seed-mask)
        if [[ "$study" == parallel ]]; then
            local ordered=() name
            for name in "${selected[@]}"; do
                if [[ "$name" == reverse-seed-mask ]]; then ordered+=(reverse-seed-parallel); fi
                ordered+=("$name")
                if [[ "$name" == reverse-seed-mask-end ]]; then ordered+=(reverse-seed-parallel-end); fi
            done
            selected=("${ordered[@]}")
            profile_variants+=(reverse-seed-parallel)
        fi
    fi
    if [[ "$phase" == kernels ]]; then
        selected=("${profile_variants[@]}")
        args=(--profile-dir "$result/profiles/g$group" --profile-metrics none)
        warm=2; iterations=1
    elif [[ "$phase" == markers ]]; then
        selected=(reverse-probe reverse-inc-register-probe reverse-inc-segments-probe reverse-consumed-probe)
        if [[ "$study" == seed || "$study" == parallel ]]; then
            selected=(reverse-probe reverse-consumed-probe reverse-seed-mask-probe)
            if [[ "$study" == parallel ]]; then selected+=(reverse-seed-parallel-probe); fi
        fi
        warm=1; iterations=1; diagnostic=1
    fi
    echo "BEGIN G$group $phase"
    python3 -B -X faulthandler -m kvip.diagnostics.replay \
        --snapshot "$inputs/g${group}-input" --build-dir "$build" --transport urma --mode fused \
        --threshold-policy recent --study scan-order --scan-variants "${selected[@]}" \
        --cold-windows 8 --isolate-variants --upstream-root "$native" --upstream-revision "$native_sha" \
        --kvaf-server "127.0.0.1:$cpu_port" --device 0 --soc Ascend950DT_9582 \
        --warmup "$warm" --iterations "$iterations" --diagnostic-replays "$diagnostic" "${args[@]}" --quiet \
        --output "$base.json" >"$base.log" 2>&1 || {
            local rc=$?
            echo "REPLAY_FAILED G$group LOG=$base.log" >&2
            python3 -B -m kvip.diagnostics.console --result "$base.json" --log "$base.log" --returncode "$rc" 2>&1 |
                awk '/^BUILD_DIAGNOSTIC_BEGIN/{block=1} block || /^REPLAY_FAILED|^FAILURE_JSON|^FAILURE_LOG|^RUN_LOG/{print} /^BUILD_DIAGNOSTIC_END/{block=0}' >&2
            return "$rc"
        }
    if [[ "$study" == budget || "$study" == seed || "$study" == parallel ]]; then
        python3 -B "$reader/tests/diagnostics/window_budget_report.py" "$base.json" "$phase" \
            "$native_sha" "$inputs/g${group}-input/manifest.json" | tee "$base.txt"
    elif [[ "$phase" == graph ]]; then
        python3 -B -m kvip.diagnostics.scan_report "$base.json" --expected-native "$native_sha" \
            --snapshot "$inputs/g${group}-input" > "$base.txt"
        awk '/^CONTROL |^variant |^(forward|reverse|reverse-inc-register|reverse-inc-light-ties|reverse-inc-segments) +[0-9]|^GATE reverse-inc-segments |^CHECK incremental_variants/{print}' "$base.txt"
    else
        python3 -B "$reader/tests/diagnostics/list_segment_times.py" "$base.json" | tee "$base.txt"
    fi
}
for group in 7 2 10 17 4; do run_case "$group" graph; done
for group in 7 2; do run_case "$group" kernels; done
if [[ "$study" == budget ]]; then
    python3 -B "$reader/tests/diagnostics/window_budget_report.py" --combined "$result"
    for group in 7 2; do run_case "$group" markers; done
elif [[ "$study" == seed || "$study" == parallel ]]; then
    overall=PASS
    label=SEED_GATE
    candidate=reverse-seed-mask
    if [[ "$study" == parallel ]]; then label=PARALLEL_GATE; candidate=reverse-seed-parallel; fi
    for group in 7 2 10 17 4; do
        grep -q "^$label PASS " "$result/g$group-graph.txt" || overall=KEEP_BASELINES
    done
    echo "FIVE_GROUP_GATE $candidate $overall"
    for group in 7 2; do run_case "$group" markers; done
fi
(cd "$result"; tar -czf results.tar.gz g*.json g*.txt)
echo "ALL_FIVE_PASS RETURN_FILE=$result/results.tar.gz"
