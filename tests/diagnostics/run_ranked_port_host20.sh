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
ranked_sha=09c4c349443c6bc1dccd33b3c29267861c6e5ee4
old_sha=46924fb8a71c5cafbbfa03c7de626f39464b63ff
tie_sha=a92b7c8cd9e72728817ad8455b01929fcafe924f
groups=(7 2)
history=()
audit=0
prior=0
prior_sha=aaa8d3429951464d0d62d70b10cbaaf7d3c2ed43
case "${1:-all}" in
    all) ;;
    g7) groups=(7); history=(--history) ;;
    audit) groups=(7); audit=1; ranked_sha=aaa8d3429951464d0d62d70b10cbaaf7d3c2ed43 ;;
    donor-scan) groups=(7); prior=1; ranked_sha=9952fad5ed36ce4d2885b66399f6543b8c5a8e7d ;;
    *) echo "Usage: $0 [all|g7|audit|donor-scan]" >&2; exit 2 ;;
esac
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
if ((audit)); then baseline_sha=$ranked_sha; fi
baseline="$work/fused-scan-sources/native-$baseline_sha"
checkout perf/urma-ranked-only "$baseline_sha" "$baseline"
if ((${#history[@]})); then
    for sha in "$old_sha" "$tie_sha"; do
        checkout perf/urma-ranked-only "$sha" "$work/fused-scan-sources/native-$sha"
    done
fi
if ((prior)); then
    checkout perf/urma-ranked-only "$prior_sha" "$work/fused-scan-sources/native-$prior_sha"
fi
for group in "${groups[@]}"; do
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
if ((audit)); then echo MODE=LI_phase_audit_only_not_a_performance_candidate; fi
if ((prior)); then
    printf 'PRIOR=%s\nDONOR=1d2486a76151ae37662c47aab835d06df50be866\nSCOPE=donor_scan_and_device_call_boundaries_same_rank_and_fetch\n' "$prior_sha"
fi
for group in "${groups[@]}"; do
    python3 -B -m kvip.diagnostics.ranked_port cpu --snapshot "$inputs/g${group}-input"
done
echo BUILD_BEGIN
if ! (
    python3 -B -m kvip.diagnostics.ranked_port build --root "$baseline" --revision "$baseline_sha" \
        --build-dir "$build" --soc Ascend950DT_9582 --baseline || exit $?
    if ((prior)); then
        python3 -B -m kvip.diagnostics.ranked_port build --root "$work/fused-scan-sources/native-$prior_sha" \
            --revision "$prior_sha" --build-dir "$build" --soc Ascend950DT_9582 --baseline || exit $?
    fi
    local_shas=("$ranked_sha")
    if ((prior)); then local_shas+=("$prior_sha"); fi
    if ((${#history[@]})); then local_shas+=("$old_sha" "$tie_sha"); fi
    for sha in "${local_shas[@]}"; do
        python3 -B -m kvip.diagnostics.ranked_port build --root "$work/fused-scan-sources/native-$sha" \
            --revision "$sha" --build-dir "$build" --soc Ascend950DT_9582 || exit $?
    done
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
        case "$name" in
            ranked-46924fb) sha=$old_sha ;;
            ranked-a92b7c8) sha=$tie_sha ;;
            ranked) sha=$ranked_sha ;;
            ranked-prior) sha=$prior_sha ;;
            *) return 2 ;;
        esac
        source="$work/fused-scan-sources/native-$sha"
        extra=(--study threshold --ranked-port)
        if [[ "$repeat" == 1 ]]; then diagnostics=1; fi
    fi
    if [[ "$name" == reverse-audit ]]; then
        sha=$prior_sha
        source="$work/fused-scan-sources/native-$sha"
        extra=(--study scan-order --scan-variants reverse)
    fi
    # One independent profiler replay after this process's headline samples.
    if [[ "$repeat" == 2 && "$name" != forward  ]] && { ((${#history[@]} || audit || prior)) || [[ "${RANKED_TIMELINE:-1}" == 1 ]]; }; then
        if ((${#history[@]} || audit || prior)); then
            extra+=(--final-get-detail)
            diagnostics=1
        fi
        extra+=(--profile-dir "$result/profiles/g${group}-$name" --profile-metrics none)
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
for group in "${groups[@]}"; do
    if ((audit)); then
        # Two fresh processes. Profiling and phase clocks are independent
        # replays; no diagnostic duration is treated as a headline speedup.
        for name in reverse ranked; do run_case "$group" "$name" 2; done
        python3 -B -m kvip.diagnostics.ranked_audit --root "$result" | tee "$result/g${group}-summary.txt"
        continue
    fi
    echo "BEGIN G$group graph"
    # Fail on candidate correctness before spending time on control timings.
    first_order=(ranked forward reverse)
    last_order=(reverse forward ranked)
    if ((prior)); then
        first_order=(ranked ranked-prior forward reverse)
        last_order=(reverse forward ranked-prior ranked)
    fi
    for name in "${first_order[@]}"; do run_case "$group" "$name" 1; done
    if ((${#history[@]})); then
        for name in ranked-46924fb ranked-a92b7c8; do run_case "$group" "$name" 2; done
    fi
    for name in "${last_order[@]}"; do run_case "$group" "$name" 2; done
    if ((prior)); then run_case "$group" reverse-audit 2; fi
    report_flags=("${history[@]}")
    if ((prior)); then report_flags=(--prior); fi
    python3 -B -m kvip.diagnostics.ranked_port report --root "$result" --group "$group" "${report_flags[@]}" | tee "$result/g${group}-summary.txt"
done
(cd "$result" && tar -czf results.tar.gz g*.json g*-summary.txt)
if ((audit)); then label=G7_AUDIT; elif ((prior)); then label=G7_DONOR_SCAN; elif ((${#history[@]})); then label=G7; else label=G7_G2; fi
printf '%s_COMPLETE RETURN_FILE=%s/results.tar.gz\n' "$label" "$result"
