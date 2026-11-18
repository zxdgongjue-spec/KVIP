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
compare_prior=1
if [[ "$study" == fetch && "${2:-}" == recheck ]]; then compare_prior=0; fi
groups=(2 7 10 17 4)
case "$study" in
    fetch)
        branch=perf/urma-fetch-rank-ab
        native_sha=051e5d0bb483ebff0aeb7010c60aeae0555334a5
        groups=(7 2)
        ;;
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
    trunk)
        branch=perf/urma-native-trunk
        native_sha=b127bb64b89e175b003313253245d531ff40e964
        ;;
    segments) ;;
    *) echo "Unknown study $study" >&2; exit 2 ;;
esac
native="$work/fused-scan-sources/native-$native_sha"
for group in "${groups[@]}"; do
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
if [[ "$study" == fetch ]]; then
    prior_sha=d5ad16037db5aa108ad634b0b0465ef3dd62f7bd
    prior="$work/fused-scan-sources/native-$prior_sha"
    checkouts=("$native")
    if ((compare_prior)); then
        if [[ ! -d "$prior" ]]; then
            git -C "$native" worktree add -q --detach "$prior" "$prior_sha"
        fi
        [[ $(git -C "$prior" rev-parse HEAD) == "$prior_sha" ]] || { echo "SOURCE_MISMATCH $prior" >&2; exit 1; }
        checkouts+=("$prior")
    fi
    for checkout in "${checkouts[@]}"; do
        if ! git -C "$checkout" diff --quiet HEAD -- csrc python; then
            echo "MODIFIED_NATIVE_SOURCE $checkout" >&2; exit 1
        fi
    done
fi
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
    local base="$result/g${group}-$phase" source="$native" source_sha="$native_sha"
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
    if [[ "$study" == trunk ]]; then
        local extra=() extra_end=()
        if [[ "$group" == 4 ]]; then extra=(reverse-cap128); extra_end=(reverse-cap128-end); fi
        selected=(forward reverse "${extra[@]}" reverse-seed-parallel reverse-native-trunk reverse-trunk-pruned
            reverse-seed-parallel-end reverse-trunk-pruned-end reverse-native-trunk-end "${extra_end[@]}" reverse-end forward-end)
        profile_variants=(reverse reverse-seed-parallel reverse-native-trunk reverse-trunk-pruned)
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
    if [[ "$study" == fetch ]]; then
        selected=(forward reverse reverse-fetch-bitmap reverse-fetch-dense)
        if [[ "$phase" == check ]]; then
            selected=(reverse-fetch-bitmap reverse-fetch-dense forward reverse)
            warm=1; iterations=1; diagnostic=1
        elif [[ "$phase" == dense-check ]]; then
            selected=(reverse-fetch-dense)
            warm=1; iterations=1; diagnostic=1
        elif [[ "$phase" == prior-open || "$phase" == prior-close ]]; then
            source="$prior"; source_sha="$prior_sha"
            if [[ "$phase" == prior-open ]]; then
                selected=(reverse-fetch-bitmap reverse-fetch-dense)
            else
                selected=(reverse-fetch-dense-end reverse-fetch-bitmap-end)
            fi
        else
            selected+=(reverse-fetch-dense-end reverse-fetch-bitmap-end reverse-end forward-end)
        fi
    fi
    echo "BEGIN G$group $phase"
    python3 -B -X faulthandler -m kvip.diagnostics.replay \
        --snapshot "$inputs/g${group}-input" --build-dir "$build" --transport urma --mode fused \
        --threshold-policy recent --study scan-order --scan-variants "${selected[@]}" \
        --cold-windows 8 --isolate-variants --upstream-root "$source" --upstream-revision "$source_sha" \
        --kvaf-server "127.0.0.1:$cpu_port" --device 0 --soc Ascend950DT_9582 \
        --warmup "$warm" --iterations "$iterations" --diagnostic-replays "$diagnostic" "${args[@]}" --quiet \
        --output "$base.json" >"$base.log" 2>&1 || {
            local rc=$?
            echo "REPLAY_FAILED G$group LOG=$base.log" >&2
            python3 -B -m kvip.diagnostics.console --result "$base.json" --log "$base.log" --returncode "$rc" 2>&1 |
                awk '/^BUILD_DIAGNOSTIC_BEGIN/{block=1} block || /^REPLAY_FAILED|^FAILURE_JSON|^FAILURE_LOG|^RUN_LOG/{print} /^BUILD_DIAGNOSTIC_END/{block=0}' >&2
            if [[ "$phase" == check && "$compare_prior" == 0 ]] &&
                grep -q '^REPLAY_FAILED variant=reverse-fetch-bitmap phase=diagnostic_verify ' "$base.log"; then
                # Distinguish a producer-bitmap fault from a shared rank fault on
                # the same input. Preserve the bitmap failure and skip all timing.
                run_case "$group" dense-check || true
            fi
            return "$rc"
        }
    if [[ "$study" == fetch ]]; then
        if [[ "$phase" == dense-check ]]; then echo "DENSE_CHECK G$group PASS"; return 0; fi
        if [[ "$phase" == prior-* ]]; then return 0; fi
        python3 -B "$reader/tests/diagnostics/fetch_ab_report.py" "$base.json" "$phase" \
            "$native_sha" "$inputs/g${group}-input/manifest.json" | tee "$base.txt"
    elif [[ "$study" == budget || "$study" == seed || "$study" == parallel || "$study" == trunk ]]; then
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
if [[ "$study" == fetch ]]; then
    python3 -B -m kvip.diagnostics.fetch_window_topk --inputs "$inputs" --groups 7 2
    export KVIP_FETCH_AB_CHECK=1
    for group in "${groups[@]}"; do run_case "$group" check; done
    unset KVIP_FETCH_AB_CHECK
    for group in "${groups[@]}"; do
        if ((compare_prior)); then run_case "$group" prior-open; fi
        run_case "$group" graph
        if ((!compare_prior)); then continue; fi
        run_case "$group" prior-close
        python3 -B "$reader/tests/diagnostics/fetch_ab_report.py" \
            "$result/g${group}-graph.json" revised "$native_sha" "$inputs/g${group}-input/manifest.json" \
            "$prior_sha" "$result/g${group}-prior-open.json" "$result/g${group}-prior-close.json" | tee "$result/g${group}-revised.txt"
    done
    (cd "$result" && tar -czf results.tar.gz g*.json g*.txt)
    echo "G7_G2_COMPLETE RETURN_FILE=$result/results.tar.gz"
    exit 0
fi
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
elif [[ "$study" == trunk ]]; then
    for label in DENSE_GATE PRUNED_GATE; do
        overall=PASS
        for group in 7 2 10 17 4; do
            grep -q "^$label PASS " "$result/g$group-graph.txt" || overall=KEEP_BASELINES
        done
        echo "FIVE_GROUP_GATE $label $overall"
    done
fi
(cd "$result"; tar -czf results.tar.gz g*.json g*.txt)
echo "ALL_FIVE_PASS RETURN_FILE=$result/results.tar.gz"
