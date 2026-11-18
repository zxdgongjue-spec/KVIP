#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fixed-graph replay on host20's existing fixtures and KVAF runtime.
set -eo pipefail
die() { printf '%s\n' "$*" >&2; exit 1; }
labels=(after)
profile=0
candidate=
high_index=0
joint=0
account=0
init_only=
if [[ $# == 4 && $1 == --parity && $4 == --high-post-index ]]; then
    joint=1
    set -- "$1" "$2" "$3"
fi
if [[ ($# == 2 || $# == 3) && $1 == --parity && $2 =~ ^[0-9a-f]{40}$ ]]; then
    labels=(serial before after after-end before-end serial-end before-profile after-profile)
    candidate=$2
    [[ $# == 2 || $3 =~ ^[0-9a-f]{40}$ ]] || die "Expected a full control SHA"
elif [[ ($# == 2 || $# == 3) && $1 == --account && $2 =~ ^[0-9a-f]{40}$ ]]; then
    account=1
    labels=(serial before after after-end before-end serial-end)
    candidate=$2
    if [[ $# == 3 ]]; then
        [[ $3 =~ ^[0-9a-f]{40}$ ]] || die "Expected a full init-only SHA"
        init_only=$3
        labels=(serial before init-only after after-end init-only-end before-end serial-end)
    fi
elif [[ $# == 1 && $1 == --parity ]]; then
    labels=(serial before after after-end before-end serial-end)
elif [[ $# == 1 && $1 == --kernels ]]; then
    labels=(serial after)
    profile=1
elif [[ $# == 2 && $1 == --high-index && $2 =~ ^[0-9a-f]{40}$ ]]; then
    high_index=1
    candidate=$2
elif [[ $# != 0 ]]; then
    die "usage: bash host20_g2.sh [--parity [CANDIDATE_SHA [CONTROL_SHA [--high-post-index]]]|--account CANDIDATE_SHA [INIT_ONLY_SHA]|--kernels|--high-index CANDIDATE_SHA]"
fi

diag=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
cache=/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold
repo=/home/c00982601/CacheOS/vLLM-DSA/third_party/KVIP
src="$cache/adaptive-sources"
build="$cache/KVAF/build-npu"
cpu="$cache/KVAF/build-cpu/runtime/kvaf_cpu_service"
snapshot="$cache/skip-results.po80Nw/g2-input"
main=532df2c8dfcb27cf8ab203f70d5bdf3510d77b6c
before=96198a2bfc9ba2e983c090225b9acdfdf40c6099
after=0f103cc1b6db1beacfb90023d02befc5eff8ae9e
if [[ -n $candidate ]]; then
    before=${3:-$after}
    if ((account)); then before=66a2f469ebaec3b35fff789a75177ee19b00213e; fi
    after=$candidate
    [[ $after != "$before" ]] && git -C "$repo" merge-base --is-ancestor "$before" "$after" ||
        die "Candidate must be a distinct descendant of $before"
    if [[ -n $init_only ]]; then
        [[ $(git -C "$repo" rev-parse "$after^") == "$init_only" && $(git -C "$repo" rev-parse "$init_only^") == "$before" ]] ||
            die "Expected baseline -> init-only -> init+LI revisions"
    fi
fi
[[ -s "$build/CMakeCache.txt" && -x "$cpu" ]] || die "Missing existing KVAF build or CPU helper: $cache"
if ((high_index)); then
    for group in 7 10 17 4; do
        [[ -s "$cache/skip-results.po80Nw/g$group-input/manifest.json" ]] || die "Missing snapshot for G$group"
    done
else
    [[ -s "$snapshot/manifest.json" ]] || die "Missing G2 snapshot: $snapshot"
fi
clean() { git -C "$1" diff --quiet && git -C "$1" diff --cached --quiet; }
clean "$diag" || die "Modified diagnostics checkout: $diag"
for label in "${labels[@]}"; do
    case "$label" in
        serial*) rev=$main; root="$src/main-${rev:0:8}" ;;
        before*) rev=$before; root="$src/auto-${rev:0:8}" ;;
        init-only*) rev=$init_only; root="$src/auto-${rev:0:8}" ;;
        after*) rev=$after; root="$src/auto-${rev:0:8}" ;;
    esac
    [[ -d "$root" && $(git -C "$root" rev-parse HEAD) == "$rev" ]] && clean "$root" ||
        die "Missing, modified or wrong production checkout: $root (want $rev)"
done

export PATH=/usr/local/python3.12.13/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
envroot="$src/candidate-c72c82cf"
has_lidu() { compgen -G "$1/csrc/operators/lidu/_custom_opp/vendors/*/op_api/lib/libcust_opapi.so" >/dev/null; }
has_lidu "$envroot" || envroot=$repo
has_lidu "$envroot" || die "Missing previously installed LIDU in $envroot"
source "$envroot/scripts/env.sh"
set -euo pipefail
cann=$(sed -n 's/^ASCEND_CANN_PACKAGE_PATH:[^=]*=//p' "$build/CMakeCache.txt")
export ASCEND_CANN_PACKAGE_PATH=${cann:-/usr/local/Ascend/ascend-toolkit/latest}
export ASCEND_HOME_PATH="$ASCEND_CANN_PACKAGE_PATH"
export ASCEND_RT_VISIBLE_DEVICES=4 SOC_VERSION=ascend950dt_9582 ASCEND_LAUNCH_BLOCKING=0
export MAX_JOBS=2 NANOVLLM_A5_OPS_BUILD_JOBS=2 OMP_NUM_THREADS=1
export PYTHONPATH="$diag/python:$build/python${PYTHONPATH:+:$PYTHONPATH}"
export KVAF_BATCH_GET_KERNEL="$build/bin/kvaf_execute_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$build/bin/kvaf_execute_batch_put_kernels.o"
[[ -s "$KVAF_BATCH_GET_KERNEL" && -s "$KVAF_BATCH_PUT_KERNEL" ]] || die "Missing Get/Put objects in $build/bin"

if ((high_index)); then
    out=$(mktemp -d "$cache/host20-high-index.XXXXXX")
    printf 'HIGH_INDEX NPU4 logical=0 candidate=%.8s\n' "$after"
else
    out=$(mktemp -d "$cache/host20-g2-map-check.XXXXXX")
    printf 'G2 NPU4 logical=0 control=%.8s candidate=%.8s harness=%.8s\n' "$before" "$after" "$(git -C "$diag" rev-parse HEAD)"
fi
printf 'RESULT_ROOT=%s\n' "$out"
port=29084
"$cpu" --listen 127.0.0.1 --port "$port" --memory-page-mode thp >"$out/.cpu.log" 2>&1 &
pid=$!
cleanup() {
    local rc=$?
    trap - EXIT
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    if ((rc == 0)); then rm -f "$out/.cpu.log"; fi
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
ready=0
for ((i=0; i<30; i++)); do
    kill -0 "$pid" 2>/dev/null || break
    if grep -Fq "[KVAF_CPU] LISTEN address=127.0.0.1 port=$port" "$out/.cpu.log"; then ready=1; break; fi
    sleep 1
done
if [[ $ready != 1 ]] || ! kill -0 "$pid" 2>/dev/null; then tail -n 8 "$out/.cpu.log" >&2; exit 1; fi

cd "$diag"
if ((high_index)); then
    python3 -B -m kvip.diagnostics.high_index_study --cache "$cache" \
        --source "$src/auto-${after:0:8}" --revision "$after" --server "127.0.0.1:$port" --result-dir "$out"
    exit
fi
common=(--snapshot "$snapshot" --build-dir "$build" --transport urma --kvaf-server "127.0.0.1:$port"
        --device 0 --soc Ascend950DT_9582 --warmup 2 --quiet)
for label in "${labels[@]}"; do
    extra=(--diagnostic-replays 0)
    capture=$profile
    iterations=10
    if [[ $label == *-profile ]]; then
        capture=1
        iterations=2  # First timed replay learns, second confirms the LOW route.
    fi
    if [[ $label == serial* ]]; then
        rev=$main; root="$src/main-${rev:0:8}"; args=(--mode serial --serial-workers 32)
    else
        rev=$after
        if [[ $label == before* ]]; then rev=$before; fi
        if [[ $label == init-only* ]]; then rev=$init_only; fi
        root="$src/auto-${rev:0:8}"
        args=(--mode fused --study post-threshold --threshold-policy recent --pipeline-variants auto-post-index --check-auto-route)
        if ((joint)); then args+=(--high-post-index); fi
        if ((!capture)) && [[ $label == before || $label == after ]]; then
            extra=(--diagnostic-replays 2 --calibrate-counter)
        fi
    fi
    role=${label%-profile}
    output="$out/g2-$label.json"
    if ((account)); then
        round=1
        if [[ $label == *-end ]]; then round=2; fi
        role=${label%-end}
        stem="r$round-$role"
        output="$out/$stem.json"
        extra=(--diagnostic-replays 0 --profile-graph-event --profile-dir "$out/profile-$stem" --profile-metrics none)
    elif ((capture)); then
        extra+=(--profile-dir "$out/$role-profile" --profile-metrics none)
    fi
    log="$out/.$label.log"
    if python3 -B -m kvip.diagnostics.replay "${common[@]}" --iterations "$iterations" "${args[@]}" "${extra[@]}" \
        --upstream-root "$root" --upstream-revision "$rev" --output "$output" >"$log" 2>&1; then
        if [[ $label == before || $label == after || $label == init-only ]]; then
            python3 -B - "$output" "$label" "$rev" "$joint" <<'PY'
import json
import sys
from pathlib import Path

path, label, revision, joint = sys.argv[1:]
case = json.loads(Path(path).read_text())["cases"]["auto-post-index"]
lifecycle = case["auto_route"]["lifecycle"]
policy = lifecycle["low_history_map_contract"]
want = "write_abc" if revision == "96198a2bfc9ba2e983c090225b9acdfdf40c6099" else "preserve_all"
rows = sum(row["mode"] == "low" and row["poisoned_history_map"] for row in lifecycle["sequence"])
if case["kvip_commit"] != revision or lifecycle["status"] != "pass" or policy != want or not rows:
    raise SystemExit(f"MAP_CHECK_FAILED label={label} policy={policy} tested_low_steps={rows} JSON={path}")
print(f"MAP_CHECK_PASS label={label} policy={policy} tested_low_steps={rows}")
if joint == "1":
    observed = [step["mode"] for step in case["auto_route"]["samples"]]
    reads = case["reads_samples"]
    if (not case.get("high_post_index") or set(observed) != {"low", "high"}
            or len(reads) != len(observed)
            or any(bool(row.get("high_post_index")) != (mode == "high")
                   or row["post_index"]["active"] != (mode == "low")
                   for mode, row in zip(observed, reads))):
        raise SystemExit(f"JOINT_ROUTE_FAILED label={label} JSON={path}")
    print(f"JOINT_ROUTE_PASS label={label} low={observed.count('low')} high={observed.count('high')}")
PY
        fi
        if ((${#labels[@]} == 1)); then
            grep -E '^REPLAY_PASS|^AUTO_ROUTE_PASS|^AUTO_LIFECYCLE_PASS' "$log"
        fi
        if ((capture)); then
            python3 -B "$diag/tests/diagnostics/tools/g2_kernel_span.py" "$out/$role-profile" "$role"
        fi
        rm -f "$log"
    else
        rc=$?
        printf 'FAILED=%s rc=%d LOG=%s\n' "$label" "$rc" "$log" >&2
        tail -n 12 "$log" >&2
        exit "$rc"
    fi
    if [[ $label == serial-end ]] && (( !account )); then
        python3 -B -m kvip.diagnostics.post_index_plan_report --directory "$out" --group 2 |
            awk '/^RESULT_BEGIN|ALL_mean\/p50=|^GAIN |^DRIFT |^CLOCK |^CHECK |^OBSERVED |^RESULT_END/ ||
                 (/^(INIT_LI|PHASE_END|INDEX_CQ|FINALIZE) / && /sample=1 /)'
    fi
done

if ((account)); then
    python3 -B -m kvip.diagnostics.graph_accounting --directory "$out"
fi
