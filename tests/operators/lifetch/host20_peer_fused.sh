#!/usr/bin/env bash
# G2 replay with the peer timing protocol and one transport shared by every arm.
set -eo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
tools="$root/tests/operators/lifetch"
work=/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold
area="$work/peer-baseline"
src="$work/adaptive-sources"
old=cdfc4da4b53b80255740183c7a0a8db8bf350dcb
new=4262f8e6be86c7da65cc3771b97d30467bb14e4e
serial_revision=da318bdb81344672b5d9013e04d517669d3eb142
baseline="$src/peer-serial-${serial_revision:0:8}"
clean="$area/KVAF-de4fd37e"
transport=a2857b460a74704114fd30a849ae38666cc2c372
kvaf="$area/KVAF-${transport:0:8}"
snapshot="$work/skip-results.po80Nw/g2-input"
mkdir -p "$area"
out=$(mktemp -d "$area/fused.XXXXXX")
printf 'RESULT_ROOT=%s\n' "$out"
export PATH=/usr/local/python3.12.13/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -euo pipefail
pin() {
    local repo=$1 checkout=$2 sha=$3
    if [[ ! -e "$checkout" ]]; then git -C "$repo" worktree add -q --detach "$checkout" "$sha"; fi
    [[ $(git -C "$checkout" rev-parse HEAD) == "$sha" ]] || { echo "Unexpected revision in $checkout" >&2; exit 1; }
    git -C "$checkout" diff --quiet && git -C "$checkout" diff --cached --quiet || { echo "Modified source: $checkout" >&2; exit 1; }
}
pin "$root" "$baseline" "$serial_revision"
git -C "$root" diff --exit-code --quiet "$serial_revision" HEAD -- csrc/operators/lidu
git -C "$root" fetch -q origin test/peer-g2-low-cq
pin "$root" "$src/auto-${old:0:8}" "$old"
pin "$root" "$src/auto-${new:0:8}" "$new"
git -C "$clean" fetch -q origin test/peer-main-urma-fused-context
pin "$clean" "$kvaf" "$transport"
pin "$clean" "$clean" de4fd37e02fa86357f71fe4fcbbbecc3e75d1d58
test -s "$snapshot/manifest.json" || { echo "Missing G2 snapshot $snapshot" >&2; exit 1; }
trace=$(bash "$tools/peer_g2_input.sh" "$work")
cann=$(sed -n 's/^ASCEND_CANN_PACKAGE_PATH:[^=]*=//p' "$clean/build-npu/CMakeCache.txt")
test -d "$cann/include/acl"
export ASCEND_HOME_PATH="$cann" ASCEND_CANN_PACKAGE_PATH="$cann"
export ASCEND_RT_VISIBLE_DEVICES=4 SOC_VERSION=Ascend950DT_9582 ASCEND_LAUNCH_BLOCKING=0
export MAX_JOBS=2 OMP_NUM_THREADS=1
venv="$area/python-$(git -C "$baseline" rev-parse --short=8 HEAD)"
export PATH="$venv/bin:$PATH" PYTHON="$venv/bin/python"
test -x "$PYTHON"
lidu="$baseline/csrc/operators/lidu"
vendors=("$lidu"/_custom_opp/vendors/*/op_api/lib/libcust_opapi.so)
[[ ${#vendors[@]} == 1 && -f ${vendors[0]} ]]
export ASCEND_CUSTOM_OPP_PATH="${vendors[0]%/op_api/lib/libcust_opapi.so}${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export PYTHONPATH="$lidu/torch_extension:$root/python:$kvaf/build-npu/python${PYTHONPATH:+:$PYTHONPATH}"
export KVAF_BATCH_GET_KERNEL="$kvaf/build-npu/bin/kvaf_execute_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$kvaf/build-npu/bin/kvaf_execute_batch_put_kernels.o"
if ! bash "$kvaf/build.sh" npu --without-aicpu --jobs 2 >"$out/build.log" 2>&1; then
    tail -15 "$out/build.log" >&2; exit 1
fi
cpu="$clean/build-cpu/runtime/kvaf_cpu_service"
test -x "$cpu"
printf 'CONFIG G2 BS16 Q4 656B NPU4 after=%s before=%s main=a56154c3 KVAF=%s\n' "${new:0:8}" "${old:0:8}" "${transport:0:8}"
run_arm() (
    set -euo pipefail
    label=$1
    arm="$out/$label"; mkdir "$arm"
    port=$("$PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
    "$cpu" --listen 127.0.0.1 --port "$port" --memory-page-mode thp >"$arm/cpu.log" 2>&1 &
    pid=$!
    monitor=
    cleanup() {
        if [[ -n "$monitor" ]]; then kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true; fi
        kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
    }
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    for ((i=0; i<30; i++)); do
        kill -0 "$pid" 2>/dev/null || break
        if grep -Fq "LISTEN address=127.0.0.1 port=$port" "$arm/cpu.log"; then break; fi
        sleep 1
    done
    if ! kill -0 "$pid" 2>/dev/null || ! grep -Fq "LISTEN address=127.0.0.1 port=$port" "$arm/cpu.log"; then
        tail -6 "$arm/cpu.log" >&2; exit 1
    fi
    "$PYTHON" -B "$tools/support/peer_thp.py" --timeout 1800 --pid "$pid" --log "$arm/cpu.log" --output "$arm/thp.json" >"$arm/thp.log" 2>&1 &
    monitor=$!
    case "$label" in
        serial*) mode=(--serial-only) ;;
        main) mode=(--fused-only) ;;
        before*) candidate=$old ;;
        after) candidate=$new ;;
    esac
    limits=(--prefetch-tokens 1024 --intermediate-fetches 4)
    if [[ "$label" == before* || "$label" == after ]]; then
        limits=(--prefetch-tokens 512 --intermediate-fetches 8)
        mode=(--fused-only --candidate-root "$src/auto-${candidate:0:8}" --candidate-snapshot "$snapshot")
    fi
    printf 'CASE_BEGIN %s\n' "$label"
    if ! "$PYTHON" -B "$tools/test_lifetch.py" diagnose "${mode[@]}" "${limits[@]}" \
        --transport urma --input real --trace-root "$trace" --group 2 --round-range 1:1 \
        --device 0 --kvaf-root "$kvaf" --kvaf-server "127.0.0.1:$port" \
        --prefetch-rank-start 1024 --warmup 4 --iterations 20 --level 1 \
        --output "$arm/results" --timeout-seconds 1800 >"$arm/replay.log" 2>&1; then
        tail -5 "$arm/replay.log" >&2
        find "$arm/results" -name worker.log -exec tail -10 {} \; >&2
        exit 1
    fi
    if ! wait "$monitor"; then cat "$arm/thp.log" >&2; exit 1; fi
    monitor=
    "$PYTHON" -B "$tools/support/peer_delta.py" "$out" "$label"
)
cd "$root"
for label in serial main before after before-end serial-end; do run_arm "$label"; done
"$PYTHON" -B "$tools/support/peer_delta.py" "$out"
