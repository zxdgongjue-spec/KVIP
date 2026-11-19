#!/usr/bin/env bash
# Compile each invocation into a fresh directory; no Git or Docker required.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CANN_ROOT=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}
KVAF_ROOT=
DEVICE=0
BATCHES=1,16
MODE=npu
SOC=Ascend950PR_9599
OUTPUT=
while (($#)); do
  case "$1" in
    --kvaf-root) KVAF_ROOT=$2; shift 2 ;;
    --cann) CANN_ROOT=$2; shift 2 ;;
    --device) DEVICE=$2; shift 2 ;;
    --batches) BATCHES=$2; shift 2 ;;
    --mode) MODE=$2; shift 2 ;;
    --soc) SOC=$2; shift 2 ;;
    --output) OUTPUT=$2; shift 2 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "$KVAF_ROOT" && -f "$KVAF_ROOT/access_operator/demo/aiv_batch_get/include/kvaf_batch_get_device.h" ]] || {
  printf 'Set --kvaf-root to the patched pinned KVAF checkout.\n' >&2; exit 2;
}
[[ "$MODE" == npu || "$MODE" == sim ]] || { printf 'Use --mode npu or sim.\n' >&2; exit 2; }
[[ "$BATCHES" =~ ^(1|16)(,(1|16))*$ && "$DEVICE" =~ ^[0-9]+$ ]] || { printf 'Use B1/B16 and a nonnegative device.\n' >&2; exit 2; }
if [[ -z "$OUTPUT" ]]; then OUTPUT=$(mktemp -d "${TMPDIR:-/tmp}/lifetch_cold_contract.XXXXXX");
else mkdir -p -- "$OUTPUT"; fi
BUILD=$(mktemp -d "$OUTPUT/build.XXXXXX")
printf 'Results directory: %s\nBuild directory: %s\n' "$OUTPUT" "$BUILD"
trap 'status=$?; if ((status)); then printf "Cold contract failed (exit %s); logs: %s\n" "$status" "$OUTPUT" >&2; tail -n 35 "$OUTPUT/build.log" 2>/dev/null || true; fi' EXIT
cmake -S "$SCRIPT_DIR" -B "$BUILD" -DRUN_MODE="$MODE" -DSOC_VERSION="$SOC" \
  -DASCEND_CANN_PACKAGE_PATH="$CANN_ROOT" \
  -DKVAF_INCLUDE="$KVAF_ROOT/access_operator/demo/aiv_batch_get/include" > "$OUTPUT/configure.log" 2>&1
cmake --build "$BUILD" --target cold_contract_host --parallel 2 > "$OUTPUT/build.log" 2>&1
if [[ "$MODE" == sim ]]; then
  export LD_LIBRARY_PATH="$CANN_ROOT/tools/simulator/$SOC/lib:${LD_LIBRARY_PATH:-}"
  export CAMODEL_LOG_PATH="$OUTPUT/model"
  mkdir -p -- "$CAMODEL_LOG_PATH"
fi
IFS=',' read -ra VALUES <<< "$BATCHES"
for batch in "${VALUES[@]}"; do
  printf 'Checking Cold predicate: B%s, Q4, H32768, device %s\n' "$batch" "$DEVICE"
  if ! "$BUILD/cold_contract_host" "$batch" "$DEVICE" > "$OUTPUT/b${batch}.log" 2>&1; then
    tail -n 40 "$OUTPUT/b${batch}.log" >&2
    exit 1
  fi
  printf 'B%s: PASS\n' "$batch"
done
printf 'PASS: bitmap/threshold/input/guard checks; scope=Cold consumer only. Logs: %s\n' "$OUTPUT"
