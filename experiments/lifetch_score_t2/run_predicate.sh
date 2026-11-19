#!/usr/bin/env bash
# Check the real T2 Cold consumer; this is not a full-chain benchmark.
set -euo pipefail
LI_PROBE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LI_PROBE_CANN=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}
LI_PROBE_KVAF=
LI_PROBE_MODE=npu
LI_PROBE_SOC=Ascend950PR_9599
LI_PROBE_DEVICE=0
LI_PROBE_OUTPUT=
while (($#)); do
  case "$1" in
    --kvaf-root) LI_PROBE_KVAF=$2; shift 2 ;;
    --cann) LI_PROBE_CANN=$2; shift 2 ;;
    --mode) LI_PROBE_MODE=$2; shift 2 ;;
    --soc) LI_PROBE_SOC=$2; shift 2 ;;
    --device) LI_PROBE_DEVICE=$2; shift 2 ;;
    --output) LI_PROBE_OUTPUT=$2; shift 2 ;;
    --help) printf 'Usage: bash run_predicate.sh --kvaf-root PATH [--device 5] [--mode npu|sim] [--cann PATH] [--soc SOC] [--output PATH]\n'; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ "$LI_PROBE_MODE" == npu || "$LI_PROBE_MODE" == sim ]] || { printf 'Use --mode npu or sim.\n' >&2; exit 2; }
[[ "$LI_PROBE_DEVICE" =~ ^[0-9]+$ ]] || { printf 'Device must be nonnegative.\n' >&2; exit 2; }
[[ -f "$LI_PROBE_KVAF/access_operator/demo/aiv_batch_get/include/kvaf_batch_get_device.h" ]] || {
  printf 'Set --kvaf-root to the patched pinned KVAF checkout.\n' >&2; exit 2;
}
if [[ -z "$LI_PROBE_OUTPUT" ]]; then
  LI_PROBE_OUTPUT=$(mktemp -d "${TMPDIR:-/tmp}/lifetch_predicate.XXXXXX")
else
  mkdir -p -- "$LI_PROBE_OUTPUT"
fi
LI_PROBE_BUILD=$(mktemp -d "$LI_PROBE_OUTPUT/build.XXXXXX")
printf 'Results directory: %s\nFresh build: %s\n' "$LI_PROBE_OUTPUT" "$LI_PROBE_BUILD"
cmake -S "$LI_PROBE_DIR/model" -B "$LI_PROBE_BUILD" -DRUN_MODE="$LI_PROBE_MODE" \
  -DSOC_VERSION="$LI_PROBE_SOC" -DASCEND_CANN_PACKAGE_PATH="$LI_PROBE_CANN" \
  -DKVAF_INCLUDE="$LI_PROBE_KVAF/access_operator/demo/aiv_batch_get/include" \
  > "$LI_PROBE_OUTPUT/configure.log" 2>&1
cmake --build "$LI_PROBE_BUILD" --target predicate_smoke --parallel 2 > "$LI_PROBE_OUTPUT/build.log" 2>&1
if [[ "$LI_PROBE_MODE" == sim ]]; then
  export LD_LIBRARY_PATH="$LI_PROBE_CANN/tools/simulator/$LI_PROBE_SOC/lib:${LD_LIBRARY_PATH:-}"
  export CAMODEL_LOG_PATH="$LI_PROBE_OUTPUT/model"
  mkdir -p -- "$CAMODEL_LOG_PATH"
  LD_PRELOAD="$LI_PROBE_CANN/tools/simulator/$LI_PROBE_SOC/lib/libruntime_camodel.so${LD_PRELOAD:+:$LD_PRELOAD}" \
    "$LI_PROBE_BUILD/predicate_smoke" "$LI_PROBE_DEVICE" > "$LI_PROBE_OUTPUT/check.log" 2>&1
else
  "$LI_PROBE_BUILD/predicate_smoke" "$LI_PROBE_DEVICE" > "$LI_PROBE_OUTPUT/check.log" 2>&1
fi
tail -n 1 "$LI_PROBE_OUTPUT/check.log"
printf 'Scope: B1/Q4/H128K Cold predicate only; no Service, TopK, Fetch or performance claim. Logs: %s\n' "$LI_PROBE_OUTPUT"
