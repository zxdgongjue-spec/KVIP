#!/usr/bin/env bash
# Direct checkout entry; no network, Docker CLI, or shared bundle needed.
set -Eeuo pipefail
LI_RADIX_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
LI_RADIX_CANN=${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}
LI_RADIX_KVAF=$LI_RADIX_ROOT/third_party/KVAF/access_operator/demo/aiv_batch_get/include
LI_RADIX_SOC=Ascend950PR_9599
LI_RADIX_DEVICE=5
LI_RADIX_PYTHON=python3
LI_RADIX_OUTPUT_ROOT=/home/w00568585
LI_RADIX_PROFILE=0
LI_RADIX_PLAN_ONLY=0
LI_RADIX_SUITE=all
LI_RADIX_DISTRIBUTION=score
LI_RADIX_OUT=
LI_RADIX_STAGE=environment
LI_RADIX_LOG=
die(){ printf 'ERROR: %s\n' "$*" >&2; exit 1; }
usage(){ cat <<'USAGE'
Usage: bash experiments/li_qk/run_radix_a5.sh [options]
Exact numeric Top2048 only, fixed Q4/H128K, immutable uint16 GM scores.
B1: original P1 vs sharded P1/P4/P16/P32.
B16: original P1 vs sharded P1/P2/P4; 64 AIV budget with per-core task loops.
  --suite all|b1|b16    Default all (nine cases)
  --distribution NAME  score (default), uniform, ties, zeros, skew
  --profile            Collect a separate msprof trace for each case
  --plan-only          Print tasks/core counts; no build or device access
  --device N           Default 5
  --output-root PATH   Default /home/w00568585
  --cann-root PATH     Matching CANN root
  --kvaf-include PATH  Existing compatible headers; no Fetch executes
  --soc NAME           Default Ascend950PR_9599
  --python PATH        Default python3
Warmup4/iterations20; discarded r0 plus three interleaved measured rounds.
USAGE
}
while (($#)); do
 case "$1" in
  -h|--help) usage; exit 0 ;;
  --profile) LI_RADIX_PROFILE=1; shift ;;
  --plan-only) LI_RADIX_PLAN_ONLY=1; shift ;;
  --suite|--distribution|--device|--output-root|--cann-root|--kvaf-include|--soc|--python)
   (($#>=2)) || die "Missing value: $1"
   case "$1" in
    --suite) LI_RADIX_SUITE=$2 ;; --distribution) LI_RADIX_DISTRIBUTION=$2 ;;
    --device) LI_RADIX_DEVICE=$2 ;; --output-root) LI_RADIX_OUTPUT_ROOT=$2 ;;
    --cann-root) LI_RADIX_CANN=$2 ;; --kvaf-include) LI_RADIX_KVAF=$2 ;;
    --soc) LI_RADIX_SOC=$2 ;; --python) LI_RADIX_PYTHON=$2 ;;
   esac
   shift 2 ;;
  *) die "Unknown option: $1" ;;
 esac
done
LI_RADIX_ARGS=(--distribution "$LI_RADIX_DISTRIBUTION")
case "$LI_RADIX_SUITE" in
 all) ;;
 b1) LI_RADIX_ARGS+=(--cases B1_original_P1,B1_sharded_P1,B1_sharded_P4,B1_sharded_P16,B1_sharded_P32) ;;
 b16) LI_RADIX_ARGS+=(--cases B16_original_P1,B16_sharded_P1,B16_sharded_P2,B16_sharded_P4) ;;
 *) die 'suite must be all, b1 or b16' ;;
esac
cd "$LI_RADIX_ROOT"
if ((LI_RADIX_PLAN_ONLY)); then exec "$LI_RADIX_PYTHON" experiments/li_qk/radix.py plan "${LI_RADIX_ARGS[@]}"; fi
[[ ${LD_PRELOAD:-} != *camodel* && ${LD_PRELOAD:-} != *cmodel* ]] || die 'Physical A5 required'
[[ $LI_RADIX_DEVICE =~ ^[0-9]+$ ]] || die 'Invalid device'
[[ -f $LI_RADIX_KVAF/kvaf_batch_get_device.h ]] || die "Missing headers: $LI_RADIX_KVAF"
for LI_RADIX_CMD in npu-smi cmake "$LI_RADIX_PYTHON"; do command -v "$LI_RADIX_CMD" >/dev/null || die "Command unavailable: $LI_RADIX_CMD"; done
if ((LI_RADIX_PROFILE)); then command -v msprof >/dev/null || die 'Command unavailable: msprof'; fi
mkdir -p "$LI_RADIX_OUTPUT_ROOT"
LI_RADIX_OUT=$(mktemp -d "$LI_RADIX_OUTPUT_ROOT/li_radix_$(date +%Y%m%d_%H%M%S)_XXXXXX")
finish(){
 local li_radix_status=$?
 trap - EXIT
 printf '%s\n' "$li_radix_status" > "$LI_RADIX_OUT/exit-status.txt"
 if ((li_radix_status)); then
  printf 'FAILED: stage=%s exit=%s\n' "$LI_RADIX_STAGE" "$li_radix_status" >&2
  if [[ -f $LI_RADIX_LOG ]]; then tail -n 40 "$LI_RADIX_LOG" >&2; fi
  if [[ -f $LI_RADIX_OUT/build/build.log ]]; then tail -n 40 "$LI_RADIX_OUT/build/build.log" >&2; fi
 fi
 printf 'Results: %s\n' "$LI_RADIX_OUT"
 exit "$li_radix_status"
}
trap finish EXIT
run_logged(){ LI_RADIX_STAGE=$1; LI_RADIX_LOG=$2; shift 2; printf '[%s] %s\n' "$(date +%H:%M:%S)" "$LI_RADIX_STAGE"; "$@" 2>&1 | tee "$LI_RADIX_LOG"; }
printf 'Results directory: %s\nSource checkout: %s\n' "$LI_RADIX_OUT" "$LI_RADIX_ROOT"
git rev-parse HEAD > "$LI_RADIX_OUT/source-commit.txt"
git status --porcelain > "$LI_RADIX_OUT/source-status.txt"
npu-smi info > "$LI_RADIX_OUT/npu-smi.log"
"$LI_RADIX_PYTHON" experiments/li_qk/radix.py plan "${LI_RADIX_ARGS[@]}" > "$LI_RADIX_OUT/plan.json"
run_logged checks "$LI_RADIX_OUT/checks.log" "$LI_RADIX_PYTHON" experiments/li_qk/test_radix.py
run_logged build "$LI_RADIX_OUT/build-console.log" "$LI_RADIX_PYTHON" experiments/li_qk/radix.py build --run-mode npu --cann-root "$LI_RADIX_CANN" --kvaf-include "$LI_RADIX_KVAF" --soc "$LI_RADIX_SOC" --output "$LI_RADIX_OUT/build"
run_logged bench "$LI_RADIX_OUT/bench.log" "$LI_RADIX_PYTHON" experiments/li_qk/radix.py bench --build "$LI_RADIX_OUT/build" --output "$LI_RADIX_OUT/results" --device "$LI_RADIX_DEVICE" "${LI_RADIX_ARGS[@]}"
if ((LI_RADIX_PROFILE)); then
 run_logged profile "$LI_RADIX_OUT/profile.log" "$LI_RADIX_PYTHON" experiments/li_qk/radix.py profile --build "$LI_RADIX_OUT/build" --output "$LI_RADIX_OUT/msprof" --device "$LI_RADIX_DEVICE" "${LI_RADIX_ARGS[@]}"
fi
cat "$LI_RADIX_OUT/results/summary.csv"
