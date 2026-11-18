#!/usr/bin/env bash
# Equal-history P16 score with/without distributed radix, plus serial LI.
set -Eeuo pipefail
LI_COMPARE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
LI_COMPARE_CANN=${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}
LI_COMPARE_KVAF=$LI_COMPARE_ROOT/third_party/KVAF/access_operator/demo/aiv_batch_get/include
LI_COMPARE_SOC=Ascend950PR_9599
LI_COMPARE_DEVICE=5
LI_COMPARE_PYTHON=python3
LI_COMPARE_OUTPUT_ROOT=/home/w00568585
LI_COMPARE_PROFILE=0
LI_COMPARE_PLAN=0
LI_COMPARE_STAGE=environment
LI_COMPARE_LOG=
LI_COMPARE_OUT=
die(){ printf 'ERROR: %s\n' "$*" >&2; exit 1; }
while (($#)); do
 case "$1" in
  --profile) LI_COMPARE_PROFILE=1; shift ;;
  --plan-only) LI_COMPARE_PLAN=1; shift ;;
  --device|--output-root|--cann-root|--kvaf-include|--soc|--python)
   (($#>=2)) || die "Missing value: $1"
   case "$1" in
    --device) LI_COMPARE_DEVICE=$2 ;; --output-root) LI_COMPARE_OUTPUT_ROOT=$2 ;;
    --cann-root) LI_COMPARE_CANN=$2 ;; --kvaf-include) LI_COMPARE_KVAF=$2 ;;
    --soc) LI_COMPARE_SOC=$2 ;; --python) LI_COMPARE_PYTHON=$2 ;;
   esac
   shift 2 ;;
  -h|--help)
   cat <<'USAGE'
Usage: bash experiments/li_qk/run_li_radix_a5.sh [--profile] [options]
B1/Q4/H128K/G32/D128/K2048, identical input and full valid H.
Three cases: serial LI; equal score P16 + original TopK;
equal score P16 + distributed radix P16.
Reports score-only, separate TopK-only, and directly measured full LI.
  --profile             Separate msprof trace for each case
  --plan-only           Print the three cases, no device/build
  --device N            Default 5
  --output-root PATH    Default /home/w00568585
  --cann-root PATH      Matching CANN root
  --kvaf-include PATH   Existing compatible headers; no Fetch executes
  --soc NAME            Default Ascend950PR_9599
  --python PATH         Default python3
Warmup4/iterations20; discard r0 and compare three interleaved rounds.
USAGE
   exit 0 ;;
  *) die "Unknown argument: $1" ;;
 esac
done
cd "$LI_COMPARE_ROOT"
if ((LI_COMPARE_PLAN)); then exec "$LI_COMPARE_PYTHON" experiments/li_qk/li_radix_compare.py plan; fi
[[ ${LD_PRELOAD:-} != *camodel* && ${LD_PRELOAD:-} != *cmodel* ]] || die 'Physical A5 required'
[[ $LI_COMPARE_DEVICE =~ ^[0-9]+$ ]] || die 'Invalid device'
for LI_COMPARE_CMD in npu-smi cmake git "$LI_COMPARE_PYTHON"; do command -v "$LI_COMPARE_CMD" >/dev/null || die "Command unavailable: $LI_COMPARE_CMD"; done
[[ -f $LI_COMPARE_KVAF/kvaf_batch_get_device.h ]] || die "Missing headers: $LI_COMPARE_KVAF"
if ((LI_COMPARE_PROFILE)); then command -v msprof >/dev/null || die 'Command unavailable: msprof'; fi
mkdir -p "$LI_COMPARE_OUTPUT_ROOT"
LI_COMPARE_OUT=$(mktemp -d "$LI_COMPARE_OUTPUT_ROOT/li_radix_full_$(date +%Y%m%d_%H%M%S)_XXXXXX")
finish(){
 local li_compare_status=$?
 trap - EXIT
 printf '%s\n' "$li_compare_status" > "$LI_COMPARE_OUT/exit-status.txt"
 if ((li_compare_status)); then
  printf 'FAILED: stage=%s exit=%s\n' "$LI_COMPARE_STAGE" "$li_compare_status" >&2
  if [[ -f $LI_COMPARE_LOG ]]; then tail -n 40 "$LI_COMPARE_LOG" >&2; fi
  if [[ -f $LI_COMPARE_OUT/build/build.log ]]; then tail -n 40 "$LI_COMPARE_OUT/build/build.log" >&2; fi
 fi
 printf 'Results: %s\n' "$LI_COMPARE_OUT"
 exit "$li_compare_status"
}
trap finish EXIT
run_logged(){ LI_COMPARE_STAGE=$1; LI_COMPARE_LOG=$2; shift 2; printf '[%s] %s\n' "$(date +%H:%M:%S)" "$LI_COMPARE_STAGE"; "$@" 2>&1 | tee "$LI_COMPARE_LOG"; }
printf 'Results directory: %s\nSource checkout: %s\n' "$LI_COMPARE_OUT" "$LI_COMPARE_ROOT"
git rev-parse HEAD > "$LI_COMPARE_OUT/source-commit.txt"
git status --porcelain > "$LI_COMPARE_OUT/source-status.txt"
npu-smi info > "$LI_COMPARE_OUT/npu-smi.log"
"$LI_COMPARE_PYTHON" experiments/li_qk/li_radix_compare.py plan > "$LI_COMPARE_OUT/plan.json"
run_logged checks "$LI_COMPARE_OUT/checks.log" "$LI_COMPARE_PYTHON" experiments/li_qk/test_li_radix_compare.py
run_logged build "$LI_COMPARE_OUT/build-console.log" "$LI_COMPARE_PYTHON" experiments/li_qk/li_radix_compare.py build --run-mode npu --cann-root "$LI_COMPARE_CANN" --kvaf-include "$LI_COMPARE_KVAF" --soc "$LI_COMPARE_SOC" --output "$LI_COMPARE_OUT/build"
run_logged bench "$LI_COMPARE_OUT/bench.log" "$LI_COMPARE_PYTHON" experiments/li_qk/li_radix_compare.py bench --build "$LI_COMPARE_OUT/build" --output "$LI_COMPARE_OUT/results" --device "$LI_COMPARE_DEVICE"
if ((LI_COMPARE_PROFILE)); then
 run_logged profile "$LI_COMPARE_OUT/profile.log" "$LI_COMPARE_PYTHON" experiments/li_qk/li_radix_compare.py profile --build "$LI_COMPARE_OUT/build" --output "$LI_COMPARE_OUT/msprof" --device "$LI_COMPARE_DEVICE"
fi
cat "$LI_COMPARE_OUT/results/summary.csv"
