#!/usr/bin/env bash
# Repository entry: B1/Q4 history axis only, no network or shared package.
set -Eeuo pipefail
LI_HISTORY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
LI_HISTORY_OUTPUT_ROOT=/home/w00568585
LI_HISTORY_CANN=${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-}}
LI_HISTORY_KVAF=
LI_HISTORY_SOC=Ascend950PR_9599
LI_HISTORY_DEVICE=5
LI_HISTORY_PYTHON=python3
LI_HISTORY_PROFILE=0
LI_HISTORY_PLAN_ONLY=0
LI_HISTORY_OUT=
LI_HISTORY_STAGE=environment
LI_HISTORY_LOG=
LI_HISTORY_COMPLETED=0
LI_HISTORY_STARTED=$SECONDS
LI_HISTORY_ERROR_REPORTED=0
die(){ printf 'ERROR: %s\n' "$*" >&2; exit 1; }
report_error(){
 local li_history_status=$1 li_history_line=$2
 if ((LI_HISTORY_ERROR_REPORTED)); then return 0; fi
 LI_HISTORY_ERROR_REPORTED=1
 printf 'FAILED: stage=%s exit=%s line=%s\n' "$LI_HISTORY_STAGE" "$li_history_status" "$li_history_line" >&2
 if [[ -n $LI_HISTORY_LOG && -f $LI_HISTORY_LOG ]]; then
  printf 'Last log lines: %s\n' "$LI_HISTORY_LOG" >&2
  tail -n 40 "$LI_HISTORY_LOG" >&2 || true
 fi
}
finish(){
 local li_history_status=$?
 trap - EXIT ERR
 if ((li_history_status==0 && LI_HISTORY_COMPLETED==0)); then
  li_history_status=1
  printf 'FAILED: exited before completing the experiment, stage=%s\n' "$LI_HISTORY_STAGE" >&2
 fi
 if [[ -n $LI_HISTORY_OUT ]]; then
  printf '%s\n' "$li_history_status" > "$LI_HISTORY_OUT/exit-status.txt"
  printf 'stage=%s\nexit=%s\nelapsed_seconds=%s\n' "$LI_HISTORY_STAGE" "$li_history_status" "$((SECONDS-LI_HISTORY_STARTED))" > "$LI_HISTORY_OUT/launcher-status.txt"
  tar -czf "$LI_HISTORY_OUT.tar.gz" -C "$(dirname "$LI_HISTORY_OUT")" "$(basename "$LI_HISTORY_OUT")" || true
  printf 'Results: %s\nArchive: %s.tar.gz\n' "$LI_HISTORY_OUT" "$LI_HISTORY_OUT"
 fi
 printf 'Experiment exit=%s; elapsed=%ss\n' "$li_history_status" "$((SECONDS-LI_HISTORY_STARTED))"
 exit "$li_history_status"
}
run_logged(){
 LI_HISTORY_STAGE=$1; LI_HISTORY_LOG=$2; shift 2
 printf '[%s] %s\n' "$(date +%H:%M:%S)" "$LI_HISTORY_STAGE"
 "$@" 2>&1 | tee "$LI_HISTORY_LOG"
}
usage(){ cat <<'USAGE'
Usage: bash experiments/li_qk/run_history_a5.sh [options]
Direct repository execution on physical A5: B1/Q4/H131072/G32/D128/K2048.
History parts: 1,2,4,8,16,32,30; one score launch at each point.
T30 balances native H128 tiles (4352 or4480 keys/core); Q is never split.
  --profile             Separately collect msprof traces for all seven points
  --plan-only           Print the seven complete task plans; do not build/run
  --device N            Logical device (default 5)
  --output-root PATH    Parent of fresh build/results (default /home/w00568585)
  --cann-root PATH      Matching CANN root containing compiler/tikcpp
  --kvaf-include PATH   Compatible device headers; no Fetch is executed
  --soc NAME            Actual SoC (default Ascend950PR_9599)
  --python PATH         Python from the existing A5 environment
Warmup4/iterations20; r0 discarded; three interleaved measured rounds.
Reports score-only and score+scatter+same Final TopK, paired to same-round T1.
USAGE
}
while (($#)); do
 case "$1" in
  -h|--help) usage; exit 0 ;;
  --profile) LI_HISTORY_PROFILE=1; shift ;;
  --plan-only) LI_HISTORY_PLAN_ONLY=1; shift ;;
  --device|--output-root|--cann-root|--kvaf-include|--soc|--python)
   (($#>=2)) && [[ -n $2 ]] || die "Missing value: $1"
   case "$1" in
    --device) LI_HISTORY_DEVICE=$2 ;; --output-root) LI_HISTORY_OUTPUT_ROOT=$2 ;;
    --cann-root) LI_HISTORY_CANN=$2 ;; --kvaf-include) LI_HISTORY_KVAF=$2 ;;
    --soc) LI_HISTORY_SOC=$2 ;; --python) LI_HISTORY_PYTHON=$2 ;;
   esac
   shift 2 ;;
  *) die "Unknown option: $1" ;;
 esac
done
trap 'report_error "$?" "$LINENO"' ERR
trap finish EXIT
command -v "$LI_HISTORY_PYTHON" >/dev/null || die "Command unavailable: $LI_HISTORY_PYTHON"
cd "$LI_HISTORY_ROOT"
if ((LI_HISTORY_PLAN_ONLY)); then
 LI_HISTORY_COMPLETED=1
 exec "$LI_HISTORY_PYTHON" experiments/li_qk/run.py plan --suite history7 --history 131072 --aics 32 --aivs 64 --details
fi
printf '[%s] Checking physical A5 environment\n' "$(date +%H:%M:%S)"
[[ $LI_HISTORY_DEVICE =~ ^[0-9]+$ && ${#LI_HISTORY_DEVICE} -le 3 ]] || die 'Invalid device'
[[ ${LD_PRELOAD:-} != *camodel* && ${LD_PRELOAD:-} != *cmodel* ]] || die 'Physical A5 required; Model preload rejected'
LI_HISTORY_CMAKE_REL=compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake
if [[ -z $LI_HISTORY_CANN ]]; then
 for LI_HISTORY_PATH in /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/cann-9.2.1; do
  if [[ -f $LI_HISTORY_PATH/$LI_HISTORY_CMAKE_REL ]]; then LI_HISTORY_CANN=$LI_HISTORY_PATH; break; fi
 done
fi
[[ -f $LI_HISTORY_CANN/$LI_HISTORY_CMAKE_REL ]] || die 'Set --cann-root to the matching CANN installation'
for LI_HISTORY_ENV in "$LI_HISTORY_CANN/set_env.sh" "$(dirname "$LI_HISTORY_CANN")/set_env.sh"; do
 if [[ -f $LI_HISTORY_ENV ]]; then set +u; source "$LI_HISTORY_ENV"; set -Eeuo pipefail; break; fi
done
for LI_HISTORY_CMD in git cmake c++ npu-smi tee; do command -v "$LI_HISTORY_CMD" >/dev/null || die "Command unavailable: $LI_HISTORY_CMD"; done
if ((LI_HISTORY_PROFILE)); then command -v msprof >/dev/null || die 'msprof unavailable; activate CANN environment'; fi
if [[ -z $LI_HISTORY_KVAF ]]; then
 for LI_HISTORY_PATH in "$LI_HISTORY_ROOT/third_party/KVAF" /home/w00568585/wt-mr14/third_party/KVAF /home/w00568585/KVAF; do
  LI_HISTORY_INC=$LI_HISTORY_PATH/access_operator/demo/aiv_batch_get/include
  if [[ -f $LI_HISTORY_INC/kvaf_batch_get_device.h && -f $LI_HISTORY_INC/batch_get_abi.h ]]; then LI_HISTORY_KVAF=$LI_HISTORY_INC; break; fi
 done
fi
[[ -f $LI_HISTORY_KVAF/kvaf_batch_get_device.h && -f $LI_HISTORY_KVAF/batch_get_abi.h ]] || die 'Set --kvaf-include to compatible KVAF device headers'
mkdir -p "$LI_HISTORY_OUTPUT_ROOT"
LI_HISTORY_OUT=$(mktemp -d "$LI_HISTORY_OUTPUT_ROOT/li_history7_$(date +%Y%m%d_%H%M%S)_XXXXXX")
printf 'Results directory: %s\n' "$LI_HISTORY_OUT"
LI_HISTORY_STAGE=source_and_device
git rev-parse HEAD > "$LI_HISTORY_OUT/source-commit.txt"
git status --porcelain > "$LI_HISTORY_OUT/source-status.txt"
LI_HISTORY_LOG=$LI_HISTORY_OUT/npu-smi.log
npu-smi info > "$LI_HISTORY_OUT/npu-smi.log"
LI_HISTORY_STAGE=plan
LI_HISTORY_LOG=$LI_HISTORY_OUT/plan.json
"$LI_HISTORY_PYTHON" experiments/li_qk/run.py plan --suite history7 --details > "$LI_HISTORY_OUT/plan.json"
run_logged cpu_checks "$LI_HISTORY_OUT/checks.log" "$LI_HISTORY_PYTHON" experiments/li_qk/test_history.py
run_logged build "$LI_HISTORY_OUT/build-console.log" "$LI_HISTORY_PYTHON" experiments/li_qk/run.py build --run-mode npu --cann-root "$LI_HISTORY_CANN" --kvaf-include "$LI_HISTORY_KVAF" --soc "$LI_HISTORY_SOC" --output "$LI_HISTORY_OUT/build"
run_logged benchmark "$LI_HISTORY_OUT/bench.log" "$LI_HISTORY_PYTHON" -u experiments/li_qk/run.py run --suite history7 --history 131072 --aics 32 --aivs 64 --device "$LI_HISTORY_DEVICE" --warmup 4 --iterations 20 --rounds 3 --build "$LI_HISTORY_OUT/build" --output "$LI_HISTORY_OUT/results"
if ((LI_HISTORY_PROFILE)); then
 run_logged profile "$LI_HISTORY_OUT/profile.log" "$LI_HISTORY_PYTHON" -u experiments/li_qk/profile.py run --suite history7 --history 131072 --device "$LI_HISTORY_DEVICE" --build "$LI_HISTORY_OUT/build" --output "$LI_HISTORY_OUT/msprof"
fi
LI_HISTORY_STAGE=summary
LI_HISTORY_LOG=
cat "$LI_HISTORY_OUT/results/summary.csv"
LI_HISTORY_COMPLETED=1
LI_HISTORY_STAGE=complete
printf 'SUCCESS: seven history points completed; measured rounds=3, r0 discarded\n'
