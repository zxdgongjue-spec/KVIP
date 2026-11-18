#!/usr/bin/env bash
# Fixed full LI, baseline plus three producer-fusion variants. No Docker/network.
set -Eeuo pipefail
LI_PRODUCER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
LI_PRODUCER_CANN=${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}
LI_PRODUCER_INC=$LI_PRODUCER_ROOT/third_party/KVAF/access_operator/demo/aiv_batch_get/include
LI_PRODUCER_SOC=Ascend950PR_9599
LI_PRODUCER_DEVICE=5
LI_PRODUCER_PYTHON=python3
LI_PRODUCER_OUTPUT_ROOT=/home/w00568585
LI_PRODUCER_PROFILE=0
while (($#)); do
 case "$1" in
  --profile) LI_PRODUCER_PROFILE=1; shift ;;
  --device|--cann-root|--kvaf-include|--soc|--python|--output-root)
   (($#>=2)) || { echo "Missing value: $1" >&2; exit 2; }
   case "$1" in
    --device) LI_PRODUCER_DEVICE=$2;; --cann-root) LI_PRODUCER_CANN=$2;; --kvaf-include) LI_PRODUCER_INC=$2;;
    --soc) LI_PRODUCER_SOC=$2;; --python) LI_PRODUCER_PYTHON=$2;; --output-root) LI_PRODUCER_OUTPUT_ROOT=$2;;
   esac
   shift 2 ;;
  -h|--help) cat <<'USAGE'
Usage: bash experiments/li_qk/run_producer_a5.sh [--device 5] [--profile]
B1/Q4/H128K/G32/D128/K2048/P16, score16AIC/32AIV, tail4AIV.
split_scatter: existing split score + scatter + original TopK
 direct: direct canonical score + original TopK
 high_fused: score + partial high histogram; one exact per-Q radix tail
 local_topk: score + local Top2048; one global candidate Top2048 tail
Warmup4/iterations20; discard r0, three interleaved rounds.
Other options: --cann-root --kvaf-include --soc --python --output-root
USAGE
   exit 0 ;;
  *) echo "Unknown argument: $1" >&2; exit 2;;
 esac
done
[[ ${LD_PRELOAD:-} != *camodel* && ${LD_PRELOAD:-} != *cmodel* ]] || { echo 'Physical A5 required' >&2; exit 1; }
for LI_PRODUCER_CMD in npu-smi cmake git "$LI_PRODUCER_PYTHON"; do command -v "$LI_PRODUCER_CMD" >/dev/null || { echo "Command unavailable: $LI_PRODUCER_CMD" >&2; exit 1; }; done
if ((LI_PRODUCER_PROFILE)); then command -v msprof >/dev/null || { echo 'Command unavailable: msprof' >&2; exit 1; }; fi
[[ -f $LI_PRODUCER_INC/kvaf_batch_get_device.h ]] || { echo "Missing headers: $LI_PRODUCER_INC" >&2; exit 1; }
mkdir -p "$LI_PRODUCER_OUTPUT_ROOT"
LI_PRODUCER_OUT=$(mktemp -d "$LI_PRODUCER_OUTPUT_ROOT/li_producer_$(date +%Y%m%d_%H%M%S)_XXXXXX")
LI_PRODUCER_STAGE=environment
LI_PRODUCER_LOG=
finish(){
 local li_producer_status=$?
 trap - EXIT
 printf '%s\n' "$li_producer_status" > "$LI_PRODUCER_OUT/exit-status.txt"
 if ((li_producer_status)); then
  printf 'FAILED: stage=%s exit=%s\n' "$LI_PRODUCER_STAGE" "$li_producer_status" >&2
  if [[ -f $LI_PRODUCER_LOG ]]; then tail -n 40 "$LI_PRODUCER_LOG" >&2; fi
  if [[ -f $LI_PRODUCER_OUT/build/build.log ]]; then tail -n 40 "$LI_PRODUCER_OUT/build/build.log" >&2; fi
 fi
 printf 'Results: %s\n' "$LI_PRODUCER_OUT"
 exit "$li_producer_status"
}
trap finish EXIT
run_logged(){ LI_PRODUCER_STAGE=$1; LI_PRODUCER_LOG=$2; shift 2; printf '[%s] %s\n' "$(date +%H:%M:%S)" "$LI_PRODUCER_STAGE"; "$@" 2>&1 | tee "$LI_PRODUCER_LOG"; }
cd "$LI_PRODUCER_ROOT"
printf 'Results directory: %s\nSource checkout: %s\n' "$LI_PRODUCER_OUT" "$LI_PRODUCER_ROOT"
git rev-parse HEAD > "$LI_PRODUCER_OUT/source-commit.txt"
git status --porcelain > "$LI_PRODUCER_OUT/source-status.txt"
npu-smi info > "$LI_PRODUCER_OUT/npu-smi.log"
run_logged checks "$LI_PRODUCER_OUT/checks.log" "$LI_PRODUCER_PYTHON" experiments/li_qk/test_producer.py
run_logged build "$LI_PRODUCER_OUT/build-console.log" "$LI_PRODUCER_PYTHON" experiments/li_qk/producer_compare.py build --run-mode npu --cann-root "$LI_PRODUCER_CANN" --kvaf-include "$LI_PRODUCER_INC" --soc "$LI_PRODUCER_SOC" --output "$LI_PRODUCER_OUT/build"
run_logged bench "$LI_PRODUCER_OUT/bench.log" "$LI_PRODUCER_PYTHON" experiments/li_qk/producer_compare.py bench --build "$LI_PRODUCER_OUT/build" --output "$LI_PRODUCER_OUT/results" --device "$LI_PRODUCER_DEVICE"
if ((LI_PRODUCER_PROFILE)); then run_logged profile "$LI_PRODUCER_OUT/profile.log" "$LI_PRODUCER_PYTHON" experiments/li_qk/producer_compare.py profile --build "$LI_PRODUCER_OUT/build" --output "$LI_PRODUCER_OUT/msprof" --device "$LI_PRODUCER_DEVICE"; fi
cat "$LI_PRODUCER_OUT/results/summary.csv"
