#!/usr/bin/env bash
# Run inside the existing A5 container and Python environment. No SSH or Docker calls.
set -Eeuo pipefail

LI_REPO=/home/w00568585/wt-mr14
LI_REF=0c1d483
LI_BRANCH=codex/split-core-unified-comparison
LI_WORK_ROOT=/home/w00568585
LI_OUTPUT_ROOT=/home/w00568585
LI_CANN=${LI_QK_CANN_ROOT:-}
LI_KVAF=${LI_QK_KVAF_INCLUDE:-}
LI_SOC=Ascend950PR_9599
LI_DEVICE=5
LI_AICS=32
LI_AIVS=64
LI_PYTHON=python3
LI_SDK=0
LI_OUT=

usage() {
    cat <<'USAGE'
Usage: bash run_a5.sh [options]
Run in the existing A5 container; fixed B1/Q4/H131072, seven orthogonal points.
  --repo PATH          Existing KVIP checkout (default /home/w00568585/wt-mr14)
  --ref COMMIT         Experiment revision on MR14 (default 0c1d483)
  --cann-root PATH     CANN root containing compiler/tikcpp; auto-detected if omitted
  --kvaf-include PATH  Directory containing kvaf_batch_get_device.h and batch_get_abi.h
  --soc NAME           Actual CANN SoC (default Ascend950PR_9599)
  --device N           Logical NPU device (default 5)
  --aics N --aivs N    Available budget, shared by all points (default 32/64)
  --python PATH        Existing Python environment executable (default python3)
  --work-root PATH     Parent of new detached worktree (default /home/w00568585)
  --output-root PATH   Parent of new results directory (default /home/w00568585)
  --sdk-reference      Also measure SDK standalone LI; requires torch_npu
  -h, --help           Show usage without fetching or running anything

Always creates new worktree/build/results directories. Does not reset the source checkout.
Warmup=4, iterations=20, measured rounds=3; r0 discarded. No Fetch/Service/Index.
USAGE
}
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --sdk-reference) LI_SDK=1; shift ;;
        --repo|--ref|--cann-root|--kvaf-include|--soc|--device|--aics|--aivs|--python|--work-root|--output-root)
            (($# >= 2)) && [[ -n $2 ]] || die "Missing value for $1"
            case "$1" in
                --repo) LI_REPO=$2 ;; --ref) LI_REF=$2 ;;
                --cann-root) LI_CANN=$2 ;; --kvaf-include) LI_KVAF=$2 ;;
                --soc) LI_SOC=$2 ;; --device) LI_DEVICE=$2 ;;
                --aics) LI_AICS=$2 ;; --aivs) LI_AIVS=$2 ;;
                --python) LI_PYTHON=$2 ;; --work-root) LI_WORK_ROOT=$2 ;;
                --output-root) LI_OUTPUT_ROOT=$2 ;;
            esac
            shift 2 ;;
        *) die "Unknown option: $1 (see --help)" ;;
    esac
done
for LI_NUM in "$LI_DEVICE" "$LI_AICS" "$LI_AIVS"; do
    [[ $LI_NUM =~ ^(0|[1-9][0-9]*)$ && ${#LI_NUM} -le 3 ]] || die 'Device/budgets must be nonnegative integers'
done
((LI_AICS >= 1 && LI_AICS <= 32 && LI_AIVS >= 2 && LI_AIVS <= 64)) || die 'Supported budget: AIC 1..32, AIV 2..64'
[[ ${LD_PRELOAD:-} != *libruntime_camodel* ]] || die 'Use the physical A5 environment, without Model preload'
for LI_CMD in git tar tee mktemp "$LI_PYTHON"; do
    command -v "$LI_CMD" >/dev/null || die "Command unavailable: $LI_CMD"
done
LI_PYTHON=$(command -v "$LI_PYTHON")
LI_REPO=$(cd "$LI_REPO" && pwd -P) || die 'Existing checkout directory unavailable'
git -C "$LI_REPO" rev-parse --git-dir >/dev/null || die 'Existing KVIP Git checkout required'

LI_CMAKE_REL=compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake
if [[ -z $LI_CANN ]]; then
    for LI_PATH in "${ASCEND_HOME_PATH:-}" "${ASCEND_CANN_PACKAGE_PATH:-}" \
        /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/cann-9.2.1; do
        if [[ -n $LI_PATH && -f $LI_PATH/$LI_CMAKE_REL ]]; then LI_CANN=$LI_PATH; break; fi
    done
fi
[[ -n $LI_CANN && -f $LI_CANN/$LI_CMAKE_REL ]] || die 'Set --cann-root to the active CANN root containing compiler/tikcpp'
LI_CANN=$(cd "$LI_CANN" && pwd -P)
# Vendor setup scripts may reference unset environment variables.
for LI_ENV in "$LI_CANN/set_env.sh" "$(dirname "$LI_CANN")/set_env.sh"; do
    if [[ -f $LI_ENV ]]; then
        set +u
        source "$LI_ENV"
        set -Eeuo pipefail
        break
    fi
done
for LI_CMD in cmake c++ npu-smi; do
    command -v "$LI_CMD" >/dev/null || die "Activate the existing CANN/build environment; command unavailable: $LI_CMD"
done
"$LI_PYTHON" -c 'import numpy' || die 'NumPy is required in the selected Python environment'
if ((LI_SDK)); then "$LI_PYTHON" -c 'import torch, torch_npu'; fi

if [[ -z $LI_KVAF ]]; then
    for LI_PATH in "$LI_REPO/third_party/KVAF" "$(dirname "$LI_REPO")/KVAF" /home/w00568585/KVAF; do
        LI_INC=$LI_PATH/access_operator/demo/aiv_batch_get/include
        if [[ -f $LI_INC/kvaf_batch_get_device.h && -f $LI_INC/batch_get_abi.h ]]; then LI_KVAF=$LI_INC; break; fi
    done
fi
[[ -n $LI_KVAF && -f $LI_KVAF/kvaf_batch_get_device.h && -f $LI_KVAF/batch_get_abi.h ]] || \
    die 'Set --kvaf-include to the existing KVAF directory containing both device headers'
LI_KVAF=$(cd "$LI_KVAF" && pwd -P)
mkdir -p "$LI_WORK_ROOT" "$LI_OUTPUT_ROOT"
LI_WORK_ROOT=$(cd "$LI_WORK_ROOT" && pwd -P)
LI_OUTPUT_ROOT=$(cd "$LI_OUTPUT_ROOT" && pwd -P)
LI_OUT=$(mktemp -d "$LI_OUTPUT_ROOT/li_qk_$(date +%Y%m%d_%H%M%S)_XXXXXX")
LI_WT=$LI_WORK_ROOT/wt-$(basename "$LI_OUT")

finish() {
    local LI_RC=$?
    trap - EXIT
    set +e
    printf '%s\n' "$LI_RC" > "$LI_OUT/exit-status.txt"
    if ((LI_RC)); then printf '\nExperiment stopped (exit %s). Logs: %s\n' "$LI_RC" "$LI_OUT" >&2; fi
    if tar -czf "$LI_OUT.tar.gz" -C "$LI_OUTPUT_ROOT" "$(basename "$LI_OUT")"; then
        printf '\nResults archive: %s.tar.gz\n' "$LI_OUT"
    else
        printf 'Archive creation failed; original results remain at %s\n' "$LI_OUT" >&2
        if ((LI_RC == 0)); then LI_RC=1; fi
    fi
    exit "$LI_RC"
}
trap finish EXIT
run_logged() { local LI_LOG=$1; shift; "$@" 2>&1 | tee "$LI_OUT/$LI_LOG"; }
printf 'repo=%s\nrequested_ref=%s\ncann=%s\nkvaf_include=%s\nsoc=%s\ndevice=%s\naics=%s\naivs=%s\npython=%s\nsdk_reference=%s\nworktree=%s\n' \
    "$LI_REPO" "$LI_REF" "$LI_CANN" "$LI_KVAF" "$LI_SOC" "$LI_DEVICE" "$LI_AICS" "$LI_AIVS" "$LI_PYTHON" "$LI_SDK" "$LI_WT" > "$LI_OUT/launcher-config.txt"
printf 'Results directory: %s\n' "$LI_OUT"
run_logged fetch.log git -C "$LI_REPO" fetch origin "$LI_BRANCH"
LI_COMMIT=$(git -C "$LI_REPO" rev-parse --verify --end-of-options "$LI_REF^{commit}")
git -C "$LI_REPO" merge-base --is-ancestor "$LI_COMMIT" FETCH_HEAD || die 'Selected revision is not on the fetched MR14 branch'
printf 'resolved_commit=%s\n' "$LI_COMMIT" >> "$LI_OUT/launcher-config.txt"
run_logged worktree.log git -C "$LI_REPO" worktree add --detach "$LI_WT" "$LI_COMMIT"
cd "$LI_WT"
[[ -f experiments/li_qk/run.py ]] || die 'Selected revision does not contain the standalone LI controls'

run_logged npu-smi.log npu-smi info
run_logged cpu-check.log "$LI_PYTHON" experiments/li_qk/test_controls.py
"$LI_PYTHON" experiments/li_qk/run.py plan --details --history 131072 \
    --aics "$LI_AICS" --aivs "$LI_AIVS" > "$LI_OUT/plan.json"
run_logged build-console.log "$LI_PYTHON" experiments/li_qk/run.py build \
    --cann-root "$LI_CANN" --soc "$LI_SOC" --kvaf-include "$LI_KVAF" --output "$LI_OUT/build"
LI_RUN_ARGS=(--build "$LI_OUT/build" --device "$LI_DEVICE" --history 131072
    --aics "$LI_AICS" --aivs "$LI_AIVS" --warmup 4 --iterations 20 --rounds 3
    --output "$LI_OUT/results")
if ((LI_SDK)); then LI_RUN_ARGS+=(--sdk-reference); fi
run_logged runner.log "$LI_PYTHON" experiments/li_qk/run.py run "${LI_RUN_ARGS[@]}"
run_logged summarize.log "$LI_PYTHON" experiments/li_qk/run.py summarize "$LI_OUT/results"
printf '\nSeven-point summary (ratios < 1 indicate acceleration):\n'
cat "$LI_OUT/results/summary.csv"
printf '\nCompleted. Summary: %s/results/summary.csv\n' "$LI_OUT"
