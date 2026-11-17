#!/usr/bin/env bash
# Run inside the existing A5 container and Python environment. No SSH or Docker calls.
set -Eeuo pipefail

LI_REPO=/home/w00568585/wt-mr14
LI_REF=FETCH_HEAD
LI_SUITE=orthogonal8
LI_PROFILE=0
LI_WORK_GRAIN=2048
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
LI_NO_PROXY=0
LI_SKIP_FETCH=0
LI_BUNDLE=
LI_OUT=

usage() {
    cat <<'USAGE'
Usage: bash run_a5.sh [options]
Run in the existing A5 container; B1/Q4/H131072 eight points, or B16/30-core controls.
  --repo PATH          Existing KVIP checkout (default /home/w00568585/wt-mr14)
  --ref COMMIT         Experiment revision on MR14 (default fetched MR14 head)
  --cann-root PATH     CANN root containing compiler/tikcpp; auto-detected if omitted
  --kvaf-include PATH  Directory containing kvaf_batch_get_device.h and batch_get_abi.h
  --soc NAME           Actual CANN SoC (default Ascend950PR_9599)
  --device N           Logical NPU device (default 5)
  --aics N --aivs N    Available budget, shared by all points (default 32/64)
  --python PATH        Existing Python environment executable (default python3)
  --work-root PATH     Parent of new detached worktree (default /home/w00568585)
  --output-root PATH   Parent of new results directory (default /home/w00568585)
  --suite NAME         orthogonal7, orthogonal8 (default), or batch16
  --profile            Also collect separate msprof LI traces after benchmark
  --work-grain N       Balanced allocation quantum in tokens (default 2048)
  --sdk-reference      Also measure SDK standalone LI; requires torch_npu
  --no-proxy           Bypass proxy only for this Git fetch; leaves Git config unchanged
  --skip-fetch         Use the selected local commit without any network fetch
  --git-bundle PATH    Import the supplied MR14 offline bundle instead of using HTTPS
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
        --profile) LI_PROFILE=1; shift ;;
        --no-proxy) LI_NO_PROXY=1; shift ;;
        --skip-fetch) LI_SKIP_FETCH=1; shift ;;
        --work-grain|--suite|--repo|--ref|--cann-root|--kvaf-include|--soc|--device|--aics|--aivs|--python|--work-root|--output-root|--git-bundle)
            (($# >= 2)) && [[ -n $2 ]] || die "Missing value for $1"
            case "$1" in
                --work-grain) LI_WORK_GRAIN=$2 ;; --suite) LI_SUITE=$2 ;; --repo) LI_REPO=$2 ;; --ref) LI_REF=$2 ;;
                --cann-root) LI_CANN=$2 ;; --kvaf-include) LI_KVAF=$2 ;;
                --soc) LI_SOC=$2 ;; --device) LI_DEVICE=$2 ;;
                --aics) LI_AICS=$2 ;; --aivs) LI_AIVS=$2 ;;
                --python) LI_PYTHON=$2 ;; --work-root) LI_WORK_ROOT=$2 ;;
                --output-root) LI_OUTPUT_ROOT=$2 ;;
                --git-bundle) LI_BUNDLE=$2 ;;
            esac
            shift 2 ;;
        *) die "Unknown option: $1 (see --help)" ;;
    esac
done
[[ $LI_SUITE == orthogonal7 || $LI_SUITE == orthogonal8 || $LI_SUITE == batch16 ]] || die "Unknown suite: $LI_SUITE"
[[ $LI_WORK_GRAIN =~ ^[1-9][0-9]*$ && ${#LI_WORK_GRAIN} -le 5 ]] || die 'Invalid work grain'
((LI_WORK_GRAIN >= 128 && LI_WORK_GRAIN <= 65536 && (LI_WORK_GRAIN & (LI_WORK_GRAIN-1)) == 0)) || die 'Work grain must be power-of-two128..65536'
[[ $LI_SUITE == batch16 || $LI_WORK_GRAIN == 2048 ]] || die 'Work grain applies to batch16 only'
if [[ $LI_SUITE == batch16 ]]; then LI_AICS=30; LI_AIVS=60; fi
if ((LI_SKIP_FETCH)) && [[ $LI_REF == FETCH_HEAD ]]; then die "--skip-fetch requires an explicit --ref COMMIT"; fi
for LI_NUM in "$LI_DEVICE" "$LI_AICS" "$LI_AIVS"; do
    [[ $LI_NUM =~ ^(0|[1-9][0-9]*)$ && ${#LI_NUM} -le 3 ]] || die 'Device/budgets must be nonnegative integers'
done
((LI_AICS >= 1 && LI_AICS <= 32 && LI_AIVS >= 2 && LI_AIVS <= 64)) || die 'Supported budget: AIC 1..32, AIV 2..64'
if [[ $LI_SUITE == orthogonal8 ]]; then
    ((LI_AICS == 32 && LI_AIVS == 64)) || die 'orthogonal8 K32 requires 32 AIC/64 AIV'
fi
[[ ${LD_PRELOAD:-} != *libruntime_camodel* && ${LD_PRELOAD:-} != *libruntime_cmodel* ]] || die 'Use the physical A5 environment, without Model preload'
for LI_CMD in git tar tee mktemp "$LI_PYTHON"; do
    command -v "$LI_CMD" >/dev/null || die "Command unavailable: $LI_CMD"
done
LI_PYTHON=$(command -v "$LI_PYTHON")
LI_REPO=$(cd "$LI_REPO" && pwd -P) || die 'Existing checkout directory unavailable'
git -C "$LI_REPO" rev-parse --git-dir >/dev/null || die 'Existing KVIP Git checkout required'
if [[ -n $LI_BUNDLE ]]; then
    [[ -f $LI_BUNDLE ]] || die 'Offline Git bundle file unavailable'
    LI_BUNDLE=$(cd "$(dirname "$LI_BUNDLE")" && pwd -P)/$(basename "$LI_BUNDLE")
fi

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
if [[ -n $LI_BUNDLE ]]; then
    printf 'fetch_mode=offline_bundle\n' >> "$LI_OUT/launcher-config.txt"
    run_logged bundle-verify.log git -C "$LI_REPO" bundle verify "$LI_BUNDLE"
    run_logged fetch.log git -C "$LI_REPO" fetch "$LI_BUNDLE" "refs/remotes/origin/$LI_BRANCH"
elif ((LI_SKIP_FETCH)); then
    printf 'fetch_mode=local_commit\n' >> "$LI_OUT/launcher-config.txt"
    printf 'Skipped network fetch; using selected local commit %s\n' "$LI_REF" | tee "$LI_OUT/fetch.log"
elif ((LI_NO_PROXY)); then
    printf 'fetch_mode=direct_https\n' >> "$LI_OUT/launcher-config.txt"
    if run_logged fetch.log env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
        -u http_proxy -u https_proxy -u all_proxy \
        git -c http.proxy= -c remote.origin.proxy= -C "$LI_REPO" fetch origin "$LI_BRANCH"; then
        :
    else
        LI_FETCH_RC=$?
        printf 'ERROR: Direct Git fetch failed. Use --git-bundle PATH if the container cannot reach GitCode, or --skip-fetch if the selected commit exists locally\n' >&2
        exit "$LI_FETCH_RC"
    fi
else
    printf 'fetch_mode=configured_https\n' >> "$LI_OUT/launcher-config.txt"
    if run_logged fetch.log git -C "$LI_REPO" fetch origin "$LI_BRANCH"; then
        :
    else
        LI_FETCH_RC=$?
        printf 'ERROR: Git fetch failed before LI build/run. For CONNECT tunnel errors, retry with --no-proxy; --git-bundle PATH is also supported\n' >&2
        exit "$LI_FETCH_RC"
    fi
fi
LI_COMMIT=$(git -C "$LI_REPO" rev-parse --verify --end-of-options "$LI_REF^{commit}") || \
    die 'Selected experiment commit is absent locally; import the supplied bundle or fetch MR14 first'
if [[ -n $LI_BUNDLE ]] || ((LI_SKIP_FETCH == 0)); then
    git -C "$LI_REPO" merge-base --is-ancestor "$LI_COMMIT" FETCH_HEAD || die 'Selected revision is not on the fetched MR14 branch'
fi
printf 'resolved_commit=%s\n' "$LI_COMMIT" >> "$LI_OUT/launcher-config.txt"
printf 'suite=%s\nwork_grain=%s\n' "$LI_SUITE" "$LI_WORK_GRAIN" >> "$LI_OUT/launcher-config.txt"
run_logged worktree.log git -C "$LI_REPO" worktree add --detach "$LI_WT" "$LI_COMMIT"
cd "$LI_WT"
[[ -f experiments/li_qk/run.py ]] || die 'Selected revision does not contain the standalone LI controls'

run_logged npu-smi.log npu-smi info
run_logged cpu-check.log "$LI_PYTHON" experiments/li_qk/test_controls.py
"$LI_PYTHON" experiments/li_qk/run.py plan --details --history 131072 \
    --suite "$LI_SUITE" --work-grain "$LI_WORK_GRAIN" --aics "$LI_AICS" --aivs "$LI_AIVS" > "$LI_OUT/plan.json"
run_logged build-console.log "$LI_PYTHON" experiments/li_qk/run.py build \
    --cann-root "$LI_CANN" --soc "$LI_SOC" --kvaf-include "$LI_KVAF" --output "$LI_OUT/build"
LI_RUN_ARGS=(--suite "$LI_SUITE" --work-grain "$LI_WORK_GRAIN" --build "$LI_OUT/build" --device "$LI_DEVICE" --history 131072
    --aics "$LI_AICS" --aivs "$LI_AIVS" --warmup 4 --iterations 20 --rounds 3
    --output "$LI_OUT/results")
if ((LI_SDK)); then LI_RUN_ARGS+=(--sdk-reference); fi
run_logged runner.log "$LI_PYTHON" experiments/li_qk/run.py run "${LI_RUN_ARGS[@]}"
run_logged summarize.log "$LI_PYTHON" experiments/li_qk/run.py summarize "$LI_OUT/results"
if ((LI_PROFILE)); then
    run_logged profile.log "$LI_PYTHON" experiments/li_qk/profile.py run --build "$LI_OUT/build" --output "$LI_OUT/msprof" --device "$LI_DEVICE" --suite "$LI_SUITE" --work-grain "$LI_WORK_GRAIN"
fi
printf '\nSelected-suite summary (ratios < 1 indicate acceleration):\n'
cat "$LI_OUT/results/summary.csv"
printf '\nCompleted. Summary: %s/results/summary.csv\n' "$LI_OUT"
