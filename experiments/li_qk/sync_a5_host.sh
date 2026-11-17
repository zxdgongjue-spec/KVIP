#!/usr/bin/env bash
# Prepare LI sources; copying into a container is optional.
set -Eeuo pipefail
LI_CONTAINER=kvip-glm53-w00568585
LI_SYNC_DIR=/tmp/li_qk_sync
LI_BRANCH=codex/split-core-unified-comparison
LI_REMOTE_REF=refs/remotes/origin/codex/split-core-unified-comparison
LI_BUNDLE=
LI_NO_PROXY=0
LI_RUNTIME=auto
LI_NAMESPACE=
LI_SYNC_ONLY=0
LI_RUN_HERE=0
LI_EXTRA_COUNT=0
LI_EXTRA_ARGS=()
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
usage() {
    cat <<'USAGE'
Usage: bash sync_a5_host.sh [options]
Creates LI code files locally; a container CLI is optional.
  --container NAME       Target running container (default kvip-glm53-w00568585)
  --sync-dir PATH        Staging directory, created automatically (default /tmp/li_qk_sync)
  --no-proxy             Bypass proxies only for this host Git fetch
  --offline-bundle PATH  Use the downloaded MR14 bundle; no network fetch
  --sync-only            Prepare local files only; requires no container runtime
  --run-here             Prepare files and run LI in the current NPU/CANN environment
  --runtime NAME         auto, docker, podman or nerdctl (default auto)
  --namespace NAME       nerdctl/containerd namespace, if required
  -- [LI options]        Forward options such as --kvaf-include to --run-here
  -h, --help             Show usage
Without a runtime, auto mode prepares local files and prints the direct run command.
With a runtime, copies the files into the target container and verifies them.
USAGE
}
while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --no-proxy) LI_NO_PROXY=1; shift ;;
        --sync-only) LI_SYNC_ONLY=1; shift ;;
        --run-here) LI_RUN_HERE=1; LI_SYNC_ONLY=1; shift ;;
        --) shift; LI_EXTRA_COUNT=$#; if (($#)); then LI_EXTRA_ARGS=("$@"); fi; break ;;
        --container|--sync-dir|--offline-bundle|--runtime|--namespace)
            (($# >= 2)) && [[ -n $2 ]] || die "Missing value for $1"
            case "$1" in
                --container) LI_CONTAINER=$2 ;;
                --sync-dir) LI_SYNC_DIR=$2 ;;
                --offline-bundle) LI_BUNDLE=$2 ;;
                --runtime) LI_RUNTIME=$2 ;;
                --namespace) LI_NAMESPACE=$2 ;;
            esac
            shift 2 ;;
        *) die "Unknown option: $1" ;;
    esac
done
for LI_CMD in git tee; do command -v "$LI_CMD" >/dev/null || die "Command unavailable: $LI_CMD"; done
[[ $LI_RUNTIME == auto || $LI_RUNTIME == docker || $LI_RUNTIME == podman || $LI_RUNTIME == nerdctl ]] || die 'Runtime must be auto, docker, podman or nerdctl'
((LI_EXTRA_COUNT == 0 || LI_RUN_HERE == 1)) || die 'Forwarded LI options require --run-here'
if ((LI_RUN_HERE)); then
    [[ ${LD_PRELOAD:-} != *libruntime_camodel* ]] || die 'Physical A5 execution required; Model preload is not allowed'
    command -v npu-smi >/dev/null || die 'npu-smi unavailable. Run --run-here in the original A5 environment with physical NPU/driver/CANN access'
    printf 'Checking physical NPU visibility in the current A5 environment:\n'
    npu-smi info || die 'NPU driver query failed. Use the A5 environment where the previous hardware benchmark ran'
fi
if ((LI_SYNC_ONLY == 0)); then
    if [[ $LI_RUNTIME == auto ]]; then
        LI_RUNTIME=
        for LI_CMD in docker podman nerdctl; do
            if command -v "$LI_CMD" >/dev/null; then LI_RUNTIME=$LI_CMD; break; fi
        done
        if [[ -z $LI_RUNTIME ]]; then
            printf 'No container CLI available. Preparing local files only.\n'
            LI_SYNC_ONLY=1
        fi
    fi
    if ((LI_SYNC_ONLY == 0)); then
        command -v "$LI_RUNTIME" >/dev/null || die "Runtime unavailable: $LI_RUNTIME. Use --sync-only or --run-here to continue without a container CLI"
        LI_ENGINE=("$LI_RUNTIME")
        if [[ -n $LI_NAMESPACE ]]; then
            [[ $LI_RUNTIME == nerdctl ]] || die '--namespace applies to nerdctl only'
            LI_ENGINE+=(--namespace "$LI_NAMESPACE")
        fi
        "${LI_ENGINE[@]}" info >/dev/null 2>&1 || die "Cannot access $LI_RUNTIME. Use --sync-only, or select the available runtime with --runtime"
        LI_RUNNING=$("${LI_ENGINE[@]}" inspect --format '{{.State.Running}}' "$LI_CONTAINER") || die "Target container unavailable: $LI_CONTAINER (runtime $LI_RUNTIME)"
        [[ $LI_RUNNING == true ]] || die "Target container is not running: $LI_CONTAINER"
    fi
fi
if [[ -n $LI_BUNDLE ]]; then
    [[ -s $LI_BUNDLE ]] || die "Offline bundle missing or empty: $LI_BUNDLE"
    LI_BUNDLE=$(cd "$(dirname "$LI_BUNDLE")" && pwd -P)/$(basename "$LI_BUNDLE")
fi
mkdir -p "$LI_SYNC_DIR"
LI_SYNC_DIR=$(cd "$LI_SYNC_DIR" && pwd -P)
LI_GIT_DIR=$LI_SYNC_DIR/KVIP.git
if [[ ! -e $LI_GIT_DIR ]]; then
    git init --bare "$LI_GIT_DIR"
    git --git-dir="$LI_GIT_DIR" remote add origin https://gitcode.com/CacheOS/KVIP.git
else
    [[ $(git --git-dir="$LI_GIT_DIR" rev-parse --is-bare-repository) == true ]] || die "Expected a bare staging repository: $LI_GIT_DIR"
fi
printf 'Staging directory: %s\n' "$LI_SYNC_DIR"
if [[ -n $LI_BUNDLE ]]; then
    git --git-dir="$LI_GIT_DIR" bundle verify "$LI_BUNDLE"
    git --git-dir="$LI_GIT_DIR" fetch "$LI_BUNDLE" "$LI_REMOTE_REF" 2>&1 | tee "$LI_SYNC_DIR/sync.log"
else
    LI_GIT=(git)
    if ((LI_NO_PROXY)); then
        LI_GIT=(env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy
            git -c http.proxy= -c remote.origin.proxy=)
    fi
    "${LI_GIT[@]}" --git-dir="$LI_GIT_DIR" fetch origin "$LI_BRANCH" 2>&1 | tee "$LI_SYNC_DIR/sync.log"
fi
LI_HEAD=$(git --git-dir="$LI_GIT_DIR" rev-parse --verify FETCH_HEAD)
git --git-dir="$LI_GIT_DIR" cat-file -e '0c1d483^{commit}' || die 'The synced revision does not contain the requested LI experiment commit'
git --git-dir="$LI_GIT_DIR" merge-base --is-ancestor 0c1d483 "$LI_HEAD" || die 'The synced head does not include the requested LI controls'
git --git-dir="$LI_GIT_DIR" update-ref "$LI_REMOTE_REF" "$LI_HEAD"
git --git-dir="$LI_GIT_DIR" show "$LI_HEAD:experiments/li_qk/run_a5.sh" > "$LI_SYNC_DIR/run_li_qk_a5.sh.pending"
bash -n "$LI_SYNC_DIR/run_li_qk_a5.sh.pending"
git --git-dir="$LI_GIT_DIR" bundle create "$LI_SYNC_DIR/li_qk_mr14_offline.bundle.pending" "$LI_REMOTE_REF"
git --git-dir="$LI_GIT_DIR" bundle verify "$LI_SYNC_DIR/li_qk_mr14_offline.bundle.pending"
mv "$LI_SYNC_DIR/run_li_qk_a5.sh.pending" "$LI_SYNC_DIR/run_li_qk_a5.sh"
mv "$LI_SYNC_DIR/li_qk_mr14_offline.bundle.pending" "$LI_SYNC_DIR/li_qk_mr14_offline.bundle"
printf '\nCode sync completed: %s\n' "$LI_HEAD"
if ((LI_SYNC_ONLY)); then
    LI_SOURCE_REPO=/home/w00568585/wt-mr14
    if [[ ! -d $LI_SOURCE_REPO ]]; then LI_SOURCE_REPO=$LI_GIT_DIR; fi
    LI_EXEC=(bash "$LI_SYNC_DIR/run_li_qk_a5.sh" --repo "$LI_SOURCE_REPO"
        --git-bundle "$LI_SYNC_DIR/li_qk_mr14_offline.bundle")
    if ((LI_EXTRA_COUNT)); then LI_EXEC+=("${LI_EXTRA_ARGS[@]}"); fi
    if ((LI_RUN_HERE)); then
        printf 'Running physical NPU LI (RUN_MODE=npu) in the current A5 environment.\n'
        "${LI_EXEC[@]}"
    else
        printf 'Local files ready (no container copy performed):\n  %s/run_li_qk_a5.sh\n  %s/li_qk_mr14_offline.bundle\n' "$LI_SYNC_DIR" "$LI_SYNC_DIR"
        printf 'Run in an environment with NPU/CANN access:\n  '
        printf '%q ' "${LI_EXEC[@]}"; printf '\n'
    fi
else
    "${LI_ENGINE[@]}" cp "$LI_SYNC_DIR/run_li_qk_a5.sh" "$LI_CONTAINER:/tmp/run_li_qk_a5.sh"
    "${LI_ENGINE[@]}" cp "$LI_SYNC_DIR/li_qk_mr14_offline.bundle" "$LI_CONTAINER:/tmp/li_qk_mr14_offline.bundle"
    "${LI_ENGINE[@]}" exec "$LI_CONTAINER" bash -c 'test -s /tmp/run_li_qk_a5.sh && test -s /tmp/li_qk_mr14_offline.bundle && ls -lh /tmp/run_li_qk_a5.sh /tmp/li_qk_mr14_offline.bundle'
    printf 'Enter the target container:\n  '
    printf '%q ' "${LI_ENGINE[@]}" exec -it "$LI_CONTAINER" bash; printf '\n'
    printf 'Then activate the original Python/CANN environment and run:\n  bash /tmp/run_li_qk_a5.sh --repo /home/w00568585/wt-mr14 --git-bundle /tmp/li_qk_mr14_offline.bundle\n'
fi
