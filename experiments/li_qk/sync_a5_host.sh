#!/usr/bin/env bash
# Run on the A5 Docker host. Creates the staging directory and copies verified files.
set -Eeuo pipefail
LI_CONTAINER=kvip-glm53-w00568585
LI_SYNC_DIR=/tmp/li_qk_sync
LI_BRANCH=codex/split-core-unified-comparison
LI_REMOTE_REF=refs/remotes/origin/codex/split-core-unified-comparison
LI_BUNDLE=
LI_NO_PROXY=0
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
usage() {
    cat <<'USAGE'
Usage: bash sync_a5_host.sh [options]
Run on the A5 Docker host, outside the container.
  --container NAME       Target running container (default kvip-glm53-w00568585)
  --sync-dir PATH        Staging directory, created automatically (default /tmp/li_qk_sync)
  --no-proxy             Bypass proxies only for this host Git fetch
  --offline-bundle PATH  Use the downloaded MR14 bundle; no network fetch
  -h, --help             Show usage
Copies /tmp/run_li_qk_a5.sh and /tmp/li_qk_mr14_offline.bundle into the target container.
It checks both copied files, then prints the container execution commands.
USAGE
}
while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --no-proxy) LI_NO_PROXY=1; shift ;;
        --container|--sync-dir|--offline-bundle)
            (($# >= 2)) && [[ -n $2 ]] || die "Missing value for $1"
            case "$1" in
                --container) LI_CONTAINER=$2 ;;
                --sync-dir) LI_SYNC_DIR=$2 ;;
                --offline-bundle) LI_BUNDLE=$2 ;;
            esac
            shift 2 ;;
        *) die "Unknown option: $1" ;;
    esac
done
for LI_CMD in docker git tee; do command -v "$LI_CMD" >/dev/null || die "Command unavailable: $LI_CMD. Run this script on the A5 Docker host"; done
docker info >/dev/null 2>&1 || die 'Cannot access Docker daemon. Run on the A5 host with Docker access; leave the container first if necessary'
LI_RUNNING=$(docker inspect --format '{{.State.Running}}' "$LI_CONTAINER") || die "Target container unavailable: $LI_CONTAINER"
[[ $LI_RUNNING == true ]] || die "Target container is not running: $LI_CONTAINER"
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
printf 'Host staging directory: %s\nTarget container: %s\n' "$LI_SYNC_DIR" "$LI_CONTAINER"
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
docker cp "$LI_SYNC_DIR/run_li_qk_a5.sh" "$LI_CONTAINER:/tmp/run_li_qk_a5.sh"
docker cp "$LI_SYNC_DIR/li_qk_mr14_offline.bundle" "$LI_CONTAINER:/tmp/li_qk_mr14_offline.bundle"
docker exec "$LI_CONTAINER" bash -c 'test -s /tmp/run_li_qk_a5.sh && test -s /tmp/li_qk_mr14_offline.bundle && ls -lh /tmp/run_li_qk_a5.sh /tmp/li_qk_mr14_offline.bundle'
printf '\nHost sync completed: %s\n' "$LI_HEAD"
printf 'Enter the target container:\n  docker exec -it %q bash\n' "$LI_CONTAINER"
printf 'Then activate the original Python/CANN environment and run:\n  bash /tmp/run_li_qk_a5.sh --repo /home/w00568585/wt-mr14 --git-bundle /tmp/li_qk_mr14_offline.bundle\n'
