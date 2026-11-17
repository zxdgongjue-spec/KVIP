#!/usr/bin/env bash
# Host: update files in an existing shared mount. Never launches LI or a container CLI.
set -Eeuo pipefail
LI_SHARE=/home/w00568585/li_qk_shared
LI_BRANCH=codex/split-core-unified-comparison
LI_REF=refs/remotes/origin/codex/split-core-unified-comparison
LI_BUNDLE=
LI_NO_PROXY=0
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
usage() {
    cat <<'USAGE'
Usage: bash update_shared_a5.sh [options]
Run on the host. Uses the existing /home/w00568585 -> /home/w00568585 mount.
  --directory PATH       Shared package directory (default /home/w00568585/li_qk_shared)
  --no-proxy             Bypass proxies only for the host Git fetch
  --offline-bundle PATH  Import an existing MR14 bundle instead of fetching GitCode
  -h, --help             Show help
Then, in the A5 container, run the shared directory's run_in_container.sh.
This script updates files only; it does not create a new mount or run NPU kernels.
USAGE
}
while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --no-proxy) LI_NO_PROXY=1; shift ;;
        --directory|--offline-bundle)
            (($# >= 2)) && [[ -n $2 ]] || die "Missing value for $1"
            case "$1" in --directory) LI_SHARE=$2 ;; --offline-bundle) LI_BUNDLE=$2 ;; esac
            shift 2 ;;
        *) die "Unknown option: $1" ;;
    esac
done
for LI_CMD in git tee mktemp; do command -v "$LI_CMD" >/dev/null || die "Command unavailable: $LI_CMD"; done
if [[ -n $LI_BUNDLE ]]; then
    [[ -s $LI_BUNDLE ]] || die "Bundle unavailable: $LI_BUNDLE"
    LI_BUNDLE=$(cd "$(dirname "$LI_BUNDLE")" && pwd -P)/$(basename "$LI_BUNDLE")
fi
mkdir -p "$LI_SHARE/releases"
LI_SHARE=$(cd "$LI_SHARE" && pwd -P)
LI_REPO=$LI_SHARE/.host-repo.git
if [[ ! -e $LI_REPO ]]; then
    git init --bare "$LI_REPO"
    git --git-dir="$LI_REPO" remote add origin https://gitcode.com/CacheOS/KVIP.git
else
    [[ $(git --git-dir="$LI_REPO" rev-parse --is-bare-repository) == true ]] || die "Expected a bare host cache: $LI_REPO"
fi
if [[ -n $LI_BUNDLE ]]; then
    git --git-dir="$LI_REPO" bundle verify "$LI_BUNDLE"
    git --git-dir="$LI_REPO" fetch "$LI_BUNDLE" "$LI_REF" 2>&1 | tee "$LI_SHARE/update.log"
else
    LI_GIT=(git)
    if ((LI_NO_PROXY)); then
        LI_GIT=(env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy
            git -c http.proxy= -c remote.origin.proxy=)
    fi
    "${LI_GIT[@]}" --git-dir="$LI_REPO" fetch origin "$LI_BRANCH" 2>&1 | tee "$LI_SHARE/update.log"
fi
LI_HEAD=$(git --git-dir="$LI_REPO" rev-parse --verify FETCH_HEAD)
git --git-dir="$LI_REPO" merge-base --is-ancestor 0c1d483 "$LI_HEAD" || die 'Synced source does not include the seven-point LI controls'
git --git-dir="$LI_REPO" update-ref "$LI_REF" "$LI_HEAD"
LI_RELEASE=$(mktemp -d "$LI_SHARE/releases/${LI_HEAD:0:12}.XXXXXX")
git --git-dir="$LI_REPO" show "$LI_HEAD:experiments/li_qk/run_a5.sh" > "$LI_RELEASE/run_li_qk_a5.sh"
bash -n "$LI_RELEASE/run_li_qk_a5.sh"
git --git-dir="$LI_REPO" bundle create "$LI_RELEASE/li_qk_mr14_offline.bundle" "$LI_REF"
git --git-dir="$LI_REPO" bundle verify "$LI_RELEASE/li_qk_mr14_offline.bundle"
printf '%s\n' "$LI_HEAD" > "$LI_RELEASE/SOURCE_COMMIT"
git hash-object --no-filters "$LI_RELEASE/run_li_qk_a5.sh" > "$LI_RELEASE/SCRIPT_BLOB"
git hash-object --no-filters "$LI_RELEASE/li_qk_mr14_offline.bundle" > "$LI_RELEASE/BUNDLE_BLOB"
cat > "$LI_SHARE/run_in_container.sh.pending" <<'LI_CONTAINER_ENTRY_END'
#!/usr/bin/env bash
# Container: read a frozen shared release, then run the physical NPU benchmark.
set -Eeuo pipefail
LI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
LI_RELEASE=$(cd "$LI_ROOT/current" && pwd -P) || { echo 'ERROR: shared release is absent; update files on the host first' >&2; exit 1; }
for LI_FILE in run_li_qk_a5.sh li_qk_mr14_offline.bundle SOURCE_COMMIT SCRIPT_BLOB BUNDLE_BLOB; do
    [[ -s $LI_RELEASE/$LI_FILE ]] || { printf 'ERROR: shared file missing: %s\n' "$LI_FILE" >&2; exit 1; }
done
command -v git >/dev/null || { echo 'ERROR: Git is required in the container for local bundle import' >&2; exit 1; }
[[ $(git hash-object --no-filters "$LI_RELEASE/run_li_qk_a5.sh") == "$(cat "$LI_RELEASE/SCRIPT_BLOB")" ]] || { echo 'ERROR: shared launcher hash mismatch' >&2; exit 1; }
[[ $(git hash-object --no-filters "$LI_RELEASE/li_qk_mr14_offline.bundle") == "$(cat "$LI_RELEASE/BUNDLE_BLOB")" ]] || { echo 'ERROR: shared code bundle hash mismatch' >&2; exit 1; }
printf 'Shared source visible. Published MR14 commit: %s\n' "$(cat "$LI_RELEASE/SOURCE_COMMIT")"
if [[ ${1:-} == --verify-only ]]; then
    printf 'Mapping verified; benchmark has not run.\n'
    exit 0
fi
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    printf 'Run in the original A5 container/Python/CANN environment.\n'
    printf 'All arguments forward to the LI runner: --device 5 --aics 32 --aivs 64 --kvaf-include PATH --cann-root PATH\n'
    exit 0
fi
[[ ${LD_PRELOAD:-} != *libruntime_camodel* ]] || { echo 'ERROR: physical NPU execution required; Model preload is not allowed' >&2; exit 1; }
command -v npu-smi >/dev/null || { echo 'ERROR: use the A5 container with NPU/driver/CANN access' >&2; exit 1; }
npu-smi info
# Do not use the host cache's worktree links from inside the container.
LI_REPO=/home/w00568585/wt-mr14
if ! git -C "$LI_REPO" rev-parse --git-dir >/dev/null 2>&1; then
    LI_REPO=$(mktemp -d /tmp/li_qk_container_repo.XXXXXX)
    git init --bare "$LI_REPO"
fi
exec bash "$LI_RELEASE/run_li_qk_a5.sh" --repo "$LI_REPO" \
    --git-bundle "$LI_RELEASE/li_qk_mr14_offline.bundle" "$@"
LI_CONTAINER_ENTRY_END
bash -n "$LI_SHARE/run_in_container.sh.pending"
mv "$LI_SHARE/run_in_container.sh.pending" "$LI_SHARE/run_in_container.sh"
if [[ -e $LI_SHARE/current && ! -L $LI_SHARE/current ]]; then die 'current must be the updater release symlink; existing directory was preserved'; fi
ln -sfn "releases/$(basename "$LI_RELEASE")" "$LI_SHARE/current"
printf '\nHost code update complete: %s\nShared directory: %s\n' "$LI_HEAD" "$LI_SHARE"
printf 'In the A5 container, verify the existing mapping:\n  bash %q/run_in_container.sh --verify-only\n' "$LI_SHARE"
printf 'Then run the seven-point physical NPU experiment:\n  bash %q/run_in_container.sh\n' "$LI_SHARE"
