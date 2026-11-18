#!/usr/bin/env bash
# Tracked container entry; also published into the existing shared directory.
set -Eeuo pipefail
LI_TAIL_ENTRY_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
LI_TAIL_ROOT=/home/w00568585/li_qk_shared
if [[ -L $LI_TAIL_ENTRY_DIR/current ]]; then LI_TAIL_ROOT=$LI_TAIL_ENTRY_DIR; fi
LI_TAIL_FORWARD=()
while (($#)); do
    case "$1" in
        --share-root)
            (($# >= 2)) && [[ -n $2 ]] || { echo 'ERROR: --share-root requires a directory' >&2; exit 2; }
            LI_TAIL_ROOT=$2
            shift 2 ;;
        *) LI_TAIL_FORWARD+=("$1"); shift ;;
    esac
done
if [[ ! -d $LI_TAIL_ROOT/current ]]; then
    printf 'ERROR: shared release unavailable: %s/current\n' "$LI_TAIL_ROOT" >&2
    printf 'On the A5 host, run the updated repository script:\n  bash experiments/li_qk/update_shared_a5.sh --directory %q --no-proxy\n' "$LI_TAIL_ROOT" >&2
    exit 1
fi
command -v git >/dev/null || { echo 'ERROR: Git is required to verify the published release' >&2; exit 1; }
LI_TAIL_RELEASE=$(cd "$LI_TAIL_ROOT/current" && pwd -P)
[[ -s $LI_TAIL_RELEASE/run_tail_a5.sh && -s $LI_TAIL_RELEASE/TAIL_SCRIPT_BLOB ]] || {
    echo 'ERROR: published revision lacks post-score controls; update MR14 on the host' >&2; exit 1;
}
[[ $(git hash-object --no-filters "$LI_TAIL_RELEASE/run_tail_a5.sh") == "$(cat "$LI_TAIL_RELEASE/TAIL_SCRIPT_BLOB")" ]] || {
    echo 'ERROR: shared post-score launcher hash mismatch' >&2; exit 1;
}
if ((${#LI_TAIL_FORWARD[@]})); then
    exec bash "$LI_TAIL_RELEASE/run_tail_a5.sh" --share-root "$LI_TAIL_ROOT" --release-root "$LI_TAIL_RELEASE" "${LI_TAIL_FORWARD[@]}"
fi
exec bash "$LI_TAIL_RELEASE/run_tail_a5.sh" --share-root "$LI_TAIL_ROOT" --release-root "$LI_TAIL_RELEASE"
