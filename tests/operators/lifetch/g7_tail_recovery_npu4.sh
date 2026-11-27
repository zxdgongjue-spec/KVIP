#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Read the previous five-window results and run one chosen terminal experiment.
set -euo pipefail
repo=$(cd "${1:?expected KVIP checkout}" && pwd -P)
reader=$(git -C "$repo" rev-parse --verify "${2:?expected controller revision}^{commit}")
remote=https://gitcode.com/CacheOS/KVIP.git
archive=a00212cf124ccabda29e58e10821c234d832f4f6
run_pinned() {
    local rev=$1 branch=$2 script=$3
    shift 3
    if ! git -C "$repo" cat-file -e "${rev}^{commit}" 2>/dev/null; then
        git -C "$repo" fetch -q --no-tags "$remote" "refs/heads/$branch"
    fi
    git -C "$repo" show "${rev}:tests/operators/lifetch/$script" | bash -s -- "$repo" "$rev" "$@"
}
mode=${3:-receipt}
case $mode in
    archive|receipt) ;;
    *) echo "Usage: $0 REPO REV [archive|receipt]" >&2; exit 2 ;;
esac
run_pinned "$archive" test/lifetch-g7-earned-tail-archive g7_earned_tail_archive.sh
if [[ $mode == receipt ]]; then
    export PEER_STAGGER_GROUPS=7 PEER_STAGGER_BATCHES='32 28'
    run_pinned "$reader" test/lifetch-final-first-cq-release first_cq_npu4.sh
fi
