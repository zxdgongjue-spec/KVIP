#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Isolate window admissions and terminal batching in ordinary ten-call graphs.
set -euo pipefail
repo=$(cd "${1:?expected KVIP checkout}" && pwd -P)
reader=$(git -C "$repo" rev-parse --verify "${2:?expected controller revision}^{commit}")
remote=https://gitcode.com/CacheOS/KVIP.git
archive=a00212cf124ccabda29e58e10821c234d832f4f6
moderate=0eed5be7dfb43ad070bc7a85bb636741620b8de2
medium=8d810939950b85f9d4c91d89e18efa094d5fb430
run_pinned() {
    local rev=$1 branch=$2 script=$3
    shift 3
    if ! git -C "$repo" cat-file -e "${rev}^{commit}" 2>/dev/null; then
        git -C "$repo" fetch -q --no-tags "$remote" "refs/heads/$branch"
    fi
    git -C "$repo" show "${rev}:tests/operators/lifetch/$script" | bash -s -- "$repo" "$rev" "$@"
}
mode=${3:-both}
batch=${4:-32}
case $batch in 32|28) ;; *) echo "Expected batch 32 or 28" >&2; exit 2;; esac
case $mode in
    window|tail|both)
        if [[ $mode != tail ]]; then
            run_pinned "$moderate" test/lifetch-five-window-224 g7_x4_npu4.sh moderate "$batch"
        fi
        if [[ $mode != window ]]; then
            run_pinned "$medium" test/lifetch-five-window-medium-tail g7_x4_npu4.sh medium "$batch"
        fi ;;
    archive) run_pinned "$archive" test/lifetch-g7-earned-tail-archive g7_earned_tail_archive.sh ;;
    receipt)
        export PEER_STAGGER_GROUPS=7 PEER_STAGGER_BATCHES="$batch"
        run_pinned "$reader" test/lifetch-final-first-cq-release first_cq_npu4.sh ;;
    *) echo "Usage: $0 REPO REV [both|window|tail|receipt|archive] [32|28]" >&2; exit 2 ;;
esac
