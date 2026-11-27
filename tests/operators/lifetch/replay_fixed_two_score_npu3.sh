#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Screen the candidate with one ten-call graph per cell. Pass 2 as the fourth
# argument to repeat the candidate while still using both archived control blocks.
set -euo pipefail
repo=${1:?expected KVIP checkout}
rev=${2:?expected replay revision}
mode=${3:-two-cold}
export PEER_PACKED_BLOCKS=${4:-1}
case $mode in
    two-cold) export PEER_LARGE_COLD_WINDOWS=2 PEER_FOUR_WINDOW_CARRY=0
              batches='28 24 32 20'; groups='2 10 7 4 17' ;;
    carry)    export PEER_LARGE_COLD_WINDOWS=3 PEER_FOUR_WINDOW_CARRY=1
              batches='32 28 24 20'; groups='7 2 10 4 17' ;;
    *) echo "Expected two-cold or carry" >&2; exit 2 ;;
esac
case $PEER_PACKED_BLOCKS in 1|2) ;; *) echo "Expected one or two candidate blocks" >&2; exit 2 ;; esac
work=${PEER_G2_WORK:-/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold}
archive="$work/peer-baseline/fixed-q4.ygIfRl"
control="$work/peer-baseline/fixed-q4-packed.IWXS7N"
if [[ ! -s "$control/packed_summary.json" || ! -s "$archive/native.txt" ||
      $(<"$control/native.txt") != 10442f1ffa314606c23f54b73e49ecfc372f1f9a ||
      $(<"$control/physical_npu.txt") != 3 ]]; then
    echo "Expected the completed B24 NPU3 archive at $control and its inputs at $archive" >&2
    exit 1
fi
mkdir -p "$work/fused-scan-sources"
runner=$(mktemp -d "$work/fused-scan-sources/two-score-$mode-npu3.XXXXXX")
git -C "$repo" worktree add -q --detach "$runner" "$rev"
export PEER_G2_WORK="$work" PEER_LOCAL_ONLY=1
export PEER_PHYSICAL_NPU=3 ASCEND_RT_VISIBLE_DEVICES=3
export PEER_PACKED_REFERENCE="$control"
export FIXED_Q4_BATCHES="$batches" FIXED_Q4_GROUPS="$groups"
unset PEER_BS32_LI PEER_PACKED_RESUME KVAF_CPU_UB_DEVICE
exec bash "$runner/tests/operators/lifetch/large_batch_npu4.sh" --packed-aiv "$archive"
