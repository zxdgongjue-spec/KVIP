#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# NPU3 is the card with the completed 45-cell strong serial reference.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
repo=$(cd "$here/../../.." && pwd -P)
work=${PEER_G2_WORK:-/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold}
control=$work/peer-baseline/fixed-q4-packed.IWXS7N
inputs=$work/peer-baseline/fixed-q4.ygIfRl
original=d0fc5963881bd38fa97eff0f6ef40c757a6e8713
[[ -f $control/packed_summary.json && $(<"$control/packed_source.txt") == "$inputs" &&
   $(<"$control/physical_npu.txt") == 3 ]]
git -C "$repo" diff --quiet HEAD --
export PEER_G2_WORK="$work" PEER_PHYSICAL_NPU=3 ASCEND_RT_VISIBLE_DEVICES=3
export FIXED_Q4_GROUPS='7 2 10 4 17' FIXED_Q4_BATCHES='32 28 24 20'
export PEER_LARGE_COLD_WINDOWS=3 PEER_FOUR_WINDOW_CARRY=0 PEER_FOUR_WINDOW_RANKS=1
export PEER_PACKED_BLOCKS=1 PEER_PACKED_REFERENCE="$control"
unset PEER_BS32_LI PEER_PACKED_RESUME PEER_RANK_DIAG_SOURCE PEER_RANK_DIAG_MODE
run_tmp=$(mktemp -d "$work/fused-scan-sources/terminal-private-credit.XXXXXX")
if [[ ${1:-} == small ]]; then
    export FIXED_Q4_BATCHES='16 8 4 2 1' PEER_FOUR_WINDOW_RANKS=0 PEER_TIMED_SOURCE="$repo"
    bash "$here/large_batch_npu4.sh" --packed-aiv "$inputs"
    exit
fi
[[ $# == 0 ]] || { echo 'Usage: terminal_private_credit_npu3.sh [small]' >&2; exit 2; }
git -C "$repo" worktree add -q --detach "$run_tmp/original" "$original"
for arm in original candidate; do
    if [[ $arm == original ]]; then export PEER_TIMED_SOURCE="$run_tmp/original"; else export PEER_TIMED_SOURCE="$repo"; fi
    bash "$here/large_batch_npu4.sh" --packed-aiv "$inputs" | tee "$run_tmp/$arm.log"
done
old=$(sed -n 's/^REPLAY_COMPLETE cells=20 RESULT_ROOT=//p' "$run_tmp/original.log")
new=$(sed -n 's/^REPLAY_COMPLETE cells=20 RESULT_ROOT=//p' "$run_tmp/candidate.log")
[[ -s $old/packed_summary.json && -s $new/packed_summary.json ]]
py=/usr/local/python3.12.13/bin/python3
[[ -x $py ]] || py=python3
"$py" -B "$here/support/compare_private_prefix_credit.py" "$old" "$new" | tee "$new/terminal-credit-comparison.json"
