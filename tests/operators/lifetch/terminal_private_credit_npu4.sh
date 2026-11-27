#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run from the exact candidate checkout on B24. The two arguments identify the existing original arm.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
repo=$(cd "$here/../../.." && pwd -P)
control=$(cd "${1:?expected original 20-cell four-rank result directory}" && pwd -P)
original_clocks=${2:?expected original four-rank-diagnostics.tar.gz}
work=$(cd "$control/../.." && pwd -P)
[[ -f $original_clocks && -f $control/packed_summary.json && $(<"$control/physical_npu.txt") == 4 ]]
git -C "$repo" diff --quiet HEAD --
export PEER_G2_WORK="$work" PEER_TIMED_SOURCE="$repo" PEER_PHYSICAL_NPU=4 ASCEND_RT_VISIBLE_DEVICES=4
export FIXED_Q4_BATCHES='32 28 24 20' FIXED_Q4_GROUPS='7 2 10 4 17'
export PEER_LARGE_COLD_WINDOWS=3 PEER_FOUR_WINDOW_CARRY=0 PEER_FOUR_WINDOW_RANKS=1
unset PEER_BS32_LI PEER_PACKED_RESUME PEER_PACKED_REFERENCE PEER_RANK_DIAG_SOURCE PEER_RANK_DIAG_MODE
log=$(mktemp "$work/terminal-private-credit-npu4.XXXXXX")
bash "$here/large_batch_npu4.sh" --four-ranks-candidate-only | tee "$log"
new=$(sed -n 's/^REPLAY_COMPLETE cells=20 RESULT_ROOT=//p' "$log")
[[ -n $new && -s $new/packed_summary.json ]]
bash "$here/four_rank_diagnostics_npu4.sh" "$repo" "$new" tail | tee -a "$log"
diag=$(sed -n 's/^DIAGNOSTICS_COMPLETE cells=2 RESULT_ROOT=//p' "$log")
[[ -n $diag && -f $diag/four-rank-diagnostics.tar.gz ]]
py=/usr/local/python3.12.13/bin/python3
[[ -x $py ]] || py=python3
"$py" -B "$here/support/compare_private_prefix_credit.py" "$control" "$new" "$original_clocks" \
    "$diag/four-rank-diagnostics.tar.gz" | tee "$new/tail-credit-comparison.json"
