#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Read the completed credit experiment first. Time only the new fixed-budget candidate.
set -euo pipefail
repo=${1:?expected KVIP checkout}
work=${PEER_G2_WORK:-/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold}
rank=afbdcd1ddc6f32f3a0a6bc71e78bf7345d79ef2c
reader=7d93ea05a2efcea694051fad46293574802d021b
entry=tests/operators/lifetch
old="$work/peer-baseline/fixed-q4-four-window-carry-one-block.VFkffz"
log=$(mktemp "$work/host24-npu3-four-window-ranks.XXXXXX")
report=$(mktemp "$work/li-phase-reader.XXXXXX")
trap 'rm -f "$report"' EXIT
git -C "$repo" show "${reader}:$entry/support/report_large_li_phases.py" > "$report"
py=/usr/local/python3.12.13/bin/python3
if [[ ! -x $py ]]; then py=python3; fi
printf 'RUN_LOG=%s\n' "$log"
"$py" -B "$report" "$old" --json "$log.archive.json" | tee "$log"
export PEER_G2_WORK="$work"
unset FIXED_Q4_BATCHES FIXED_Q4_GROUPS
git -C "$repo" show "${rank}:$entry/replay_fixed_two_score_npu3.sh" |
    bash -s -- "$repo" "$rank" four-ranks 1 | tee -a "$log"
new=$(sed -n 's/^REPLAY_COMPLETE cells=20 RESULT_ROOT=//p' "$log")
[[ -n $new && -f "$new/packed_summary.json" ]] || {
    printf 'Missing completed twenty-case rank result in %s\n' "$log" >&2
    exit 1
}
"$py" -B "$report" "$new" --rank-revision "$rank" --json "$new/li_phase_comparison.json" | tee -a "$log"
printf 'PHASE_REPORT=%s\n' "$new/li_phase_comparison.json"
