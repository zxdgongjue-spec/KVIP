#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
repo=${1:?expected KVIP checkout}
revision=${2:-$(git -C "$repo" rev-parse --verify 'FETCH_HEAD^{commit}')}
card=${3:-${PEER_PHYSICAL_NPU:-3}}
case $card in 3|4) ;; *) echo "Select physical NPU 3 or 4" >&2; exit 2 ;; esac
work=${PEER_G2_WORK:-/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold}
export PEER_G2_WORK="$work" PEER_PHYSICAL_NPU="$card" ASCEND_RT_VISIBLE_DEVICES="$card" PEER_LOCAL_ONLY=1
export PEER_PACKED_BLOCKS=1 PEER_LARGE_COLD_WINDOWS=3 PEER_FOUR_WINDOW_CARRY=0 PEER_FOUR_WINDOW_RANKS=1
export FIXED_Q4_BATCHES='32 28 24 20' FIXED_Q4_GROUPS='7 2 10 4 17'
unset PEER_BS32_LI PEER_PACKED_RESUME PEER_PACKED_REFERENCE KVAF_CPU_UB_DEVICE
mkdir -p "$work/fused-scan-sources" "$work/peer-baseline"
runner=$(mktemp -d "$work/fused-scan-sources/four-ranks-npu${card}.XXXXXX")
git -C "$repo" worktree add -q --detach "$runner" "$revision"
revision=$(git -C "$runner" rev-parse HEAD)
dir="$runner/tests/operators/lifetch"
log=$(mktemp "$work/host-npu${card}-four-window-ranks.XXXXXX")
identity=$(mktemp -d "$work/rank-host.XXXXXX")
trap 'rm -rf -- "$identity"' EXIT
py=/usr/local/python3.12.13/bin/python3
[[ -x $py ]] || py=python3
"$py" -B -c 'import os; print(os.uname().nodename)' > "$identity/host.txt"
printf '%s\n' "$card" > "$identity/physical_npu.txt"
if [[ -r /etc/machine-id ]]; then cp /etc/machine-id "$identity/machine_id.txt"; fi
cp /proc/sys/kernel/random/boot_id "$identity/boot_id.txt"
echo "RUN_LOG=$log"
echo "HOST=$(<"$identity/host.txt") physical_NPU=$card groups=7,2,10,4,17 BS=32,28,24,20 calls=10 candidate_blocks=1" | tee "$log"
"$py" -B "$dir/support/rank_replay_archives.py" "$work/peer-baseline" "$identity" > "$identity/paths"
mapfile -t paths < "$identity/paths"
[[ ${#paths[@]} == 3 ]]
if [[ ${paths[2]} != - ]]; then
    if ! "$py" -B "$dir/support/report_large_li_phases.py" "${paths[2]}" --json "$log.archive.json" 2>&1 | tee -a "$log"; then
        echo "Prior credit-policy report unavailable; continuing with the candidate" | tee -a "$log"
    fi
else
    echo 'Prior credit-policy diagnostics unavailable on this card' | tee -a "$log"
fi
if [[ ${paths[0]} != - ]]; then
    export PEER_PACKED_REFERENCE=${paths[0]}
    echo "CONTROL=$PEER_PACKED_REFERENCE source=${paths[1]}" | tee -a "$log"
    args=(--packed-aiv "${paths[1]}")
else
    echo 'CONTROL=UNAVAILABLE running candidate only with local input scores' | tee -a "$log"
    args=(--four-ranks-candidate-only)
fi
bash "$dir/large_batch_npu4.sh" "${args[@]}" | tee -a "$log"
new=$(sed -n 's/^REPLAY_COMPLETE cells=20 RESULT_ROOT=//p' "$log")
[[ -n $new && -s $new/packed_summary.json ]] || { echo "Incomplete candidate result; see $log" >&2; exit 1; }
"$py" -B "$dir/support/report_large_li_phases.py" "$new" --rank-revision "$revision" --json "$new/li_phase_comparison.json" | tee -a "$log"
"$py" -B - "$new/li_phase_comparison.json" <<'PYREPORT'
import json
import sys
from pathlib import Path

rows = json.loads(Path(sys.argv[1]).read_text())
expected = {(group, batch) for group in (7, 2, 10, 4, 17) for batch in (32, 28, 24, 20)}
if len(rows) != len(expected) or {(r["group"], r["batch"]) for r in rows} != expected:
    raise SystemExit("Incomplete twenty-cell LI report: " + sys.argv[1])
PYREPORT
printf 'PHASE_REPORT=%s\n' "$new/li_phase_comparison.json"
