#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Collect only missing stand-alone diagnostic calls on the same card and original timing tree.
set -euo pipefail
repo=${1:?expected KVIP Git checkout}
here=$(cd "$(dirname "$0")" && pwd -P)
mode=${3:-all}
arg=${2:?local reference directory or AUTO required}
work=${PEER_G2_WORK:-/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold}
py="$work/peer-baseline/python-da318bdb/bin/python"
if ! "$py" -B -c 'import numpy, hashlib; assert hasattr(hashlib,"file_digest")' 2>/dev/null; then py=/usr/local/python3.12.13/bin/python3; fi
if [[ $mode == window-tails ]]; then
    for task in window-tail-x2 window-tail-x4 window-tail-x3; do bash "$0" "$repo" "$arg" "$task"; done
    exit 0
fi
case $mode in tail|li|all|prefetch-ready|window-tail-x2|window-tail-x3|window-tail-x4|window-tail-x3-g7) ;; *) echo "Unknown diagnostic $mode" >&2; exit 2 ;; esac
if [[ $arg == AUTO ]]; then
    [[ $mode == window-tail-x* ]] || { echo 'AUTO requires a window-tail diagnostic' >&2; exit 2; }
    baseline=$("$py" -B "$here/support/b24_window_tail_diagnostic.py" --origin "$work/peer-baseline" --mode "$mode")
else
    baseline=$(cd "$arg" && pwd -P)
fi
if [[ $baseline == LOCAL ]]; then
    native=d0fc5963881bd38fa97eff0f6ef40c757a6e8713
    echo "LOCAL_UNPAIRED measured_calls=0 serial_runs=0 physical_NPU=4"
else
    work=$(cd "$baseline/../.." && pwd -P)
    [[ ${baseline%/*} == "$work/peer-baseline" && -s $baseline/packed_summary.json &&
       $(<"$baseline/physical_npu.txt") == 4 && $(<"$baseline/selection_oracle.txt") == captured_logical_scores &&
       $(<"$baseline/host.txt") == "$(uname -n)" ]] || { echo "Wrong candidate archive $baseline" >&2; exit 2; }
    if [[ -s /etc/machine-id && -s $baseline/machine_id.txt ]]; then cmp -s /etc/machine-id "$baseline/machine_id.txt"; else cmp -s /proc/sys/kernel/random/boot_id "$baseline/boot_id.txt"; fi
    native=$(<"$baseline/native.txt")
fi
sources="$work/fused-scan-sources"
mkdir -p "$sources"
picked=
for candidate in "$sources"/four-ranks-npu4.*; do
    if [[ -f $candidate/.git ]] && [[ $(git -C "$candidate" rev-parse HEAD) == "$native" ]] &&
        git -C "$candidate" diff --quiet HEAD --; then picked=$candidate; break; fi
done
if [[ -z $picked ]]; then
    picked=$(mktemp -d "$sources/four-ranks-npu4.XXXXXX")
    git -C "$repo" worktree add -q --detach "$picked" "$native"
fi
export PEER_G2_WORK="$work" PEER_PHYSICAL_NPU=4 ASCEND_RT_VISIBLE_DEVICES=4 PEER_LOCAL_ONLY=1
export PEER_TIMED_SOURCE="$picked" PEER_RANK_DIAG_SOURCE="$baseline" PEER_RANK_DIAG_MODE="$mode"
export PEER_LARGE_COLD_WINDOWS=3 PEER_FOUR_WINDOW_CARRY=0 PEER_FOUR_WINDOW_RANKS=1
unset PEER_BS32_LI PEER_PACKED_RESUME PEER_PACKED_REFERENCE PEER_PACKED_BLOCKS PEER_FOUR_RANK_AUDIT
unset PEER_FOUR_RANK_FOLLOWUP PEER_AUDIT_REFERENCE PEER_LOCAL_WINDOW_CLOCKS PEER_LOCAL_WINDOW_POLICY
unset PEER_FOUR_RANK_SCOPE PEER_FOUR_RANK_NEW_DEVICE_BUILD PEER_FOUR_RANK_SAME_PACING PEER_PORTABLE_TAIL
unset FIXED_Q4_BATCHES FIXED_Q4_GROUPS KVAF_CPU_UB_DEVICE PEER_RANKED_INPUT PEER_RANKED_HASH_FILE
unset KVIP_BS32_SHARE_FINAL KVIP_EXPERIMENT_SPLIT_ROLE KVIP_TEST_WINDOW_RANK1 KVIP_TEST_WINDOW_RANK2
bash "$here/large_batch_npu4.sh" --four-ranks-candidate-only
