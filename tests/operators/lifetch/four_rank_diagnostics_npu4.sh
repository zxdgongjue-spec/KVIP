#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Collect only missing stand-alone diagnostic calls on the same card and original timing tree.
set -euo pipefail
repo=${1:?expected KVIP Git checkout}
baseline=$(cd "${2:?expected four-rank result root}" && pwd -P)
mode=${3:-all}
case $mode in tail|li|all|prefetch-ready) ;; *) echo 'Expected tail, li, prefetch-ready or all' >&2; exit 2 ;; esac
here=$(cd "$(dirname "$0")" && pwd -P)
work=$(cd "$baseline/../.." && pwd -P)
[[ ${baseline%/*} == "$work/peer-baseline" && -s $baseline/packed_summary.json &&
   $(<"$baseline/physical_npu.txt") == 4 && $(<"$baseline/selection_oracle.txt") == captured_logical_scores &&
   $(<"$baseline/host.txt") == "$(uname -n)" ]] || { echo "Wrong candidate archive $baseline" >&2; exit 2; }
if [[ -s /etc/machine-id && -s $baseline/machine_id.txt ]]; then
    cmp -s /etc/machine-id "$baseline/machine_id.txt" || { echo 'Machine ID differs'; exit 2; }
else
    cmp -s /proc/sys/kernel/random/boot_id "$baseline/boot_id.txt" || { echo 'Cannot verify archive host'; exit 2; }
fi
native=$(<"$baseline/native.txt")
sources="$work/fused-scan-sources"
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
unset PEER_BS32_LI PEER_PACKED_RESUME PEER_PACKED_REFERENCE KVAF_CPU_UB_DEVICE
bash "$here/large_batch_npu4.sh" --four-ranks-candidate-only
