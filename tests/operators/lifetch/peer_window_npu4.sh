#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run on either peer host. All paths used for comparison come from local receipts.
set -euo pipefail
repo=$(cd "${1:?KVIP repository required}" && pwd -P)
reader=$(git -C "$repo" rev-parse --verify "${2:?reader revision required}^{commit}")
mode=${3:-li}
work=${PEER_G2_WORK:-/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold}
base=d0fc5963881bd38fa97eff0f6ef40c757a6e8713
sources="$work/fused-scan-sources"
mkdir -p "$sources" "$work/peer-baseline"
stage() (
    local target="$sources/peer-npu4-$1"
    exec 8>"$target.lock"
    flock 8
    if [[ -d "$target" && $(git -C "$target" rev-parse HEAD 2>/dev/null) == "$1" &&
          -z $(git -C "$target" status --porcelain) ]]; then printf '%s\n' "$target"; return; fi
    if [[ -e "$target" ]]; then target=$(mktemp -d "$target.XXXXXX"); fi
    git -C "$repo" worktree add -q --detach "$target" "$1"
    printf '%s\n' "$target"
)
entry="$(stage "$reader")/tests/operators/lifetch"
py="$work/peer-baseline/python-da318bdb/bin/python"
if ! "$py" -B -c 'import numpy,hashlib; assert hasattr(hashlib,"file_digest")' 2>/dev/null; then
    py=/usr/local/python3.12.13/bin/python3
fi
export PEER_G2_WORK="$work" PEER_PHYSICAL_NPU=4 ASCEND_RT_VISIBLE_DEVICES=4
export PEER_LOCAL_ONLY=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset KVAF_CPU_UB_DEVICE PEER_FOUR_RANK_FOLLOWUP PEER_FOUR_RANK_AUDIT PEER_PACKED_RESUME PEER_RANK_DIAG_SOURCE PEER_BS32_LI PEER_LOCAL_WINDOW_CLOCKS
export PEER_FOUR_WINDOW_RANKS=1 PEER_LARGE_COLD_WINDOWS=3 PEER_FOUR_WINDOW_CARRY=0
helper="$entry/support/peer_npu4_evidence.py"
if [[ $mode == tail ]]; then
    exec "$py" -B "$entry/support/portable_two_row_audit.py" "$work"
fi
[[ $mode == li ]] || { echo "Usage: $0 REPO READER li|tail" >&2; exit 2; }
ref=$("$py" -B "$helper" origin "$work/peer-baseline")
stamp=$(mktemp -d "$work/peer-window-evidence.XXXXXX")
exec > >(tee "$stamp/run.log") 2>&1
echo "HOST=$(uname -n) physical_NPU=4 LOCAL_FOUR_WINDOW_ARCHIVE=${ref:-UNAVAILABLE} LOG=$stamp/run.log"
export PEER_TIMED_SOURCE
PEER_TIMED_SOURCE=$(stage "$base")
export FIXED_Q4_GROUPS=2
local_shapes=()
paired_shapes=()
for b in 28 32; do
    if [[ -n $ref ]]; then
        paired=$("$py" -B "$helper" paired "$ref" "$b")
        if [[ -n $paired ]]; then
            old=$("$py" -B "$helper" paired-clock "$ref" "$reader" "$b")
            if [[ -n $old ]]; then
                echo "REUSED G2 BS$b paired-window-clocks source=$old"
            else
                paired_shapes+=("$b")
            fi
            continue
        fi
    fi
    old=$("$py" -B "$helper" clock "$work/peer-baseline" "$reader" "$b")
    if [[ -n $old ]]; then
        printf '%s\n' "$old" > "$stamp/g2-bs$b.txt"
        echo "REUSED G2 BS$b local-window-clocks source=$old"
    else
        local_shapes+=("$b")
    fi
done
if (( ${#paired_shapes[@]} )); then
    PEER_AUDIT_ORIGIN=$ref PEER_AUDIT_REFERENCE=$ref PEER_FOUR_RANK_AUDIT=g2-window-clocks FIXED_Q4_BATCHES="${paired_shapes[*]}" \
        bash "$entry/large_batch_npu4.sh" --four-ranks-candidate-only
fi
if (( ${#local_shapes[@]} )); then
    echo "LOCAL_DIAGNOSTIC G2 BS=${local_shapes[*]} timed_calls=0 per_shape=1 serial_runs=0"
    PEER_LOCAL_WINDOW_CLOCKS=1 FIXED_Q4_BATCHES="${local_shapes[*]}" \
        bash "$entry/large_batch_npu4.sh" --four-ranks-candidate-only | tee "$stamp/local.log"
    result=$(sed -n 's/^RESULT_ROOT=//p' "$stamp/local.log")
    for b in "${local_shapes[@]}"; do
        arm="$result/g2/bs$b/block1/clocks"
        [[ -s $arm/single_diagnostic.json && -f $arm/local_placement_measured ]]
        printf '%s\n' "$arm" > "$stamp/g2-bs$b.txt"
    done
fi
"$py" -B "$helper" package "$stamp" "$ref" "$reader"
