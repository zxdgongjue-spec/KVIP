#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
repo=$(cd "${1:?expected KVIP checkout}" && pwd -P)
reader=$(git -C "$repo" rev-parse --verify "${2:?expected reader}^{commit}")
work=${PEER_G2_WORK:-/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold}
case ${3:-original} in
    original) window=window-x4-early;;
    earned) window=window-x4-earned;;
    medium) window=window-x4-earned-medium;;
    *) echo "Expected original, earned or medium" >&2; exit 2;;
esac
sources="$work/fused-scan-sources"
area="$work/peer-baseline"
mkdir -p "$sources" "$area"
stage() (
    local sha=$1 path="$sources/g7-x4-$1"
    exec 8>"$path.lock"; flock 8
    if [[ -d $path && $(git -C "$path" rev-parse HEAD 2>/dev/null) == "$sha" && -z $(git -C "$path" status --porcelain -uno) ]]; then
        echo "$path"; return
    fi
    if [[ -e $path ]]; then path=$(mktemp -d "$path.XXXXXX"); fi
    git -C "$repo" worktree add -q --detach "$path" "$sha"
    echo "$path"
)
entry="$(stage "$reader")/tests/operators/lifetch"
py="$area/python-da318bdb/bin/python"
if ! "$py" -B -c 'import hashlib,numpy; assert hasattr(hashlib,"file_digest")' 2>/dev/null; then py=/usr/local/python3.12.13/bin/python3; fi
export PEER_G2_WORK="$work" PEER_PHYSICAL_NPU=4 ASCEND_RT_VISIBLE_DEVICES=4 PEER_LOCAL_ONLY=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export FIXED_Q4_GROUPS=7 PEER_FOUR_WINDOW_RANKS=1 PEER_LARGE_COLD_WINDOWS=3 PEER_FOUR_WINDOW_CARRY=0
unset KVAF_CPU_UB_DEVICE PEER_PACKED_RESUME PEER_RANK_DIAG_SOURCE PEER_FOUR_RANK_FOLLOWUP PEER_BS32_LI
helper="$entry/support/g7_x4_local.py"
batches=(32 28)
if [[ $window == window-x4-earned-medium ]]; then
    case ${4:-32} in
        32|28) batches=("${4:-32}");;
        both) batches=(32 28);;
        *) echo "Expected medium-tail batch 32, 28 or both" >&2; exit 2;;
    esac
fi
for batch in "${batches[@]}"; do
    selection=$("$py" -B "$helper" choose "$area" "$batch" "$reader" --mode "$window")
    IFS=$'\t' read -r mode ref hit <<<"$selection"
    echo "G7 BS$batch source=$mode physical_NPU=4"
    need=0
    case $mode in
        paired-cache) "$py" -B "$entry/support/b24_window_report.py" show "$hit" "$ref" 7 "$batch" "$window";;
        local-cache) "$py" -B "$helper" show-local "$hit" "$batch" --mode "$window";;
        paired-run) export PEER_G7_X4_LOCAL=0; need=1;;
        local-run) export PEER_G7_X4_LOCAL=1; ref=$area; need=1;;
        *) echo "Unknown local G7 selection: $mode" >&2; exit 1;;
    esac
    if (( need )); then
        export PEER_TIMED_SOURCE
        PEER_TIMED_SOURCE=$(stage d0fc5963881bd38fa97eff0f6ef40c757a6e8713)
        export FIXED_Q4_BATCHES=$batch PEER_G7_X4_ARCHIVE_BATCH=$batch
        export PEER_FOUR_RANK_AUDIT="$window" PEER_AUDIT_ORIGIN="$ref" PEER_AUDIT_REFERENCE="$ref"
        bash "$entry/large_batch_npu4.sh" --four-ranks-candidate-only
    fi
    if [[ $window == window-x4-earned-medium ]]; then "$py" -B "$helper" compare-medium "$area" "$batch" "$reader"; fi
    if [[ $window == window-x4-earned ]]; then "$py" -B "$helper" compare "$area" "$batch" "$reader"; fi
done
