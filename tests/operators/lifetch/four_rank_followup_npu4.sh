#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Can run from git show | bash -s. All sources are obtained through the local Git repository.
set -euo pipefail
repo=$(cd "${1:?expected KVIP checkout}" && pwd -P)
controller=$(git -C "$repo" rev-parse --verify "${2:?expected runner revision}^{commit}")
candidate=${3:?expected candidate revision or worktree}
ref=$(cd "${4:?expected fixed four-window archive}" && pwd -P)
scope=${5:-focus}
case $scope in focus|all) ;; *) echo 'Expected focus or all' >&2; exit 2 ;; esac
work=$(cd "$ref/../.." && pwd -P)
[[ ${ref%/*} == "$work/peer-baseline" && -s $ref/packed_summary.json &&
    $(<"$ref/physical_npu.txt") == 4 && $(<"$ref/host.txt") == "$(uname -n)" ]] ||
    { echo "Unexpected archive or host: $ref" >&2; exit 2; }
if [[ -s /etc/machine-id && -s $ref/machine_id.txt ]]; then
    cmp -s /etc/machine-id "$ref/machine_id.txt" || { echo 'Archive belongs to a different host' >&2; exit 2; }
else
    cmp -s /proc/sys/kernel/random/boot_id "$ref/boot_id.txt" || { echo 'Cannot identify archive host' >&2; exit 2; }
fi
sources="$work/fused-scan-sources"
mkdir -p "$sources"
ensure_tree() (
    local sha=$1 target="$sources/four-rank-src-$1" path
    exec 8>"$target.lock"; flock 8
    if [[ -d $target ]] && [[ $(git -C "$target" rev-parse HEAD 2>/dev/null) == "$sha" ]] &&
        [[ -z $(git -C "$target" status --porcelain) ]]; then
        printf '%s\n' "$target"; return
    fi
    path=$target
    if [[ -e $path ]]; then path=$(mktemp -d "$target.XXXXXX"); fi
    git -C "$repo" worktree add -q --detach "$path" "$sha"
    printf '%s\n' "$path"
)
if [[ -d $candidate ]]; then
    timed=$(cd "$candidate" && pwd -P)
    [[ -z $(git -C "$timed" status --porcelain) ]] || { echo "Candidate worktree has local changes: $timed" >&2; exit 2; }
    native=$(git -C "$timed" rev-parse --verify HEAD)
else
    native=$(git -C "$repo" rev-parse --verify "${candidate}^{commit}")
    timed=$(ensure_tree "$native")
fi
[[ $native != "$(<"$ref/native.txt")" ]] || { echo 'Candidate is the archived baseline' >&2; exit 2; }
control=$(ensure_tree "$controller")
test -f "$timed/tests/operators/lifetch/support/large_two_cold_native/kernel.cpp"
export PEER_G2_WORK="$work" PEER_PHYSICAL_NPU=4 ASCEND_RT_VISIBLE_DEVICES=4 PEER_LOCAL_ONLY=1
export PEER_FOUR_RANK_FOLLOWUP="$ref" PEER_FOUR_RANK_SCOPE="$scope" PEER_TIMED_SOURCE="$timed"
export PEER_LARGE_COLD_WINDOWS=3 PEER_FOUR_WINDOW_CARRY=0 PEER_FOUR_WINDOW_RANKS=1
export FIXED_Q4_GROUPS="7 2 10" FIXED_Q4_BATCHES="32 28"
if [[ $scope == all ]]; then export FIXED_Q4_GROUPS="7 2 10 4 17" FIXED_Q4_BATCHES="32 28 24 20"; fi
unset PEER_BS32_LI PEER_RANK_DIAG_SOURCE PEER_PACKED_REFERENCE PEER_PACKED_RESUME KVAF_CPU_UB_DEVICE
bash "$control/tests/operators/lifetch/large_batch_npu4.sh" --four-ranks-candidate-only
