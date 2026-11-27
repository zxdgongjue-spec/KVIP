#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Call through git show <exact-reader>:<this path> | bash -s -- REPO READER CANDIDATE
set -euo pipefail
repo=$(cd "${1:?expected KVIP checkout}" && pwd -P)
reader=$(git -C "$repo" rev-parse --verify "${2:?expected reader commit}^{commit}")
mode=${3:?expected candidate commit or serial-only}
extra=${4:-natural}
[[ $extra == natural || ( $extra == paired-frozen && $mode != serial-only ) ]] || exit 2
native=
if [[ $mode != serial-only ]]; then native=$(git -C "$repo" rev-parse --verify "$mode^{commit}"); fi
base=d0fc5963881bd38fa97eff0f6ef40c757a6e8713
serial=10442f1ffa314606c23f54b73e49ecfc372f1f9a
work=/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold
ref=$(cd "$work/peer-baseline/fixed-q4-four-ranks-one-block.MRDDaH" && pwd -P)
[[ $(<"$ref/physical_npu.txt") == 4 && $(<"$ref/native.txt") == "$base" &&
   $(<"$ref/host.txt") == "$(uname -n)" && $native != "$base" && $native != "$reader" ]]
sources="$work/fused-scan-sources"
mkdir -p "$sources"
staged_tree() (
    local sha=$1 d="$sources/four-rank-accept-$1"
    exec 8>"$d.lock"
    flock 8
    if [[ -d $d && $(git -C "$d" rev-parse HEAD 2>/dev/null) == "$sha" &&
        -z $(git -C "$d" status --porcelain) ]]; then printf '%s\n' "$d"; return; fi
    if [[ -e $d ]]; then d=$(mktemp -d "$d.XXXXXX"); fi
    git -C "$repo" worktree add -q --detach "$d" "$sha"
    printf '%s\n' "$d"
)
control=$(staged_tree "$reader")
entry="$control/tests/operators/lifetch"
stamp=$(mktemp -d "$work/four-rank-accept.XXXXXX")
export PYTHONDONTWRITEBYTECODE=1 PEER_G2_WORK="$work" PEER_PHYSICAL_NPU=4 ASCEND_RT_VISIBLE_DEVICES=4
export PEER_LOCAL_ONLY=1 PEER_AUDIT_ORIGIN="$ref" PEER_FOUR_WINDOW_RANKS=1 PEER_LARGE_COLD_WINDOWS=3 PEER_FOUR_WINDOW_CARRY=0
export FIXED_Q4_GROUPS='7 2 10 4 17' FIXED_Q4_BATCHES='32 28 24 20'
unset PEER_FOUR_RANK_AUDIT PEER_FOUR_RANK_FOLLOWUP PEER_PACKED_RESUME PEER_RANK_DIAG_SOURCE PEER_BS32_LI KVAF_CPU_UB_DEVICE
py="$work/peer-baseline/python-da318bdb/bin/python"
if ! "$py" -B -c 'import numpy, hashlib; assert hasattr(hashlib, "file_digest")' 2>/dev/null; then
    py=/usr/local/python3.12.13/bin/python3
fi
audit() {
    local timed rc
    if "$py" -B "$entry/support/four_rank_accept.py" coverage "$3" "$1"; then return 0; else rc=$?; fi
    (( rc == 3 )) || return "$rc"
    timed=$(staged_tree "$2")
    PEER_FOUR_RANK_AUDIT=$1 PEER_TIMED_SOURCE=$timed PEER_AUDIT_REFERENCE=$3 \
        bash "$entry/large_batch_npu4.sh" --four-ranks-candidate-only
}
if [[ $mode == serial-only ]]; then
    if "$py" -B "$entry/support/four_rank_accept.py" serial-summary "$ref"; then exit 0; else rc=$?; fi
    (( rc == 3 )) || exit "$rc"
    audit serial "$serial" "$ref"
    "$py" -B "$entry/support/four_rank_accept.py" serial-summary "$ref"
    exit 0
fi
audit serial "$serial" "$ref"
audit journal "$base" "$ref"
cached=$("$py" -B "$entry/support/four_rank_accept.py" timed "$ref" "$native")
if [[ -n $cached ]]; then
    candidate=$cached
    echo "REUSED_TIMED_CANDIDATE=$candidate"
else
    PEER_FOUR_RANK_VALIDATED_RESUME=$("$py" -B "$entry/support/four_rank_accept.py" partial "$ref" "$native")
    export PEER_FOUR_RANK_VALIDATED_RESUME
    bash "$entry/four_rank_followup_npu4.sh" "$repo" "$reader" "$native" "$ref" all | tee "$stamp/candidate.log"
    candidate=$(sed -n 's/^RESULT_ROOT=//p' "$stamp/candidate.log")
fi
[[ -s $candidate/followup_summary.json && $(<"$candidate/native.txt") == "$native" ]]
audit journal "$native" "$candidate"
if [[ $extra == natural ]]; then
    "$py" -B "$entry/support/four_rank_accept.py" report "$candidate" "$ref" | tee "$stamp/acceptance.log"
    exit 0
fi
# A natural ordering difference is visible evidence and does not prevent the controlled comparison.
rc=0
"$py" -B "$entry/support/four_rank_accept.py" report "$candidate" "$ref" > "$stamp/acceptance.log" || rc=$?
cat "$stamp/acceptance.log"
[[ $rc == 0 || $rc == 3 ]] || exit "$rc"
audit frozen "$base" "$ref"
audit frozen "$native" "$candidate"
"$py" -B "$entry/support/four_rank_accept.py" frozen-report "$candidate" "$ref"
