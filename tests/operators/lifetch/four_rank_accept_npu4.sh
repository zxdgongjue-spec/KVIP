#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Call through git show <exact-reader>:<this path> | bash -s -- REPO READER CANDIDATE
set -euo pipefail
repo=$(cd "${1:?expected KVIP checkout}" && pwd -P)
reader=$(git -C "$repo" rev-parse --verify "${2:?expected reader commit}^{commit}")
mode=${3:?expected candidate commit or serial-only}
extra=${4:-natural}
case $extra in
    full) extra=isolated-all ;;
    tail) extra=isolated-scheduling ;;
    bs20) extra=isolated-schedule-focus ;;
esac
case $extra in windows-focus|windows-all|windows-g7-x4) [[ $mode == window-study ]] || exit 2 ;; natural) ;; paired-frozen|isolated-li|isolated-tail|isolated-all|isolated-scheduling|isolated-schedule-only|isolated-schedule-focus) [[ $mode != serial-only ]] || exit 2 ;; *) exit 2 ;; esac
if [[ $extra == windows-g7-x4 ]]; then
    # Use exactly the worker that produced the BS32 archive, including its cache key.
    reader=$(git -C "$repo" rev-parse --verify 'e602ff4a688a733664aaabe84aa30732b9e68062^{commit}')
fi
native=
if [[ $mode != serial-only && $mode != window-study ]]; then native=$(git -C "$repo" rev-parse --verify "$mode^{commit}"); fi
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
    if [[ $1 == team-schedule || $1 == window-x* ]]; then
        local checker="$entry/support/b24_team_schedule.py"
        if [[ $1 == window-x* ]]; then checker="$entry/support/b24_window_report.py"; fi
        local group batch complete=1 hit lookup="$3"
        if [[ $1 == window-x* ]]; then lookup=$reader; fi
        for batch in $FIXED_Q4_BATCHES; do
            for group in $FIXED_Q4_GROUPS; do
                hit=$("$py" -B "$checker" cached "$lookup" "$3" "$group" "$batch" "$1")
                if [[ -n $hit ]]; then
                    "$py" -B "$checker" show "$hit" "$3" "$group" "$batch" "$1"
                else
                    complete=0
                fi
            done
        done
        if (( complete )); then return 0; fi
    elif [[ $1 != two-row-* ]]; then
        local -a batches
        read -ra batches <<< "$FIXED_Q4_BATCHES"
        if "$py" -B "$entry/support/four_rank_accept.py" coverage "$3" "$1" --batches "${batches[@]}"; then return 0; else rc=$?; fi
        (( rc == 3 )) || return "$rc"
    fi
    timed=$(staged_tree "$2")
    PEER_FOUR_RANK_AUDIT=$1 PEER_TIMED_SOURCE=$timed PEER_AUDIT_REFERENCE=$3 \
        bash "$entry/large_batch_npu4.sh" --four-ranks-candidate-only
}
if [[ $mode == window-study ]]; then
    [[ $extra == windows-focus || $extra == windows-all ]]
    if [[ -s /etc/machine-id && -s $ref/machine_id.txt ]]; then
        cmp -s /etc/machine-id "$ref/machine_id.txt"
    else
        cmp -s /proc/sys/kernel/random/boot_id "$ref/boot_id.txt"
    fi
    exec > >(tee "$stamp/windows.log") 2>&1
    echo "WINDOW_LOG=$stamp/windows.log"
    if [[ $extra == windows-g7-x4 ]]; then
        # Keep the existing BS32 timing offline and test only a missing shape.
        for batch in 32 28; do
            FIXED_Q4_GROUPS=7 FIXED_Q4_BATCHES=$batch audit window-x4-early "$base" "$ref"
        done
        exit 0
    elif [[ $extra == windows-focus ]]; then
        # The low-miss shortfall and the busy link give opposite predictions.
        FIXED_Q4_GROUPS=2 FIXED_Q4_BATCHES="28 32" audit window-x2-early "$base" "$ref"
        FIXED_Q4_GROUPS="10 7" FIXED_Q4_BATCHES=32 audit window-x2-early "$base" "$ref"
        FIXED_Q4_GROUPS=7 FIXED_Q4_BATCHES=32 audit window-x4-early "$base" "$ref"
    else
        for study in window-x2-early window-x4-early window-x3-linear window-x8-early window-x2-linear window-x4-linear; do
            audit "$study" "$base" "$ref"
        done
    fi
    "$py" -B "$entry/support/b24_window_report.py" summary "$reader" "$ref"
    exit 0
fi
if [[ $mode == serial-only ]]; then
    if "$py" -B "$entry/support/four_rank_accept.py" serial-summary "$ref"; then exit 0; else rc=$?; fi
    (( rc == 3 )) || exit "$rc"
    audit serial "$serial" "$ref"
    "$py" -B "$entry/support/four_rank_accept.py" serial-summary "$ref"
    exit 0
fi
# The independent tail experiment keeps the archived LI. With sixteen teams,
# both rows exist for every team only at BS32.
tail_only() (
    export FIXED_Q4_BATCHES=32
    audit two-row-natural "$base" "$ref"
    audit journal "$base" "$ref"
    audit frozen "$base" "$ref"
    audit two-row-frozen "$base" "$ref"
)
if [[ $extra == isolated-* ]]; then
    if [[ -s /etc/machine-id && -s $ref/machine_id.txt ]]; then
        cmp -s /etc/machine-id "$ref/machine_id.txt" || { echo 'Archive machine ID mismatch' >&2; exit 2; }
    else
        cmp -s /proc/sys/kernel/random/boot_id "$ref/boot_id.txt" || { echo 'Archive boot ID mismatch' >&2; exit 2; }
    fi
    exec > >(tee "$stamp/verify.log") 2>&1
    echo "VERIFY_LOG=$stamp/verify.log"
    if [[ $extra == isolated-schedule-focus ]]; then
        FIXED_Q4_GROUPS=7 FIXED_Q4_BATCHES=20 audit team-schedule "$base" "$ref"
        exit 0
    fi
    if [[ $extra == isolated-schedul* || $extra == isolated-all ]]; then
        FIXED_Q4_BATCHES="20 24 28 32" audit team-schedule "$base" "$ref"
        if [[ $extra == isolated-schedule-only ]]; then exit 0; fi
    fi
    if [[ $extra == isolated-tail || $extra == isolated-scheduling ]]; then tail_only; exit 0; fi
    # An independently started natural run uses this lock. Let it finish
    # before checking completed cells, so we reuse its output.
    lock="$sources/largebs-$(</proc/sys/kernel/random/boot_id)-npu4.lock"
    ( flock 9 ) 9>"$lock"
else
    audit serial "$serial" "$ref"
    audit journal "$base" "$ref"
fi
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
if [[ $extra == isolated-* ]]; then
    "$py" -B "$entry/support/four_rank_accept.py" natural-summary "$candidate" "$ref"
    if [[ $extra == isolated-all ]]; then tail_only; fi
    audit journal "$base" "$ref"
    audit frozen "$base" "$ref"
    audit frozen "$native" "$candidate"
    "$py" -B "$entry/support/four_rank_accept.py" frozen-report "$candidate" "$ref"
    exit 0
fi
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
