#!/usr/bin/env bash
# Run both fusion stacks serially on the same NPU. Reuse a successful arm on retry.
set -euo pipefail
repo=${1:?supply the local KVIP repository}
reader=${2:?supply the full reader revision}
run=${3:?supply the directory for this comparison}
export PEER_G2_WORK=${PEER_G2_WORK:-/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold}
work=$PEER_G2_WORK
native=20915a78b682378ea60889afbcd14b9a21123913
main=d6cfd392f722dd3f96ef6e05175f503aac511dbb
export PEER_PHYSICAL_NPU=4 ASCEND_RT_VISIBLE_DEVICES=4 PEER_LOCAL_ONLY=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset PEER_AVAILABILITY_ONLY PEER_BORROW_AUDIT PEER_RANKED_INPUT PEER_RANKED_HASH_FILE
entry=tests/operators/lifetch
mkdir -p "$run"
run=$(cd "$run" && pwd)
for rev in "$reader" "$native" "$main"; do git -C "$repo" cat-file -e "${rev}^{commit}"; done
if [[ ! -d "$run/reader" ]]; then git -C "$repo" worktree add -q --detach "$run/reader" "$reader"; fi
if [[ $(git -C "$run/reader" rev-parse HEAD) != "$reader" || -n $(git -C "$run/reader" status --porcelain) ]]; then
    echo "Unexpected reader worktree at $run/reader" >&2
    exit 1
fi
scripts="$run/reader/$entry"
printf 'PAIR_DIRECTORY=%s\n' "$run"
printf 'RETRY_COMMAND=PEER_G2_WORK=%q bash %q %q %q %q\n' "$work" "$scripts/replay_combined_vs_live_main.sh" "$repo" "$reader" "$run"
measure_arm() {
    local name=$1 script=$2 rev=$3 log="$run/$1.log" result
    if [[ -s "$run/$name.root" ]]; then
        echo "REUSE_COMPLETED_ARM=$name RESULT_ROOT=$(<"$run/$name.root")"
        return 0
    fi
    echo "MEASURING=$name REVISION=$rev LOG=$log"
    bash "$scripts/$script" "$repo" "$reader" "$rev" 2>&1 | tee "$log" || return $?
    result=$(sed -n 's/^REPLAY_COMPLETE RESULT_ROOT=//p' "$log")
    [[ -n "$result" && -d "$result" ]] || { echo "Missing completion in $log" >&2; return 1; }
    printf '%s\n' "$result" > "$run/$name.root.tmp"
    mv "$run/$name.root.tmp" "$run/$name.root"
}
rc=0
for arm in candidate main; do
    if [[ $arm == candidate ]]; then script=main_five_npu4.sh; rev=$native
    else script=unified_main_npu4.sh; rev=$main; fi
    status=0
    measure_arm "$arm" "$script" "$rev" || status=$?
    case $status in 0) ;; 130|143) exit "$status" ;; *) rc=$status; echo "FAILED_MEASUREMENT=$arm rc=$status" >&2 ;; esac
done
(( rc == 0 )) || exit "$rc"
py="$work/peer-baseline/python-da318bdb/bin/python"
"$py" -B "$scripts/support/live_main_comparison.py" "$(<"$run/candidate.root")" "$(<"$run/main.root")" \
    "$reader" "$run/comparison.json" | tee "$run/summary.txt"
echo "PAIR_COMPLETE SUMMARY=$run/summary.txt"
