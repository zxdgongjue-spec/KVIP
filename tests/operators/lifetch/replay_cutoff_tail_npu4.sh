#!/usr/bin/env bash
# Bootstrap from the fetched reader in the existing B24 repository.
set -euo pipefail
repo=${1:?expected KVIP repository}
reader=${2:?expected full reader revision}
mode=${3:-cutoff-tail}
case "$mode" in
    cutoff-tail)
        native=3f24b6ff66153da3605f0f6f24dacf69ce4b66c4
        groups=(17 4 10 7) ;;
    early-index-handoff)
        native=${4:?expected full candidate revision}
        groups=(10 4 17 7) ;;
    owner-topk)
        native=${4:?expected full candidate revision}
        groups=(4 10 17 7 2)
        if (( $# > 4 )); then groups=("${@:5}"); fi ;;
    fill-g10-g2)
        native=${4:?expected full candidate revision}
        [[ $# == 4 ]] || { echo 'fill-g10-g2 takes only the current candidate revision' >&2; exit 2; }
        groups=(2) ;;
    score-equivalence)
        native=1040d7133028ff4cd36904bef0393783be884877
        groups=(10 4 17 7)
        if (( $# > 3 )); then groups=("${@:4}"); fi ;;
    *) echo "Unknown replay mode: $mode" >&2; exit 2 ;;
esac
work=${PEER_G2_WORK:-/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold}
ref="$work/peer-baseline/fused.m3IizR"
if [[ $mode == score-equivalence ]]; then ref="$work/peer-baseline/fused.0SOIpe"; fi
if [[ $mode == owner-topk || $mode == fill-g10-g2 ]]; then ref="$work/peer-baseline/fused.fmn5ww"; fi
runner="$work/fused-scan-sources/reader-$reader"
git -C "$repo" cat-file -e "$native^{commit}"
git -C "$repo" cat-file -e "$reader^{commit}"
test -r "$ref/logical-ties.txt"
reader_ready() {
    local path=$1 head status
    if [[ ! -e "$path/.git" ]]; then
        printf 'READER_PATH=%s missing .git at the worktree root\n' "$path" >&2
        return 1
    fi
    head=$(git -C "$path" rev-parse --verify HEAD) || return 1
    status=$(git --no-optional-locks -C "$path" status --porcelain=v1 --untracked-files=normal) || return 1
    if [[ $head != "$reader" || -n $status ]]; then
        printf 'READER_PATH=%s expected=%s actual=%s\n' "$path" "$reader" "$head" >&2
        if [[ -n $status ]]; then sed -n '1,12p' <<< "$status" >&2; fi
        return 1
    fi
}
prepare_reader() {
    local base=$runner
    mkdir -p "$(dirname "$base")" || return 1
    if [[ ! -e "$base" && ! -L "$base" ]]; then
        git -C "$repo" worktree add -q --detach "$base" "$reader" || echo "Could not create preferred reader at $base" >&2
    fi
    if ! reader_ready "$base"; then
        # Keep the original checkout and its Git registration intact.
        runner=$(mktemp -d "$base.XXXXXX") || return 1
        printf 'READER_RETRY=%s\n' "$runner"
        git -C "$repo" worktree add -q --detach "$runner" "$reader" || return 1
        reader_ready "$runner" || return 1
    fi
    printf 'READER_READY=%s commit=%s\n' "$runner" "$reader"
}
export PEER_G2_WORK="$work" PEER_PHYSICAL_NPU=4 ASCEND_RT_VISIBLE_DEVICES=4 PEER_LOCAL_ONLY=1
export PATH="/usr/local/python3.12.13/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset PEER_RANKED_INPUT PEER_RANKED_HASH_FILE KVIP_Q4_NATIVE_LI_CACHE
log=$(mktemp "$work/host24-npu4-$mode.XXXXXX")
echo "LOG=$log"
prepare_rc=0
prepare_reader >"$log" 2>&1 || prepare_rc=$?
cat "$log"
if (( prepare_rc )); then exit "$prepare_rc"; fi
if [[ $mode == score-equivalence || $mode == owner-topk || $mode == fill-g10-g2 ]]; then
    run_selected() {
        if [[ $mode == fill-g10-g2 ]]; then
            local current="$work/peer-baseline/fused.N7n8ZP"
            local serial_rc=0
            [[ $(<"$current/logical-ties.txt") == "$native" ]] || { echo "Expected current owner TopK result at $current" >&2; return 1; }
            bash "$runner/tests/operators/lifetch/peer_fused.sh" --serial-g10-only "$current" || serial_rc=$?
            if (( serial_rc )); then printf 'G10 FAILED rc=%s log=%s\n' "$serial_rc" "$log" >&2; fi
            bash "$runner/tests/operators/lifetch/peer_fused.sh" --owner-topk "$native" --reference "$ref" 2 || return $?
            return "$serial_rc"
        else
            bash "$runner/tests/operators/lifetch/peer_fused.sh" "--$mode" "$native" --reference "$ref" "${groups[@]}"
        fi
    }
    rc=0
    run_selected 2>&1 |
        tee -a "$log" | while IFS= read -r line; do
            case "$line" in RESULT_ROOT=*|CASE_BEGIN\ *|G[0-9]*\ *|REPLAY_COMPLETE*) printf '%s\n' "$line" ;; esac
        done || rc=$?
    if (( rc != 0 )); then tail -n 30 "$log" >&2; fi
    exit "$rc"
fi
bash "$runner/tests/operators/lifetch/peer_fused.sh" "--$mode" "$native" --reference "$ref" "${groups[@]}" 2>&1 | tee -a "$log"
