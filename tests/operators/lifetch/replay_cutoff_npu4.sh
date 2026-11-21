#!/usr/bin/env bash
# Run the pinned physical-NPU4 candidate. Optional arguments are logs from earlier attempts.
set -eo pipefail
work=${PEER_G2_WORK:-/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold}
native=24c619b9581aad43fd53c815928def26a1e19560
reader=c5ae5ce43f458ab2657b9e65767739e8e70476bb
runner="$work/fused-scan-sources/reader-$reader"
export PEER_PHYSICAL_NPU=4 ASCEND_RT_VISIBLE_DEVICES=4 PEER_LOCAL_ONLY=1
export PEER_G2_WORK="$work" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PATH="/usr/local/python3.12.13/bin:$PATH"
unset PEER_RANKED_INPUT PEER_RANKED_HASH_FILE KVIP_Q4_NATIVE_LI_CACHE

if [[ "$(git -C "$runner" rev-parse HEAD)" != "$reader" || -n "$(git -C "$runner" status --porcelain)" ]]; then
    echo "RUNNER_MISMATCH=$runner" >&2
    exit 1
fi
git -C "$runner" cat-file -e "$native^{commit}"

show() {
    grep -E '^(RESULT_ROOT=|CONFIG URMA physical_NPU=|CASE_BEGIN |REPLAY_COMPLETE|TOPK_MISMATCH|FAILED_ARM=|G(17|4|10|7) (CHECK |logical-ties |vs_local_81e22933 |vs_serial2_archive |FIRST_DIFFERENCE |EAGER_ARTIFACT |NATIVE_REPEAT_DIFFERENCE |TIE_RULE_DIFFERENCE |(archived_6bca602a|6bca602a_archive) .*LI_saved_mean/P50_us=))' "$1" || :
}

pending=()
for g in 17 4 10 7; do
    for old in "$@"; do
        [[ -r "$old" ]] || continue
        root=$(sed -n 's/^RESULT_ROOT=//p' "$old")
        [[ -n "$root" ]] || continue
        grep -Fqx "$native" "$root/logical-ties.txt" 2>/dev/null || continue
        grep -Fqx 4 "$root/physical_npu.txt" 2>/dev/null || continue
        s="$root/g$g/li-summary.txt"
        [[ -s "$s" ]] || continue
        grep -Fqx "G$g CHECK TopK_set_vs_strong_serial=PASS calls=20/20 queries=1280/1280 serial_repeat_queries=64/64 KV=PASS ordered_Index=PASS rows=320/320" "$s" || continue
        grep -Eq "^G$g logical-ties .*graph_mean/P50_us=.* tail_mean/P50_us=" "$s" || continue
        grep -Eq "^G$g (archived_6bca602a|6bca602a_archive) .*LI_saved_mean/P50_us=" "$s" || continue
        echo "REUSED G$g $s"
        show "$s"
        continue 2
    done
    pending+=("$g")
done

(( ${#pending[@]} )) || exit 0
log=$(mktemp "$work/host24-npu4-cutoff-perf.XXXXXX")
echo "LOG=$log"
echo "RUN_GROUPS=${pending[*]}"
rc=0
bash "$runner/tests/operators/lifetch/peer_fused.sh" --logical-ties "$native" "${pending[@]}" >"$log" 2>&1 || rc=$?
show "$log"
if (( rc != 0 && rc != 3 )); then tail -n 30 "$log" >&2; fi
exit "$rc"
