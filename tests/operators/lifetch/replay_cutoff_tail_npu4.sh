#!/usr/bin/env bash
# Bootstrap from the fetched reader in the existing B24 repository.
set -euo pipefail
repo=${1:?expected KVIP repository}
reader=${2:?expected full reader revision}
mode=${3:-single}
native=3f24b6ff66153da3605f0f6f24dacf69ce4b66c4
if [[ $mode == g7-prefetch-cq ]]; then
    native=b4f73163e63629e9b8f66cd6221e84dcb9ceccc7
    [[ ${4:-$native} == "$native" ]] || { echo '需要与此回放脚本配套的预取回执诊断算子' >&2; exit 2; }
fi
work=${PEER_G2_WORK:-/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold}
ref="$work/peer-baseline/fused.m3IizR"
runner="$work/fused-scan-sources/reader-$reader"
git -C "$repo" cat-file -e "$native^{commit}"
test -r "$ref/logical-ties.txt"
mkdir -p "$(dirname "$runner")"
if [[ ! -e "$runner" ]]; then
    git -C "$repo" worktree add -q --detach "$runner" "$reader"
fi
if [[ "$(git -C "$runner" rev-parse HEAD)" != "$reader" || -n "$(git -C "$runner" status --porcelain)" ]]; then
    echo "RUNNER_MISMATCH=$runner" >&2
    exit 1
fi
export PEER_G2_WORK="$work" PEER_PHYSICAL_NPU=4 ASCEND_RT_VISIBLE_DEVICES=4 PEER_LOCAL_ONLY=1
export PATH="/usr/local/python3.12.13/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset PEER_RANKED_INPUT PEER_RANKED_HASH_FILE KVIP_Q4_NATIVE_LI_CACHE
tools="$runner/tests/operators/lifetch"
case "$mode" in
    single) args=(--cutoff-tail "$native" --reference "$ref" 17 4 10 7) ;;
    g7-prefetch-cq) args=(--prefetch-cq-proof "$native" --reference "$ref" --measured "$work/peer-baseline/fused.Cmp6BN" 7) ;;
    g7-graph) args=(--cutoff-tail-graph "$native" --reference "$ref" 7) ;;
    *) echo "Unknown capture mode: $mode" >&2; exit 2 ;;
esac
prefix=host24-npu4-cutoff-tail
if [[ $mode == g7-prefetch-cq ]]; then prefix=host24-npu4-g7-prefetch-cq-trace; fi
log=$(mktemp "$work/$prefix.XXXXXX")
echo "LOG=$log"
prior="$work/peer-baseline/fused.WvlohG"
if [[ "$mode" == g7-graph && -f "$prior/owner-tail.tar.gz" ]]; then
    echo "PRIOR_FILE=$prior/owner-tail.tar.gz"
    python3 -B "$tools/support/owner_balance.py" "$prior" | tee "$log"
fi
bash "$tools/peer_fused.sh" "${args[@]}" 2>&1 | tee -a "$log"
if [[ "$mode" == g7-graph && -f "$prior/owner-tail.tar.gz" ]]; then
    result=$(sed -n 's/^RETURN_FILE=//p' "$log")
    [[ -f "$result" && ${result##*/} == owner-graph.tar.gz ]]
    parent=$(dirname "$result")
    tar -cf "$parent/tail-evidence.tar" -C "$prior" owner-tail.tar.gz -C "$parent" owner-graph.tar.gz
    echo "SEND_FILE=$parent/tail-evidence.tar"
fi
