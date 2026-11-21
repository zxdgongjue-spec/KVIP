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
    early-index-handoff|prefetch-cq-overlap)
        native=${4:?expected full candidate revision}
        groups=(10 4 17 7) ;;
    *) echo "Unknown replay mode: $mode" >&2; exit 2 ;;
esac
work=${PEER_G2_WORK:-/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold}
ref="$work/peer-baseline/fused.m3IizR"
references=(--reference "$ref")
if [[ $mode == prefetch-cq-overlap ]]; then
    references+=(--previous "$work/peer-baseline/fused.0SOIpe")
fi
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
log=$(mktemp "$work/host24-npu4-$mode.XXXXXX")
echo "LOG=$log"
bash "$runner/tests/operators/lifetch/peer_fused.sh" "--$mode" "$native" "${references[@]}" "${groups[@]}" 2>&1 | tee "$log"
