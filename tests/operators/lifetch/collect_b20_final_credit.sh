#!/usr/bin/env bash
# Run each independent collection once. An optional group argument retries only that group.
set -euo pipefail
repo=${1:?expected KVIP repository}
reader=${2:?expected full reader revision}
shift 2
groups=("$@")
if (( ${#groups[@]} == 0 )); then groups=(7 10); fi
for g in "${groups[@]}"; do case "$g" in 7|10) ;; *) exit 2 ;; esac; done
export PEER_PHYSICAL_NPU=5 ASCEND_RT_VISIBLE_DEVICES=5
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PATH="/usr/local/python3.12.13/bin:$PATH"
export PEER_G2_WORK=${PEER_G2_WORK:-/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold}
unset PEER_RANKED_INPUT PEER_RANKED_HASH_FILE
mkdir -p "$PEER_G2_WORK"
out=$(mktemp -d "$PEER_G2_WORK/host20-npu5-credit-data.XXXXXX")
echo "COLLECTION_LOG=$out/collection.log"
exec > >(tee "$out/collection.log") 2>&1
tmp=$(mktemp -d "$out/reader.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
git -C "$repo" archive "$reader" tests/operators/lifetch | tar -xf - -C "$tmp"
lifetch="$tmp/tests/operators/lifetch"
run_group() (
    set -euo pipefail
    if [[ $1 == 7 ]]; then
        native=957f59ace1520d3da8ef839a4d341902f62f8f66
        mode=g7-credit
    else
        native=6d7f257b3d664a8b4991cfd3ebd12da4f2038d04
        mode=g10-four-batch
        if ! python3 -B "$lifetch/support/query_owner_input.py" "$PEER_G2_WORK" 10 >"$out/g10-local-input.log" 2>&1; then
            PEER_LOCAL_ONLY=0 bash "$lifetch/peer_g2_input.sh" "$PEER_G2_WORK" 10
        fi
    fi
    git -C "$repo" cat-file -e "$native^{commit}"
    bash "$lifetch/main_five_npu4.sh" "$repo" "$reader" "$native" "$mode"
)
failed=()
for g in "${groups[@]}"; do
    set +e
    run_group "$g"
    rc=$?
    set -e
    case "$rc" in
        0) ;;
        130|143) exit "$rc" ;;
        *) failed+=("G$g"); echo "COLLECTION_FAILED G$g rc=$rc" >&2 ;;
    esac
done
if (( ${#failed[@]} )); then echo "MISSING=${failed[*]}"; exit 1; fi
