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
unset PEER_RANKED_INPUT PEER_RANKED_HASH_FILE KVIP_CREDIT_G10_INPUT KVIP_CREDIT_G10_HASHES
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
        if python3 -B "$lifetch/support/query_owner_input.py" "$PEER_G2_WORK" 10 >"$out/g10-local-input.log" 2>&1; then
            mapfile -t local_input < "$out/g10-local-input.log"
            if [[ ${local_input[0]} == merged ]]; then
                export KVIP_CREDIT_G10_INPUT="${local_input[1]}"
                saved_hashes="$(dirname "$KVIP_CREDIT_G10_INPUT")/capture_operand_hashes.json"
                if [[ -s "$saved_hashes" ]]; then
                    export KVIP_CREDIT_G10_HASHES="$saved_hashes"
                else
                    host=${PEER_TRACE_SOURCE_HOST:-root@141.62.38.24}
                    receipt=/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold/peer-baseline/main-five.hF6qQ8/g10/mainline/results/mainline/cases/diagnose/fused/mainline_checks.json
                    ssh -T -o ConnectTimeout=20 -- "$host" "cat '$receipt'" |
                        python3 -c 'import json,sys; p=json.load(sys.stdin); assert p["production_revision"]=="609fa26fd715f9409bbe74e1b2d3685c58dfb0cf" and p["runtime"]["visible"]=="4"; json.dump(p["capture_operand_hashes"],sys.stdout)' > "$out/g10-hashes.json"
                    export KVIP_CREDIT_G10_HASHES="$out/g10-hashes.json"
                fi
            fi
        else
            if ! PEER_LOCAL_ONLY=0 bash "$lifetch/peer_g2_input.sh" "$PEER_G2_WORK" 10; then
                # The original B24 run may have read a merged capture or a prepared
                # input outside the usual suite paths. Recover only its recorded inputs.
                host=${PEER_TRACE_SOURCE_HOST:-root@141.62.38.24}
                mkdir "$out/g10-import"
                ssh -T -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -- "$host" \
                    "/usr/local/python3.12.13/bin/python3 - /home/c00982601/CacheOS/ubmem-validation/recent-index-threshold" \
                    < "$lifetch/support/credit_g10_snapshot.py" | tar -xf - -C "$out/g10-import"
                export KVIP_CREDIT_G10_INPUT="$out/g10-import/g10-input"
                export KVIP_CREDIT_G10_HASHES="$out/g10-import/capture_operand_hashes.json"
            fi
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
