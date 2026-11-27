# SPDX-License-Identifier: Apache-2.0
# Sourced after the selected card lock, SDK and the ordinary local-URMA worker are ready.
ref=$PEER_AUDIT_REFERENCE
report="$controller_root/tests/operators/lifetch/support/four_rank_accept.py"
for group in "${groups[@]}"; do
    trace=
    for batch in "${batches[@]}"; do
        old=$("$PYTHON" -B "$report" cached "$out" "$ref" "$group" "$batch" "$audit_mode")
        if [[ -n $old ]]; then
            echo "REUSE G$group BS$batch $audit_mode $old" >&3
            continue
        fi
        if [[ -z $trace ]]; then
            prepared_bank="$ref/g$group/donors"
            if [[ ! -s $prepared_bank/source.json ]]; then prepared_bank="$PEER_AUDIT_ORIGIN/g$group/donors"; fi
            if [[ -s $prepared_bank/source.json ]]; then
                trace=$("$PYTHON" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["trace"])' "$prepared_bank/source.json")
            else
                trace=$(bash "$helpers/peer_g2_input.sh" "$work" "$group")
                mkdir -p "$out/g$group"
                prepared_bank="$out/g$group/donors"
                "$PYTHON" -B "$controller_root/tests/operators/lifetch/support/large_batch_input.py" "$trace" "$group" "$prepared_bank"
            fi
        fi
        pair="$out/g$group/bs$batch/block1"
        mkdir -p "$pair"
        thp="$ref/g$group/bs$batch/block1/fused/thp.json"
        placement=$("$PYTHON" -B "$report" placement "$thp")
        read -r node adapter <<<"$placement"
        set_cpu_placement "$node" "$adapter" archived
        echo "COLLECT G$group BS$batch $audit_mode" >&3
        run_arm "$group" "$batch" "$trace" "$audit_mode" "$pair"
        "$PYTHON" -B "$controller_root/tests/operators/lifetch/support/large_batch_placement.py" compare "$thp" "$pair/$audit_mode/thp.json"
        touch "$pair/$audit_mode/placement_verified"
        "$PYTHON" -B "$report" check "$out" "$ref" "$group" "$batch" "$audit_mode"
    done
done
echo "AUDIT_COMPLETE mode=$audit_mode RESULT_ROOT=$out" >&3
