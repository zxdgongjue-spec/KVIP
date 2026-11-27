# SPDX-License-Identifier: Apache-2.0
# Runs within large_batch_npu4.sh, which owns the card lock, transport SDK and donor helpers.
for group in "${groups[@]}"; do
    trace=
    for batch in "${batches[@]}"; do
        pair="$out/g$group/bs$batch/block1"
        mkdir -p "$pair"
        if [[ -z $trace ]]; then
            prepared_bank="$ref/g$group/donors"
            if [[ -s $prepared_bank/source.json && ${PEER_STAGGER_LOCAL:-0} != 1 ]]; then
                trace=$("$PYTHON" -B -c 'import json,sys;print(json.load(open(sys.argv[1]))["trace"])' "$prepared_bank/source.json")
            else
                trace=$(bash "$helpers/peer_g2_input.sh" "$work" "$group")
                prepared_bank="$out/g$group/donors"
                "$PYTHON" -B "$controller_root/tests/operators/lifetch/support/large_batch_input.py" "$trace" "$group" "$prepared_bank"
            fi
        fi
        PEER_FROZEN_LIST=
        if [[ ${PEER_STAGGER_LOCAL:-0} != 1 ]]; then
            thp="$ref/g$group/bs$batch/block1/fused/thp.json"
            placement=$("$PYTHON" -B "$report" placement "$thp")
            read -r node adapter <<<"$placement"
            set_cpu_placement "$node" "$adapter" archived
            PEER_FROZEN_LIST=$("$PYTHON" -B "$report" cached "$out" "$ref" "$group" "$batch" journal)
        else
            set_cpu_placement "$planned_node" "$planned_adapter"
        fi
        if [[ -z $PEER_FROZEN_LIST ]]; then
            echo "CAPTURE G$group BS$batch one untimed original journal" >&3
            run_arm "$group" "$batch" "$trace" stagger-journal "$pair"
            PEER_FROZEN_LIST="$pair/stagger-journal"
            if [[ ${PEER_STAGGER_LOCAL:-0} == 1 ]]; then thp="$PEER_FROZEN_LIST/thp.json"; fi
            "$PYTHON" -B "$controller_root/tests/operators/lifetch/support/large_batch_placement.py" compare "$thp" "$PEER_FROZEN_LIST/thp.json"
            touch "$PEER_FROZEN_LIST/placement_verified"
        fi
        export PEER_FROZEN_LIST
        placement=$("$PYTHON" -B "$report" placement "$thp")
        read -r node adapter <<<"$placement"
        set_cpu_placement "$node" "$adapter" archived
        for gate in 0 1; do
            echo "COLLECT G$group BS$batch private_gate=$gate" >&3
            PEER_STAGGER_ENABLE=$gate run_arm "$group" "$batch" "$trace" "gate$gate" "$pair"
            "$PYTHON" -B "$controller_root/tests/operators/lifetch/support/large_batch_placement.py" compare "$thp" "$pair/gate$gate/thp.json"
            touch "$pair/gate$gate/placement_verified"
        done
        "$PYTHON" -B "$controller_root/tests/operators/lifetch/support/terminal_stagger_result.py" "$pair" >&3
    done
done
echo "AUDIT_COMPLETE mode=$audit_mode RESULT_ROOT=$out" >&3
