# SPDX-License-Identifier: Apache-2.0
# The caller owns the card lock, native SDK, and transient KVAF worker.
study="$controller_root/tests/operators/lifetch/support/terminal_stagger_natural.py"
for group in "${groups[@]}"; do
    trace=
    for batch in "${batches[@]}"; do
        pair="$out/g$group/bs$batch/block1"
        mkdir -p "$pair"
        selected=$("$PYTHON" -B "$study" find "$out" "$ref" "$group" "$batch")
        if [[ -n $selected ]]; then
            echo "REUSE G$group BS$batch fixed_GET_pair=$selected" >&3
        fi
        if [[ ${PEER_STAGGER_LOCAL:-0} != 1 ]]; then
            thp="$ref/g$group/bs$batch/block1/fused/thp.json"
            placement=$("$PYTHON" -B "$report" placement "$thp")
            read -r node adapter <<<"$placement"
            set_cpu_placement "$node" "$adapter" archived
        else
            set_cpu_placement "$planned_node" "$planned_adapter"
        fi
        if [[ -z $trace ]]; then
            prepared_bank="$ref/g$group/donors"
            if [[ -n $selected && ! -s $prepared_bank/source.json ]]; then
                prepared_bank="$(dirname "$(dirname "$selected")")/donors"
            fi
            if [[ -s $prepared_bank/source.json ]]; then
                trace=$("$PYTHON" -B -c 'import json,sys;print(json.load(open(sys.argv[1]))["trace"])' "$prepared_bank/source.json")
            else
                trace=$(bash "$helpers/peer_g2_input.sh" "$work" "$group")
                prepared_bank="$out/g$group/donors"
                "$PYTHON" -B "$controller_root/tests/operators/lifetch/support/large_batch_input.py" "$trace" "$group" "$prepared_bank"
            fi
        fi
        if [[ -z $selected ]]; then
            PEER_FROZEN_LIST=
            if [[ ${PEER_STAGGER_LOCAL:-0} != 1 ]]; then
                PEER_FROZEN_LIST=$("$PYTHON" -B "$report" cached "$out" "$ref" "$group" "$batch" journal)
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
                echo "COLLECT G$group BS$batch defer_partner_until_first_CQ=$gate" >&3
                PEER_STAGGER_ENABLE=$gate run_arm "$group" "$batch" "$trace" "gate$gate" "$pair"
                "$PYTHON" -B "$controller_root/tests/operators/lifetch/support/large_batch_placement.py" compare "$thp" "$pair/gate$gate/thp.json"
                touch "$pair/gate$gate/placement_verified"
            done
            selected="$pair"
        fi
        checked=$("$PYTHON" -B "$study" eligible "$out" "$ref" "$group" "$batch" "$selected")
        echo "$checked" >&3
        if [[ ${checked##*$'\n'} != PROMOTE ]]; then
            echo "G$group BS$batch NATURAL_SKIPPED fixed-list correctness or graph/tail gain did not qualify" >&3
            continue
        fi
        unset PEER_FROZEN_LIST PEER_STAGGER_ENABLE
        thp="$selected/gate1/thp.json"
        hit=$("$PYTHON" -B "$study" find-natural "$out" "$ref" "$group" "$batch" "$selected")
        if [[ -n $hit ]]; then
            echo "REUSE G$group BS$batch natural_graph=$hit" >&3
            "$PYTHON" -B "$study" summary "$hit" "$ref" "$group" "$batch" "$selected" >&3
            continue
        fi
        placement=$("$PYTHON" -B "$report" placement "$thp")
        read -r node adapter <<<"$placement"
        set_cpu_placement "$node" "$adapter" archived
        echo "COLLECT G$group BS$batch natural_four_window_graph ordinary_GETs=1 calls=10" >&3
        run_arm "$group" "$batch" "$trace" fused "$pair"
        "$PYTHON" -B "$controller_root/tests/operators/lifetch/support/large_batch_placement.py" compare "$thp" "$pair/fused/thp.json"
        touch "$pair/fused/placement_verified"
        "$PYTHON" -B "$study" summary "$out" "$ref" "$group" "$batch" "$selected" >&3
    done
done
echo "AUDIT_COMPLETE mode=$audit_mode RESULT_ROOT=$out" >&3
