#!/usr/bin/env bash
# Called by peer_fused.sh after preparing the URMA runtime and its build cache.

ranked_reference_case() {
    "$PYTHON" -B - "$help_reference" "$1" <<'PY'
import json
import sys
from pathlib import Path

root, group = Path(sys.argv[1]), int(sys.argv[2])
case = root / f'g{group}'
for label in ('baseline', 'candidate'):
    arm = case / label
    result, = (arm / 'results').glob('**/fused/path_result.json')
    data = json.loads(result.read_text())
    config = json.loads((result.parents[1] / 'config.json').read_text())
    thp = json.loads((arm / 'thp.json').read_text())
    if (data['correctness'] != 'PASS' or data['failures']
            or data['production_revision'] != (root / f'{label}.txt').read_text().strip()
            or config['group'] != group or (config['warmup'], config['iterations']) != (4, 20)):
        raise ValueError(f'Incomplete or incompatible historical {arm}')
    if label == 'candidate':
        nodes = thp['numa_pages_overlapping_vmas']
        node = next(iter(nodes)) if len(nodes) == 1 else ''
        print(node if node.isdecimal() and nodes[node] > 0 else '')
        print(config['trace_root'])
PY
}

ranked_arm_complete() {
    local group=$1 label=$2 revision=$3 source_name=ranked-${3:0:8}
    local case_dir="$out/g$group"
    [[ -s "$case_dir/$label/thp.json" ]] || return 1
    PYTHONPATH="$tools/support${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -B - \
        "$case_dir" "$label" "$revision" "$group" "$out/$source_name.binary.sha256" <<'PY'
import sys
from pathlib import Path
from ranked_help_report import load

case, label, revision, group, binary = sys.argv[1:]
try:
    arm = load(Path(case).parent, int(group), label, revision, 'fixed' if label == 'anchor' else 'flexible')
    cfg = arm['config']
    if (arm['binary'] != Path(binary).read_text().strip() or cfg['group'] != int(group)
            or (cfg['warmup'], cfg['iterations']) != (4, 20)):
        raise ValueError('current binary, group or timing protocol differs')
except (OSError, ValueError, KeyError, IndexError) as error:
    print(f'INCOMPLETE G{group} {label}: {error}', file=sys.stderr)
    sys.exit(1)
PY
}

ranked_help_groups() {
    local group params preferred trace_path trace hash_file arm revision case_dir saved started
    local -a labels pending summaries=()
    for group in 7 2 10 4 17; do
        case_dir="$out/g$group"
        mkdir -p "$case_dir"
        params=$(ranked_reference_case "$group")
        # The first line can be empty when the original placement was mixed.
        preferred=${params%%$'\n'*}
        trace_path=${params#*$'\n'}
        labels=(candidate)
        if [[ -f "$case_dir/anchor.requested" ]]; then labels=(anchor candidate); fi
        pending=()
        for arm in "${labels[@]}"; do
            revision=$ranked
            if [[ "$arm" == anchor ]]; then revision=$four_field; fi
            if ranked_arm_complete "$group" "$arm" "$revision"; then
                printf 'REUSED G%s %s %s\n' "$group" "$arm" "$case_dir/$arm"
            else
                pending+=("$arm")
            fi
        done
        if (( ${#pending[@]} )); then
            bind_cpu_service "$preferred"
            printf 'CPU_PLACEMENT G%s reference_node=%s policy=%s actual=measured_per_arm\n' \
                "$group" "${preferred:-mixed_or_unknown}" "$policy"
            if [[ ! -d "$trace_path" ]]; then
                echo "The G$group reference input is missing at $trace_path. Select a reference made with an available local capture." >&2
                return 1
            fi
            hash_file=$(mktemp "$case_dir/input.XXXXXX")
            trace=$(PEER_RANKED_INPUT="$trace_path" PEER_RANKED_HASH_FILE="$hash_file" \
                bash "$tools/peer_g2_input.sh" "$work" "$group")
            if [[ -s "$case_dir/input.sha256" ]] && ! cmp -s "$hash_file" "$case_dir/input.sha256"; then
                echo "Changed capture in resumed G$group; previous results require the original input" >&2
                return 1
            fi
            if [[ -s "$help_reference/g$group/input.sha256" ]] && ! cmp -s "$hash_file" "$help_reference/g$group/input.sha256"; then
                echo "Current G$group input hash differs from the reference" >&2
                return 1
            fi
            mv "$hash_file" "$case_dir/input.sha256"
            for arm in "${pending[@]}"; do
                if [[ -e "$case_dir/$arm" ]]; then
                    saved=$(mktemp -d "$case_dir/incomplete-$arm.XXXXXX")
                    mv "$case_dir/$arm" "$saved/$arm"
                    printf 'PRESERVED_INCOMPLETE G%s %s %s\n' "$group" "$arm" "$saved"
                fi
                started=$SECONDS
                run_arm "$arm" "$group" "$case_dir"
                printf 'CASE_COMPLETE G%s %s elapsed_s=%s\n' "$group" "$arm" "$(( SECONDS - started ))"
                revision=$ranked
                if [[ "$arm" == anchor ]]; then revision=$four_field; fi
                ranked_arm_complete "$group" "$arm" "$revision"
            done
        fi
        "$PYTHON" -B "$tools/support/ranked_help_report.py" "$out" \
            --reference "$help_reference" --group "$group" | tee "$case_dir/summary.txt"
        summaries+=("$case_dir/summary.txt")
        cat "${summaries[@]}" >"$out/summary.tmp"
        mv "$out/summary.tmp" "$out/summary.txt"
    done
    printf 'REPLAY_COMPLETE RESULT_ROOT=%s\n' "$out"
}
