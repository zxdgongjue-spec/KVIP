# SPDX-License-Identifier: Apache-2.0
"""Side-by-side feature deltas from two production graph captures in opposite order."""
import json
from pathlib import Path
from statistics import mean
import sys

from reporting import atomic_json
from pr30_archive_compare import marker

GROUPS = (10, 4, 17, 7, 2)
STAGES = ('span', 'LI', 'tail', 'kernel_gaps')
SCALARS = ('window10_per_call_us', 'event_mean_us', 'between_calls_per_call_us', 'event_edge_residual_per_call_us')


def summarize(root):
    groups = {}
    modes = [marker(root / f'block{n}', 'live_main') == '1' for n in (1, 2)]
    if modes[0] != modes[1]:
        raise ValueError('Both blocks must use the same mainline measurement mode')
    live_main = modes[0]
    fresh = ('mainline', 'dualcut', 'paced') if live_main else ('dualcut', 'paced')
    calls_per_graph = {'mainline': 10 if live_main else 20, 'dualcut': 10, 'paced': 10}
    comparisons = {
        'A_vs_main': ('mainline', 'dualcut', 'main_tail_equals_dualcut'),
        'B_vs_A': ('dualcut', 'paced', 'dualcut_li_equals_paced'),
    }
    if live_main:
        comparisons['B_vs_main'] = ('mainline', 'paced', None)
    queries_per_block = calls_per_graph['paced'] * 16 * 4
    for group in GROUPS:
        records = []
        orders = []
        for block in (1, 2):
            directory = root / f'block{block}' / f'g{group}'
            records.append(json.loads((directory / 'feature_steps.json').read_text()))
            orders.append((directory / 'order.txt').read_text().split())
        if (any(row['group'] != group for row in records) or sorted(orders[0]) != sorted(fresh)
                or orders[1] != orders[0][::-1]):
            raise ValueError(f'G{group}: expected two complete comparisons run in reverse order')
        if any(row.get('live_main', False) != live_main for row in records):
            raise ValueError(f'G{group}: block data differs from the declared mainline measurement mode')
        if live_main:
            if any(row['reference'] is not None or row['order'] != order or set(row['measured']) != set(fresh)
                   for row, order in zip(records, orders)):
                raise ValueError(f'G{group}: expected three fresh arms with measured receipts in the declared order')
            if any(records[0][k] != records[1][k] for k in ('native', 'runtime', 'executable_sha256')):
                raise ValueError(f'G{group}: runtime or operator build changed between blocks')
        elif (records[0]['reference'] != records[1]['reference'] or
                records[0]['timing'].get('mainline') != records[1]['timing'].get('mainline')):
            raise ValueError(f'G{group}: the two blocks used different upstream baselines')
        expected = {arm: n for arm, n in calls_per_graph.items()
                    if live_main or arm != 'mainline' or records[0]['reference'] is not None}
        if any(row['calls_per_graph'] != expected or set(row['timing']) != set(expected) for row in records):
            detail = '10 measured calls for all three arms' if live_main else '10 measured candidate calls and 20 archived main calls'
            raise ValueError(f'G{group}: expected {detail}')
        split = [row['measured']['paced']['split_q'] for row in records]
        groups[group] = {'reference': records[0]['reference']}
        for label, (before, after, check) in comparisons.items():
            if before not in expected:
                groups[group][label] = None
                print(f'G{group} {label} UNAVAILABLE reason=no_matching_local_main_archive')
                continue
            saved = {stage: [row['timing'][before][stage][0] - row['timing'][after][stage][0] for row in records]
                     for stage in STAGES}
            saved.update({key: [row['timing'][before][key] - row['timing'][after][key] for row in records]
                          for key in SCALARS})
            groups[group][label] = saved
            isolation = ''
            if check:
                matched = [row['device_text'][check] for row in records]
                groups[group].setdefault('unmodified_stage_match', {})[label] = matched
                isolation = ' unmodified_stage_binary=' + '/'.join('MATCH' if ok else 'DIFF' for ok in matched)
            print(f'G{group} {label} mean_saved_us_block1/block2 '
                  + ' '.join(f'{stage}={a:+.2f}/{b:+.2f}' for stage, (a, b) in saved.items())
                  + isolation
                  + (f' paced_split_Q_per_{queries_per_block}={split[0]}/{split[1]}' if label == 'B_vs_A' else ''))
        groups[group][f'paced_split_Q_per_{queries_per_block}'] = split
    equal_weight = {}
    for label, (_, _, check) in comparisons.items():
        missing = [str(g) for g in GROUPS if groups[g][label] is None]
        if missing:
            equal_weight[label] = None
            print(f'FIVE_GROUP_EQUAL_WEIGHT {label} UNAVAILABLE missing_groups={",".join(missing)}')
            continue
        avg = {stage: [mean(groups[g][label][stage][i] for g in GROUPS) for i in range(2)]
               for stage in (*STAGES, *SCALARS)}
        equal_weight[label] = avg
        isolation = ''
        if check:
            matched = all(all(groups[g]['unmodified_stage_match'][label]) for g in GROUPS)
            isolation = f' unmodified_stage_binary={"MATCH" if matched else "DIFF"}'
        print(f'FIVE_GROUP_EQUAL_WEIGHT {label} mean_saved_us_block1/block2 '
              + ' '.join(f'{stage}={a:+.2f}/{b:+.2f}' for stage, (a, b) in avg.items())
              + isolation)
    atomic_json(root / 'feature_comparison.json',
                dict(groups=groups, equal_weight=equal_weight, live_main=live_main, calls_per_graph=calls_per_graph))
    print(f'FEATURE_STEPS_COMPLETE groups=5 blocks=2 fresh_arms={len(fresh)} measured_calls_per_arm_per_group=20')


if __name__ == '__main__':
    summarize(Path(sys.argv[1]))
