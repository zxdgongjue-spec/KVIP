# SPDX-License-Identifier: Apache-2.0
"""Compare the released PR30 and PR31 sources under PR30's packed-graph timing protocol."""
import argparse
from pathlib import Path
from statistics import mean, median
import subprocess

import numpy as np

from cap_pair_report import SDK, SDK_TREE, inspect_cap, oracle_for
from main_five_report import case_folder, measured_parallel, read
from peer_delta import pair_placement, single_numa_node
from pr30_archive_compare import marker
from pr30_feature_steps import FRESH_CALLS, GROUPS, feature_times, receipt_summary
from reporting import atomic_json

REVISIONS = {
    'pr30': '4bd5d90eaffe9db71ffc64778f9a0cf81bc2e872',
    'pr31_table': 'cfe2b52f6403d5edd612c314bc7110525a5f5789',
    'pr31': '58155892a4f3b7a3bf9789d6055fa4a0f1e95c07',
}
COMPARISONS = (('pr30', 'pr31_table'), ('pr30', 'pr31'), ('pr31_table', 'pr31'))
MEANS = ('event_per_call', 'LI', 'tail', 'kernel_span', 'kernel_gaps', 'between_calls', 'event_edge')


def identity(root):
    expected = {**REVISIONS, 'kvaf': SDK, 'kvaf_tree': SDK_TREE, 'physical_npu': '4'}
    if any(marker(root, name) != value for name, value in expected.items()):
        raise ValueError(f'{root}: unexpected measured source, KVAF tree or device')
    machine = {name: marker(root, name) for name in ('host', 'boot_id', 'machine_id')}
    if not machine['host'] or not machine['boot_id']:
        raise ValueError(f'{root}: host and kernel boot ID are required')
    return {**expected, **machine}


def collect_arm(root, group, arm, owner, oracle, host):
    folder = case_folder(root, group, arm)
    revision = REVISIONS[arm]
    times, exact, thp, proof, cfg = measured_parallel(
        root, group, arm, revision, owner, *oracle, verify_library=True,
        capture_meta_name='capture_input.json', expected_calls=FRESH_CALLS)
    runtime = proof['runtime']
    if runtime['host'] != host or runtime['visible'] != '4':
        raise ValueError(f'{arm}: measured runtime differs from device or host marker')
    production_root = Path(cfg['production_root']).resolve()
    if (subprocess.check_output(['git', '-C', str(production_root), 'rev-parse', 'HEAD'], text=True).strip() != revision
            or not Path(proof['production_python']).resolve().is_relative_to(production_root / 'python')):
        raise ValueError(f'{arm}: production checkout differs from the recorded PR revision')
    if proof['independent_buffers'] != dict.fromkeys(('kv', 'index', 'winners', 'window'), FRESH_CALLS):
        raise ValueError(f'{arm}: timed calls must use independent mutable device state')
    inputs = read(folder.parent / 'replay_input.json')
    capture = read(folder.parent / 'capture_input.json')
    cap = proof['history_cap']
    geometry = inputs['geometry']
    inspect_cap('cap128k', capture, inputs, cap, oracle[1]['native_reference'], expected_calls=FRESH_CALLS)
    ends = np.asarray(cap['effective_ends'])
    if (cfg['history'] != 131328 or geometry['token_bytes'] != 656 or geometry['resident'] != 16384
            or ends.shape != (16, 4) or np.any(ends <= 0) or np.any(ends > 131072 + 3)
            or not np.all(np.diff(ends, axis=1) == 1)):
        raise ValueError(f'{arm}: expected BS16 Q4 and the original 128K geometry/causal bounds')
    cq = read(folder / 'ablation_receipts.json')
    if cq['policy'] != 'paced' or cq['independent_status_allocations'] != FRESH_CALLS:
        raise ValueError(f'{arm}: missing independent paced CQ evidence')
    traffic = receipt_summary(cq['measured'], True)
    receipt_summary(cq['detail'], True)
    for i, call in enumerate(cq['measured']):
        if (sum(call['prefetch']) != cap['measured_requests']['prefetch'][i]
                or sum(sum(qs) for qs in call['final']) != cap['measured_requests']['final'][i]):
            raise ValueError(f'{arm}: live queue and capacity traffic disagree at call {i}')
    raw = read(folder / 'diagnose/round-1/full_attribution.json')
    t = feature_times(raw, times)
    values = dict(event_per_call=t['event_mean_us'], LI=t['LI'][0], tail=t['tail'][0],
                  kernel_span=t['span'][0], kernel_gaps=t['kernel_gaps'][0],
                  between_calls=t['between_calls_per_call_us'], event_edge=t['event_edge_residual_per_call_us'])
    spans = [s['us'] for s in raw['samples']]
    varying = {'production_root', 'production_revision', 'kvaf_server', 'output', 'trace_root'}
    controls = dict(config={k: v for k, v in cfg.items() if k not in varying}, runtime=runtime,
                    capture=capture, operands=proof['capture_operand_hashes'],
                    cap={k: v for k, v in cap.items() if k != 'measured_requests'}, cpu_bytes=thp['bytes'],
                    oracle=oracle[1]['serial_set_digest'], placement=[single_numa_node(thp), thp['ub_device']])
    result = dict(means=values, event_total_us=t['event_total_us'], spans_us=spans,
                  exact_id_queries=int(exact.sum()), library=proof['library_sha256'], traffic=traffic)
    return result, controls, thp


def format_values(values, signed=False):
    spec = '+.2f' if signed else '.2f'
    return ' '.join(f'{key}_us={value:{spec}}' for key, value in values.items())


def deltas(means):
    return {f'{new}_minus_{old}': {k: means[new][k] - means[old][k] for k in MEANS}
            for old, new in COMPARISONS}


def report(root, group, owner):
    machine = identity(root)
    order = (root / f'g{group}/order.txt').read_text().split()
    if sorted(order) != sorted(REVISIONS):
        raise ValueError(f'G{group}: expected all three current measurements in {root}')
    oracle = oracle_for(owner, group)
    arms = {}
    for name in order:
        arm, controls, thp = collect_arm(root, group, name, owner, oracle, machine['host'])
        if not arms:
            reference, placement = controls, thp
        if controls != reference or pair_placement(placement, thp, local_baseline=True) != 'matched':
            raise ValueError(f'G{group} {name}: input, execution settings, runtime, NUMA or UB device differs')
        arms[name] = arm
    row = dict(group=group, order=order, identity=machine, controls=reference, arms=arms)
    atomic_json(root / f'g{group}/pr_head_parity.json', row)
    for name in order:
        arm = arms[name]
        print(f'G{group} {root.name} CHECK arm={name} TopK_score_vs_strong_serial=PASS queries=640/640 '
              f'exact_id_sets={arm["exact_id_queries"]}/640 KV_and_Index=PASS rows=160/160 '
              'ordered_diagnostic_rows=160/160 private_CQ=PASS placement=matched')
        print(f'G{group} {root.name} {name}', format_values(arm['means']),
              f'kernel_span_p50_us={median(arm["spans_us"]):.2f} event_total_us={arm["event_total_us"]:.2f}',
              'kernel_span_samples_us=' + '/'.join(f'{x:.2f}' for x in arm['spans_us']))
    for name, diff in deltas({name: a['means'] for name, a in arms.items()}).items():
        print(f'G{group} {root.name} {name}', format_values(diff, signed=True))


def summarize_pair(group, first, second):
    if (first['group'] != group or second['group'] != group or first['identity'] != second['identity']
            or first['controls'] != second['controls'] or set(first['order']) != set(REVISIONS)
            or len(first['order']) != len(REVISIONS) or second['order'] != first['order'][::-1]):
        raise ValueError(f'G{group}: reverse-order blocks differ in workload or execution controls')
    means = {}
    spans_p50 = {}
    for arm in REVISIONS:
        a, b = first['arms'][arm], second['arms'][arm]
        if a['library'] != b['library'] or any(len(r['spans_us']) != FRESH_CALLS for r in (a, b)):
            raise ValueError(f'G{group} {arm}: source binary or ten-call protocol changed between blocks')
        means[arm] = {k: mean((a['means'][k], b['means'][k])) for k in MEANS}
        spans_p50[arm] = median(a['spans_us'] + b['spans_us'])
    return dict(means=means, kernel_span_p50_us=spans_p50, deltas=deltas(means))


def summary(root):
    machine = identity(root)
    rows = {}
    for group in GROUPS:
        blocks = [read(root / f'block{block}/g{group}/pr_head_parity.json') for block in (1, 2)]
        if any(block['identity'] != machine for block in blocks):
            raise ValueError(f'G{group}: group was recorded on different builds, machine or boot')
        if rows and blocks[0]['controls']['runtime'] != runtime:
            raise ValueError('Runtime changed between trace groups')
        runtime = blocks[0]['controls']['runtime']
        row = summarize_pair(group, *blocks)
        rows[group] = row
        for arm, values in row['means'].items():
            print(f'G{group} two_block_mean {arm}', format_values(values),
                  f'kernel_span_p50_us={row["kernel_span_p50_us"][arm]:.2f}')
        for name, values in row['deltas'].items():
            print(f'G{group} two_block_mean {name}', format_values(values, signed=True))
    means = {arm: {k: mean(row['means'][arm][k] for row in rows.values()) for k in MEANS} for arm in REVISIONS}
    diffs = deltas(means)
    for arm, values in means.items():
        print(f'FIVE_GROUP_EQUAL_WEIGHT {arm}', format_values(values))
    for name, values in diffs.items():
        print(f'FIVE_GROUP_EQUAL_WEIGHT {name}', format_values(values, signed=True))
    atomic_json(root / 'pr_head_parity.json', dict(identity=machine, groups=rows, means=means, deltas=diffs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('group', type=int, nargs='?', choices=GROUPS)
    parser.add_argument('owner', type=Path, nargs='?')
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    if args.summary and args.group is None and args.owner is None:
        summary(args.root)
    elif not args.summary and args.group is not None and args.owner is not None:
        report(args.root, args.group, args.owner)
    else:
        parser.error('pass ROOT/blockN GROUP OWNER or ROOT --summary')
