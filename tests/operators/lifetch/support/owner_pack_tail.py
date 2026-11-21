"""Capture the exact-cutoff tail against its archived Level 1 run and native serial token sets."""
import argparse
import json
from statistics import median
from math import isfinite
from pathlib import Path

import numpy as np

from query_owner_report import checked_layout_plan, compare_layout_receipts, load
from li_restore_report import read_sets
from native_li_reference import fused_li_output
from peer_delta import pair_placement, read, single_numa_node
from reporting import atomic_json

REPEATS = 5


def archived_references(root, group, identity=None):
    _, digest, _, _ = reference(root, group)
    path, = (root / f'g{group}/logical-ties/results').glob('*/cases/diagnose/fused/native_li_compare.json')
    comparison = read(path)
    if (comparison['group'], comparison['phase'], comparison['measured_calls'],
            comparison['queries_per_call']) != (group, 'measured', 20, 64):
        raise ValueError('Incomplete archived serial comparison')
    measured_identity = comparison['native_reference']['identity']
    if identity is not None:
        # Boot and package inode changes do not change saved token IDs. Require
        # the actual operands plus the logical-device and runtime configuration.
        for key in ('inputs', 'host', 'npu', 'visible', 'operator', 'sparse_mode',
                    'torch', 'torch_npu', 'cann'):
            if measured_identity[key] != identity[key]:
                raise ValueError(f'Archived LI identity differs on {key}')
    checked = read_sets(path.parent, comparison, 20, [digest] * 20)
    if not checked[4]:
        raise ValueError(f'G{group}: archived strong-serial token sets are unstable or different')
    with np.load(path.parent / comparison['raw_ids'], allow_pickle=False) as raw:
        refs = {name: raw[name] for name in ('strong_serial', 'strong_serial_repeat', 'same_layout_native')}
    metadata = dict(comparison['native_reference'], origin='completed-archive', archive_path=str(path))
    if identity is not None:
        metadata['replay_identity'] = identity
    return refs, metadata


def archived_serial(root, group, identity=None):
    refs, _ = archived_references(root, group, identity)
    return np.sort(refs['strong_serial'], axis=-1)


def check_serial(state, expected):
    # fused_li_output checks that the raw physical-position LI output maps to
    # the same logical token set that the tail returns to the caller.
    actual = np.sort(fused_li_output(state), axis=-1)
    if not np.array_equal(actual, expected):
        row, q = map(int, np.argwhere(np.any(actual != expected, axis=-1))[0])
        missing = np.setdiff1d(expected[row, q], actual[row, q])[:8].tolist()
        extra = np.setdiff1d(actual[row, q], expected[row, q])[:8].tolist()
        raise AssertionError(f'Strong serial TopK set differs at row {row}, Q{q+1}: missing={missing}, extra={extra}')


def workload(state):
    rows = state.rotating.rows.cpu().numpy().view(np.uint32)
    posts = state.rotating.final_posts.cpu().numpy()
    selected = state.rotating.arguments[9].cpu().numpy().view(np.int32).reshape(16, 4, 2048)
    requests = []
    for row in range(16):
        windows = list(map(int, rows[row, 16:20]))
        prefetch, final = int(rows[row, 9]), int(rows[row, 247])
        if sum(windows) != prefetch or max(windows) > 256 or int(posts[row, :, 0].sum()) != final:
            raise ValueError(f'Row {row} window/final counts differ from the completed descriptors')
        queries, seen = [], set()
        for q in range(4):
            ids = set(map(int, selected[row, q]))
            count, safe = int(posts[row, q, 0]), int(posts[row, q, 5])
            if not 0 <= safe <= count:
                raise ValueError(f'Invalid safe prefix for row {row}, Q{q+1}')
            queries.append(dict(query=q+1, final_posted=count, final_safe=safe,
                                cross_q_duplicate=len(ids & seen)))
            seen.update(ids)
        requests.append(dict(request=row, prefetch=prefetch, final=final,
                             window_posted=windows, queries=queries))
    return requests


def capture(state, inputs, config, directory, ticks_per_us, replay_and_time, expected):
    if not isfinite(ticks_per_us) or ticks_per_us <= 0:
        raise ValueError('Invalid owner counter frequency')
    samples = []
    checked = 0
    for repeat in range(REPEATS):
        modes = (False, True) if repeat % 2 == 0 else (True, False)
        sample = dict(repeat=repeat, first='off' if not modes[0] else 'on')
        for enabled in modes:
            state.reset()
            state.forward.set_query_owner_capture(enabled)
            if enabled:
                state.forward.query_owner_trace.zero_()
            elapsed = replay_and_time()
            if not isfinite(elapsed) or elapsed <= 0:
                raise ValueError('Invalid diagnostic graph event interval')
            sample['event_on_us' if enabled else 'event_off_us'] = elapsed
            check_serial(state, expected)
            state.validate(inputs)
            checked += 1
            if not enabled:
                sample['off_layout_receipts'] = state.layout_receipts
            else:
                ticks = state.forward.query_owner_trace.cpu().numpy()
                if ticks.shape != (64, 848) or np.any(ticks[:, 16] <= 0):
                    raise ValueError(f'Missing per-Q native owner stamps, observed shape {ticks.shape}')
                sample.update(ticks=ticks.tolist(), requests=workload(state),
                              topk_set_digest=state.topk_set_digest,
                              serial_set_queries=64, query_owners=state.query_owners,
                              layout_receipts=state.layout_receipts)
        samples.append(sample)
    atomic_json(directory / 'owner_tail.json', dict(
        schema_version=2, group=config.group, warmup=config.warmup, independent_states=1,
        native_revision=config.production_revision, input_paths=list(map(str, inputs.source_paths)),
        source_round=inputs.source_round, ticks_per_us=ticks_per_us, checked_calls=checked, samples=samples))
    return checked


def reference(root, group):
    if (root/'logical-ties.txt').read_text().strip() != '24c619b9581aad43fd53c815928def26a1e19560':
        raise ValueError('Unexpected uninstrumented cutoff revision')
    summary = load(root, group, 'logical-ties', row_dependencies=True)
    compare_layout_receipts({'logical-ties': summary[11]}, 'logical-ties')
    if len(set(summary[1])) != 1:
        raise ValueError('Archived TopK changes between invocations')
    return summary[0], summary[1][0], summary[11], summary[2]


def match_plans(samples, archive):
    if len(samples) != REPEATS or len(archive) != 20:
        raise ValueError('Expected five paired diagnostics and twenty archived invocations')
    states = {}
    archived_plans = {checked_layout_plan(rows, states, 'archive') for rows in archive}
    archived_rows = {entry for call in archived_plans for entry in enumerate(call)}
    archive_rows = archive_calls = archive_off_rows = archive_off_calls = on_off_rows = 0
    matched_deltas = []
    for sample in samples:
        on = checked_layout_plan(sample['layout_receipts'], states, 'trace-on')
        off = checked_layout_plan(sample['off_layout_receipts'], states, 'trace-off')
        matched_on = sum(entry in archived_rows for entry in enumerate(on))
        matched_off = sum(entry in archived_rows for entry in enumerate(off))
        paired_rows = sum(a == b for a, b in zip(on, off))
        sample['plan_match'] = dict(archive_on_rows=matched_on, archive_off_rows=matched_off,
                                    archive_on_call=on in archived_plans, archive_off_call=off in archived_plans,
                                    on_off_rows=paired_rows, on_off_call=on == off)
        archive_rows += matched_on
        archive_off_rows += matched_off
        archive_calls += int(on in archived_plans)
        archive_off_calls += int(off in archived_plans)
        on_off_rows += paired_rows
        if on == off:
            matched_deltas.append(sample['event_on_us'] - sample['event_off_us'])
    return dict(archive_rows=archive_rows, archive_calls=archive_calls,
                archive_off_rows=archive_off_rows, archive_off_calls=archive_off_calls, on_off_rows=on_off_rows,
                on_off_calls=len(matched_deltas),
                matched_on_minus_off_p50_us=median(matched_deltas) if matched_deltas else None)


def verify(root, group):
    base = Path((root / 'owner-tail-reference.txt').read_text().strip())
    inputs, digest, archive, old_thp = reference(base, group)
    directory = root / f'g{group}'
    paths = list((directory / 'logical-ties/results').glob('*/cases/diagnose/fused/owner_tail.json'))
    if len(paths) != 1:
        raise ValueError(f'G{group}: expected one completed owner-tail trace, found {len(paths)}')
    path = paths[0]
    data = json.loads(path.read_text())
    result = read(path.parent / 'path_result.json')
    config = read(path.parent.parent / 'config.json')
    expected_config = dict(group=group, query_owner_arm='owner-pack', mode='diagnose', level=1, device=0,
                           warmup=4, iterations=1, ranked_q_workers=2, intermediate_fetches=4,
                           prefetch_tokens=1024, prefetch_rank_start=1024, window_limit=256, fused_only=True)
    if (any(config.get(key) != value for key, value in expected_config.items())
            or config.get('route_policy') or config.get('ranked_auto_route')
            or result['transport'] != 'urma' or result['runtime_workers'] != 40):
        raise ValueError(f'G{group}: diagnostic configuration differs from the expected query-owner route')
    if (result['correctness'] != 'PASS' or result['failures'] or data['checked_calls'] != 2 * REPEATS
            or result['production_revision'] != data['native_revision']
            or result['binary_sha256'] != (root / f'ranked-{data["native_revision"][:8]}.binary.sha256').read_text().strip()):
        raise ValueError(f'G{group}: incomplete diagnostic correctness checks')
    actual_inputs = json.loads((path.parent.parent / 'replay_input.json').read_text())
    if inputs != actual_inputs or data['input_paths'] != actual_inputs['source_paths']:
        raise ValueError(f'G{group}: diagnostic input differs from the archived exact-cutoff run')
    samples = data['samples']
    if (data['group'] != group or data['native_revision'] != (root/'logical-ties.txt').read_text().strip()
            or len(samples) != REPEATS or [s['repeat'] for s in samples] != list(range(REPEATS))
            or any(s['topk_set_digest'] != digest or s['serial_set_queries'] != 64 for s in samples)):
        raise ValueError(f'G{group}: diagnostic TopK or input count differs from the reference')
    data['plan_match'] = match_plans(samples, archive)
    data['plan_match']['placement'] = pair_placement(old_thp, read(directory/'logical-ties/thp.json'),
                                                      local_baseline=True)
    atomic_json(path, data)
    (directory / path.name).symlink_to(path.relative_to(directory))
    print(f'G{group} CHECK TopK_set_vs_archived_strong_serial=PASS calls={2*REPEATS}/{2*REPEATS} '
          f'queries={128*REPEATS}/{128*REPEATS} KV=PASS ordered_Index=PASS rows={32*REPEATS}/{32*REPEATS} '
          f'placement={data["plan_match"]["placement"]} trace={directory / path.name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('numa', 'verify'))
    parser.add_argument('root', type=Path)
    parser.add_argument('group', type=int)
    args = parser.parse_args()
    if args.command == 'numa':
        thp = read(args.root / f'g{args.group}/logical-ties/thp.json')
        print(single_numa_node(thp) or '')
    else:
        verify(args.root, args.group)
