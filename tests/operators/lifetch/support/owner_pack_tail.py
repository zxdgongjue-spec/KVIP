"""Capture the instrumented owner-pack tail against an archived Level 1 run."""
import argparse
import hashlib
import json
from statistics import median
from math import isfinite
from pathlib import Path

import numpy as np

from query_owner_report import checked_layout_plan, compare_layout_receipts, load
from reporting import atomic_json

REPEATS = 5


def ordered_topk(inputs, layout, k):
    # The native physical-trunk merge emits GT, then EQ, each stable in its
    # input order. Prefix cutoffs never decrease, so final GTs stay in physical
    # order and no earlier final-equal token loses to a later equal token.
    result = np.empty((len(layout), 4, k), dtype=np.int32)
    for row, physical in enumerate(layout):
        for q in range(4):
            valid = inputs.logical_score_valid[row, q, physical]
            tokens = physical[valid]
            keys = inputs.logical_score_keys[row, q, tokens]
            cutoff = np.partition(keys, len(keys) - k)[-k]
            gt = tokens[keys > cutoff]
            result[row, q] = np.concatenate((gt, tokens[keys == cutoff][:k-len(gt)]))
    return result


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


def capture(state, inputs, config, directory, ticks_per_us, replay_and_time):
    if not isfinite(ticks_per_us) or ticks_per_us <= 0:
        raise ValueError('Invalid owner counter frequency')
    expected = ordered_topk(inputs, state.input_layout, 2048)
    samples, sequences = [], []
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
            actual = state.update.logical_output.cpu().numpy().reshape(16, 4, 2048).astype('<i4')
            mismatch = np.argwhere(actual != expected)
            if len(mismatch):
                b, q, i = map(int, mismatch[0])
                np.savez_compressed(directory / 'owner_tail_topk.npz', expected=expected, failed=actual)
                detail = dict(repeat=repeat, trace_enabled=enabled, row=b, query=q+1, rank=i,
                              actual=int(actual[b,q,i]), expected=int(expected[b,q,i]),
                              mismatched_positions=len(mismatch))
                atomic_json(directory / 'owner_tail_order_failure.json', detail)
                raise AssertionError(f'Owner-pack ordered logical token mismatch: {detail}')
            state.validate(inputs)
            checked += 1
            if not enabled:
                sample['off_layout_receipts'] = state.layout_receipts
            else:
                sequences.append(actual)
                ticks = state.forward.query_owner_trace.cpu().numpy()
                if ticks.shape != (64, 848) or np.any(ticks[:, 16] <= 0):
                    raise ValueError(f'Missing per-Q native owner stamps, observed shape {ticks.shape}')
                sample.update(ticks=ticks.tolist(), requests=workload(state),
                              topk_set_digest=state.topk_set_digest,
                              topk_order_digest=hashlib.sha256(actual.tobytes()).hexdigest(),
                              ordered_queries=64, query_owners=state.query_owners,
                              layout_receipts=state.layout_receipts)
        samples.append(sample)
    np.savez_compressed(directory / 'owner_tail_topk.npz', expected=expected, actual=sequences)
    atomic_json(directory / 'owner_tail.json', dict(
        schema_version=1, group=config.group, warmup=config.warmup, independent_states=1,
        native_revision=config.production_revision, input_paths=list(map(str, inputs.source_paths)),
        source_round=inputs.source_round, ticks_per_us=ticks_per_us, checked_calls=checked, samples=samples))
    return checked


def reference(root, group):
    summary = load(root, group, 'owner-pack', row_dependencies=True)
    compare_layout_receipts({'owner-pack': summary[11]}, 'owner-pack')
    if len(set(summary[1])) != 1:
        raise ValueError('Archived TopK changes between invocations')
    return summary[0], summary[1][0], summary[11]


def match_plans(samples, archive):
    if len(samples) != REPEATS or len(archive) != 20:
        raise ValueError('Expected five paired diagnostics and twenty archived invocations')
    states = {}
    archived_plans = {checked_layout_plan(rows, states, 'archive') for rows in archive}
    archived_rows = {entry for call in archived_plans for entry in enumerate(call)}
    archive_rows = archive_calls = on_off_rows = 0
    matched_deltas = []
    for sample in samples:
        on = checked_layout_plan(sample['layout_receipts'], states, 'trace-on')
        off = checked_layout_plan(sample['off_layout_receipts'], states, 'trace-off')
        archive_rows += sum(entry in archived_rows for entry in enumerate(on))
        archive_calls += int(on in archived_plans)
        on_off_rows += sum(a == b for a, b in zip(on, off))
        if on == off:
            matched_deltas.append(sample['event_on_us'] - sample['event_off_us'])
    return dict(archive_rows=archive_rows, archive_calls=archive_calls, on_off_rows=on_off_rows,
                on_off_calls=len(matched_deltas),
                matched_on_minus_off_p50_us=median(matched_deltas) if matched_deltas else None)


def verify(root, group):
    base = Path((root / 'owner-tail-reference.txt').read_text().strip())
    inputs, digest, archive = reference(base, group)
    directory = root / f'g{group}'
    paths = list((directory / 'owner-pack/results').glob('*/cases/diagnose/fused/owner_tail.json'))
    if len(paths) != 1:
        raise ValueError(f'G{group}: expected one completed owner-tail trace, found {len(paths)}')
    path = paths[0]
    data = json.loads(path.read_text())
    result = json.loads((path.parent / 'path_result.json').read_text())
    if (result['correctness'] != 'PASS' or result['failures'] or data['checked_calls'] != 2 * REPEATS
            or result['production_revision'] != data['native_revision']):
        raise ValueError(f'G{group}: incomplete diagnostic correctness checks')
    actual_inputs = json.loads((path.parent.parent / 'replay_input.json').read_text())
    if inputs != actual_inputs or data['input_paths'] != actual_inputs['source_paths']:
        raise ValueError(f'G{group}: diagnostic input differs from the archived owner-pack run')
    samples = data['samples']
    if (data['group'] != group or data['native_revision'] != (root/'owner-pack.txt').read_text().strip()
            or len(samples) != REPEATS or [s['repeat'] for s in samples] != list(range(REPEATS))
            or any(s['topk_set_digest'] != digest or s['ordered_queries'] != 64 for s in samples)
            or len({s['topk_order_digest'] for s in samples}) != 1):
        raise ValueError(f'G{group}: diagnostic TopK or input count differs from the reference')
    data['plan_match'] = match_plans(samples, archive)
    atomic_json(path, data)
    for name in ('owner_tail.json', 'owner_tail_topk.npz'):
        target = path.parent / name
        if not target.is_file():
            raise FileNotFoundError(target)
        (directory / name).symlink_to(target.relative_to(directory))
    print(f'G{group} OWNER_TAIL samples={REPEATS} exact_ordered_TopK={64*REPEATS}/{64*REPEATS} '
          f'KV=PASS ordered_Index=PASS trace={directory / "owner_tail.json"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('reference','verify'))
    parser.add_argument('root', type=Path)
    parser.add_argument('group', type=int)
    args = parser.parse_args()
    (reference if args.command == 'reference' else verify)(args.root, args.group)
