# SPDX-License-Identifier: Apache-2.0
"""Capture and replay the native P2 window visibility in one A5 fixture."""

import json
from pathlib import Path

import numpy as np

BATCH = 16
QUERIES = 4
HISTORY = 131072
RESIDENT = 16384
WINDOWS = 4
TILE = 128
BUDGET = 512
ROUTES = {'p2': 1, 'cold_gm_fenced': 2}
GEOMETRY = dict(batch=BATCH, queries=QUERIES, history=HISTORY, resident=RESIDENT,
                windows=WINDOWS, miss_percent=10, aic=32, aiv=64, index_heads=32,
                prefetch_tokens=BUDGET, prefetch_rank_start=1024, seed=43)
FILE_NAME = 'cold_frontier.json'


def check_config(config, sequence, route):
    initial = next(iter(sequence))
    geometry = dict(batch=initial.geometry['batch'], queries=config.queries, history=config.history,
                    resident=config.resident, windows=config.intermediate_fetches, miss_percent=config.miss_percent,
                    aic=32, aiv=64, index_heads=config.index_heads, prefetch_tokens=config.prefetch_tokens,
                    prefetch_rank_start=config.prefetch_rank_start, seed=config.seed)
    if (geometry != GEOMETRY or config.mode != 'bench' or config.transport != 'urma'
            or tuple(config.batches) != (BATCH,) or config.input != 'synthetic'
            or len(sequence.rounds) != 1 or initial.source_round != 1
            or route not in ROUTES or config.mr28_cold_frontier_mode != ROUTES[route]
            or config.mr28_dedicated_service != route or not config.mr28_cold_gm_experiment):
        raise ValueError('Cold frontier requires the fixed BS16/Q4/H128K/R16K/W4/miss10 A5 comparison')


def header(fixture_sha256, library_sha256, calls):
    return dict(schema_version=1, geometry=GEOMETRY, source_round=1, calls=calls,
                fixture_sha256=fixture_sha256, library_sha256=library_sha256)


def validate_tape(values):
    """Accept only complete raw packed frontiers for the fixed tile geometry."""
    tape = np.asarray(values)
    if tape.shape != (WINDOWS, BATCH, QUERIES) or tape.dtype.kind not in 'iu':
        raise ValueError(f'Ill-shaped Cold frontier: {tape.shape}, {tape.dtype}')
    if np.any(tape < 0) or np.any(tape > HISTORY + (HISTORY << 32)):
        raise ValueError('Cold frontier contains a missing or out-of-range packed word')
    tape = tape.astype(np.int64, copy=False)
    for half in (tape & 0xffffffff, tape >> 32):
        if np.any(half > HISTORY) or np.any(half % TILE):
            raise ValueError('Cold frontier contains an out-of-range or non-tile-aligned cursor')
        if np.any(half[1:] < half[:-1]):
            raise ValueError('Cold frontier cursor regressed between windows')
    return tape


def load_tapes(path, expected_header):
    evidence = json.loads(Path(path).read_text())
    records = evidence['invocations']
    if (evidence['header'] != expected_header or evidence['mode'] != 1 or evidence['issues']
            or len(records) != expected_header['calls']):
        raise ValueError('Native Cold frontier evidence does not match the paired fixture, library or calls')
    if any(record['invocation'] != index or record['source_round'] != expected_header['source_round']
           for index, record in enumerate(records)):
        raise ValueError('Native Cold frontier invocation or source round out of order')
    return [validate_tape(record['frontier']) for record in records]


def attach(state, mode, source=None):
    """Install the immutable graph pointer and retain its backing device tensor."""
    forward = state.forward
    if forward.terminal is None or (mode == 2) != (source is not None):
        raise ValueError('Cold frontier capture needs a P2 Terminal and replay needs one tape')
    words = np.full((WINDOWS, BATCH, 16), -1, dtype=np.int64)
    if source is not None:
        words[..., :QUERIES] = source
    forward.cold_frontier_tape = state.torch.from_numpy(words).to(state.device)
    address = forward.cold_frontier_tape.data_ptr()
    if address % 128:
        raise RuntimeError('Cold frontier rows require cache-line alignment')
    encoded = np.asarray([address, mode], dtype='<u8').view('u1')
    forward.workspace[200:216].copy_(state.torch.from_numpy(encoded))


def prepare_capture(state, mode):
    """Clear evidence from untimed executions before the timed invocation."""
    tape = state.forward.cold_frontier_tape
    if mode == 1:
        tape[..., :QUERIES].fill_(-1)
    else:
        tape[..., QUERIES:2 * QUERIES].fill_(-1)
    state.forward.events_src.fill_(-1)


def decode_tokens(rows, sources, change_count, layout, counts):
    """Decode the retired GET assignments into per-request logical multisets."""
    if (rows.shape != (BATCH, 256) or sources.shape != (BATCH, 10240)
            or change_count.shape != (BATCH,) or layout.shape != (BATCH, HISTORY)
            or len(counts) != BATCH):
        raise ValueError('Unexpected Cold frontier Terminal evidence dimensions')
    results = []
    for b, record in enumerate(counts):
        early, final, total = (record[key] for key in
                               ('prefetch_requests', 'final_get_requests', 'total_get_requests'))
        windows = record['prefetch_by_window']
        by_query = record['final_get_by_query']
        if (record['request'] != b or len(windows) != WINDOWS or len(by_query) != QUERIES
                or int(rows[b, 8]) != total or int(change_count[b]) != total or total > sources.shape[1]):
            raise ValueError(f'Request {b}: inconsistent recorded and submitted Terminal assignments')
        indexed = sources[b, :total]
        if np.any(indexed < 0) or np.any(indexed >= HISTORY):
            raise ValueError(f'Request {b}: incomplete or invalid measured Get token source')
        logical = layout[b, indexed]
        if np.any(logical < 0) or np.any(logical >= HISTORY):
            raise ValueError(f'Request {b}: logical Get token outside history')
        offsets = np.cumsum([0, *windows])
        results.append(dict(
            **record,
            prefetch_tokens_by_window=[sorted(logical[start:stop].tolist())
                                       for start, stop in zip(offsets, offsets[1:])],
            final_get_tokens=sorted(logical[early:early + final].tolist()),
        ))
    return results


def read_invocation(state, mode, source_round, index, rows, counts):
    tape = state.forward.cold_frontier_tape.cpu().numpy()
    expected = validate_tape(tape[..., :QUERIES])
    if mode == 2:
        applied = validate_tape(tape[..., QUERIES:2 * QUERIES])
        if not np.array_equal(applied, expected):
            raise ValueError('Device-applied frontier differs from native capture')
    records = decode_tokens(rows, state.forward.events_src.cpu().numpy(),
                            state.rotating.change_count.cpu().numpy(), state.input_layout, counts)
    return dict(source_round=source_round, invocation=index, frontier=expected.tolist(), requests=records)


def compare_invocations(baseline, candidate):
    """Check exact multisets and query counts, independent of posting order."""
    if baseline['header'] != candidate['header'] or (baseline['mode'], candidate['mode']) != (1, 2):
        return [{'reason': 'evidence headers or modes differ'}]
    issues = [dict(route=mode, **issue) for mode, evidence in (('p2', baseline), ('cold_gm_fenced', candidate))
              for issue in evidence['issues']]
    for mode, evidence in (('p2', baseline), ('cold_gm_fenced', candidate)):
        records = evidence['invocations']
        if ([r['invocation'] for r in records] != list(range(evidence['header']['calls']))
                or any(r['source_round'] != evidence['header']['source_round']
                       or [s['request'] for s in r['requests']] != list(range(BATCH)) for r in records)):
            issues.append(dict(route=mode, reason='incomplete measured invocation/request coverage'))
    if issues:
        return issues
    for before, after in zip(baseline['invocations'], candidate['invocations']):
        identity = dict(source_round=before['source_round'], invocation=before['invocation'])
        if before['frontier'] != after['frontier']:
            issues.append(dict(**identity, reason='applied frontier differs'))
        for left, right in zip(before['requests'], after['requests']):
            windows = [epoch for epoch in range(WINDOWS)
                       if left['prefetch_tokens_by_window'][epoch] != right['prefetch_tokens_by_window'][epoch]]
            final = left['final_get_tokens'] != right['final_get_tokens']
            queries = [q for q in range(QUERIES)
                       if left['final_get_by_query'][q] != right['final_get_by_query'][q]]
            if windows or final or queries:
                issues.append(dict(**identity, request=left['request'], prefetch_windows=windows,
                                   final_get_tokens=final, final_get_query_counts=queries))
    return issues
