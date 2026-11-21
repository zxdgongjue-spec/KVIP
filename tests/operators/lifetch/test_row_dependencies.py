"""The row-dependency replay requires TopK, KV and ordered physical Index parity."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / 'support'))
from execution import FusedReplayState
from query_owner_report import report_row_dependencies


class Array:
    """Expose the post-graph CPU readback interface without a device."""

    def __init__(self, values):
        self.values = np.asarray(values)

    def __getitem__(self, key):
        return Array(self.values[key])

    def cpu(self):
        return self

    def numpy(self):
        return self.values


@pytest.mark.parametrize('fault', (None, 'keys', 'kv', 'layout', 'valid'))
def test_ordered_layout_with_a_displaced_final_source(fault):
    # Prefetch swaps 4 into slot 0. Q1 selects the token displaced from 0;
    # its current source is now 4, so the final swap moves it from 4 to 2.
    layout = np.array([[4, 1, 0, 3, 2, 5]])
    valid = np.ones((1, 3), dtype=np.int32)
    if fault == 'layout':
        layout[0, [2, 4]] = layout[0, [4, 2]]
    if fault == 'valid':
        valid[0, 2] = 0
    pre = np.arange(6).reshape(1, -1)
    addresses = 100 + pre
    candidates = np.zeros((1, 2049), dtype=np.int32)
    destinations = candidates.copy()
    candidates[0, 0], destinations[0, 0] = 4, 0
    candidates[0, 2048], destinations[0, 2048] = 0, 2
    flags = np.zeros((1, 16, 2), dtype=np.int32)
    flags[0, [0, 12], 1] = 1
    keys = layout + 10
    if fault == 'keys':
        keys[0, 2] += 1
    values = layout[:, :3, None] + 200
    if fault == 'kv':
        values[0, 2, 0] += 1
    state = object.__new__(FusedReplayState)
    state.batch = 1
    state.verify_ranked_events = True
    state.query_owner_arm = 'row-deps'
    state.input_layout = pre
    state.logical_addresses = addresses
    state.pages = SimpleNamespace(token_payload=lambda row, ids: ids[:, None] + 200)
    state.storage = SimpleNamespace(
        logical_ids=Array(np.r_[pre, layout]), keys=Array(np.r_[pre, keys]),
        scales=Array(np.r_[pre, layout + 20]), addresses=Array(np.r_[pre, addresses[0, layout]]),
        hot_valid=Array(np.r_[valid, valid]),
    )
    state.forward = SimpleNamespace(events_src=Array(candidates), events_dst=Array(destinations), flags=Array(flags))
    inputs = SimpleNamespace(
        index_for_layout=lambda ids: (ids + 10, ids + 20, np.ones_like(ids)),
        captured_hot_valid=np.array([[1, 1, 0]], dtype=np.int32),
    )
    if fault is None:
        state.validate_index(inputs, values)
        assert len(state.layout_receipts) == 1
        assert all(len(digest) == 64 for digest in state.layout_receipts[0])
        return
    error = {
        'keys': 'Index keys mismatch',
        'kv': 'persistent hot KV mismatch',
        'layout': 'does not implement ordered',
        'valid': 'hot-valid bitmap differs',
    }
    with pytest.raises((ValueError, RuntimeError), match=error[fault]):
        state.validate_index(inputs, values)


@pytest.mark.parametrize('fault', (None, 'topk', 'layout', 'kv', 'tail', 'sample_order',
                                   'unmatched', 'partial', 'uneven'))
def test_report_gates_and_measures_graph_and_resolver(tmp_path, capsys, fault):
    def write(path, value):
        path.write_text(json.dumps(value))

    for arm, revision, full, tail in (('fixed-order', '1', 70, 31),
                                      ('offset', '2', 60, 25),
                                      ('row-deps', '3', 50, 20)):
        candidate = arm == 'row-deps'
        fused = tmp_path / 'g7' / arm / 'results' / 'case' / 'fused'
        details = fused / 'diagnose' / 'round-1'
        details.mkdir(parents=True)
        digest = revision * 64
        sha = revision * 40
        (tmp_path / f'{arm}.txt').write_text(sha)
        (tmp_path / f'ranked-{sha[:8]}.binary.sha256').write_text(digest)
        cfg = dict(device=0, window_limit=256, query_owner_arm=arm, group=7, intermediate_fetches=4,
                   prefetch_tokens=1024, level=1, warmup=4, iterations=20)
        write(fused.parent / 'config.json', cfg)
        inputs = dict(geometry=dict(batch=16, token_bytes=656, history=131328, resident=16384),
                      rounds=1, source_paths=['trace'], identities=[[i // 2, i, 0] for i in range(16)])
        write(fused.parent / 'replay_input.json', inputs)
        write(fused.parents[2] / 'thp.json', dict(
            coverage_min_pct=100, ub_device='fixture', numa_pages_overlapping_vmas={'0': 42}))
        spans = [full - 1] * 20
        timing = dict(samples_us=spans, mean_us=full, stage_mean_us={'li': full - tail, 'resolve_update': tail},
                      measured_graphs=1, timing_protocol='one_profiled_graph')
        out = dict(production_revision=sha, correctness='PASS', failures=[], transport='urma', runtime_workers=40,
                   input_paths=['trace'], all_rounds=timing, binary_sha256=digest,
                   topk_set_digests=['a' * 64] * 20,
                   query_owners=[dict(owners=64, queues=64, processed=16)] * 20,
                   offset_checks=[dict(extended_slots=0, independent_rows=16)] * 20,
                   layout_receipts=[[['b' * 64, 'c' * 64] for _ in range(16)] for _ in range(20)])
        if candidate and fault == 'topk':
            out['topk_set_digests'][-1] = 'd' * 64
        if candidate and fault == 'layout':
            out['layout_receipts'][-1][-1][1] = 'd' * 64
        if candidate and fault == 'unmatched':
            out['layout_receipts'] = [[['d' * 64, 'e' * 64] for _ in range(16)] for _ in range(20)]
        if candidate and fault == 'partial':
            out['layout_receipts'][-1][-1] = ['d' * 64, 'e' * 64]
        if fault == 'uneven':
            for call in out['layout_receipts'][(-2 if candidate else -1):]:
                call[-1] = ['d' * 64, 'e' * 64]
        if candidate and fault == 'kv':
            out.update(correctness='FAIL', failures=['KV bytes: request=15'])
        write(fused / 'path_result.json', out)
        totals = dict(KV_read_bytes=16 * 656, final=16, prefetch=0, unused_prefetch=0, selected_unique=128,
                      initial_cold=16)
        requests = [dict(windows=[dict(posted=0)] * 4, prefetch=0)] * 16
        rows = [dict(us=span, accounting=dict(invocation=i, stages_us={'resolve_update': tail}),
                     algorithm=dict(totals=totals, requests=requests)) for i, span in enumerate(spans)]
        if candidate and fault == 'tail':
            rows[0]['accounting']['stages_us']['resolve_update'] = full + 1
        if candidate and fault == 'sample_order':
            rows[0]['accounting']['invocation'] = 1
        write(details / 'full_attribution.json', dict(samples=rows))
    if fault not in (None, 'unmatched', 'partial', 'uneven'):
        with pytest.raises(ValueError):
            report_row_dependencies(tmp_path, 7)
        assert capsys.readouterr().out == ''
        return
    report_row_dependencies(tmp_path, 7)
    report = capsys.readouterr().out
    assert 'same_TopK=20/20 KV=PASS ordered_Index=PASS per_arm=320/320' in report
    assert 'graph_saved_mean/P50_us=+10.00/+10.00 tail_saved_mean/P50_us=+5.00/+5.00' in report
    coverage = {None: '320/320 same_event_calls=20/20 placement=matched POSITIVE',
                'unmatched': '0/320 same_event_calls=0/20 placement=matched NO_SHARED_INDEX_EVENTS',
                'partial': '319/320 same_event_calls=19/20 placement=matched PARTIAL_INDEX_EVENT_COVERAGE',
                'uneven': '320/320 same_event_calls=19/20 placement=matched PARTIAL_INDEX_EVENT_COVERAGE'}
    assert f'same_event_layout_rows={coverage[fault]}' in report
