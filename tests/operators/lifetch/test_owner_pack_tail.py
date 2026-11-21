import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / 'support'))
from owner_pack_tail import capture


class Buffer:
    def __init__(self, array):
        self.array = array

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def zero_(self):
        self.array.fill(0)


def test_single_state_checked_capture_and_off_on_pairs(tmp_path):
    layout = np.tile(np.arange(2050)[::-1], (16, 1))
    positions = np.tile(np.arange(2048), (16, 4, 1))
    actual = layout[np.arange(16)[:, None, None], positions]
    expected = np.sort(actual, axis=-1)
    inputs = SimpleNamespace(source_paths=('/capture',), source_round=1)
    ordered = Buffer(actual.copy())
    trace = Buffer(np.zeros((64, 848), dtype=np.int64))
    actions = []
    state = SimpleNamespace(
        input_layout=layout, update=SimpleNamespace(logical_output=ordered),
        batch=16, history=2050, workspace=SimpleNamespace(topk=Buffer(positions),
            inputs=SimpleNamespace(unpack_indices=Buffer(np.arange(64)))),
        forward=SimpleNamespace(query_owner_trace=trace, set_query_owner_capture=actions.append),
        rotating=SimpleNamespace(rows=Buffer(np.zeros((16, 256), dtype=np.uint32)),
                                 final_posts=Buffer(np.zeros((16, 4, 32), dtype=np.uint32)),
                                 arguments=[None]*9+[Buffer(actual.copy().astype(np.int32))]),
        reset=lambda: actions.append('reset'), validate=lambda inputs: actions.append('validate'),
        topk_set_digest='receipt', query_owners={'owners':64}, layout_receipts=[])
    config = SimpleNamespace(group=17, warmup=4, production_revision='abc')

    def replay():
        enabled = actions[-1]
        if enabled:
            assert not trace.array.any()
            trace.array[:, 16] = 45
        return 3 if enabled else 2

    assert capture(state, inputs, config, tmp_path, 1200, replay, expected) == 10
    data = json.loads((tmp_path/'owner_tail.json').read_text())
    assert [s['first'] for s in data['samples']] == ['off', 'on', 'off', 'on', 'off']
    assert len(actions) == 30 and actions.count('reset') == actions.count('validate') == 10
    for sample in data['samples']:
        assert (sample['event_off_us'],sample['event_on_us']) == (2,3)
        assert sample['requests'][0]['queries'][1]['cross_q_duplicate'] == 2048
    # Position-order variations within each Q satisfy the serial set contract.
    ordered.array[..., [3, 4]] = ordered.array[..., [4, 3]]
    assert capture(state, inputs, config, tmp_path, 1200, replay, expected) == 10
    # Substitute the same wrong token in both raw LI and tail output.
    ordered.array[0, 0, 0] = layout[0, 2048]
    positions[0, 0, 0] = 2048
    with pytest.raises(AssertionError, match='Strong serial TopK set differs at row 0, Q1'):
        capture(state, inputs, config, tmp_path, 1200, replay, expected)


def test_archived_coverage_and_same_repeat_pairs():
    from copy import deepcopy
    from owner_pack_tail import match_plans

    def receipt(plan):
        return (f'{plan:064x}', f'{plan+10:064x}')
    base = [receipt(0)] * 16
    altered = [receipt(1), *base[1:]]
    other = [receipt(2), *base[1:]]
    samples = [
        dict(layout_receipts=on, off_layout_receipts=off, event_on_us=11+i, event_off_us=10)
        for i, (on, off) in enumerate([(base, base), (base, altered), (altered, altered),
                                       (base, other), (altered, base)])
    ]
    archive = [base] * 20
    assert match_plans(samples, archive) == dict(
        archive_rows=78, archive_calls=3, archive_off_rows=77, archive_off_calls=2, on_off_rows=77, on_off_calls=2, matched_on_minus_off_p50_us=2)
    # All runs still have valid semantics if the event plans never repeat.
    mismatch = [dict(s, off_layout_receipts=other) for s in samples]
    assert match_plans(mismatch, archive)['matched_on_minus_off_p50_us'] is None
    corrupted = deepcopy(samples)
    corrupted[1]['off_layout_receipts'] = [(receipt(0)[0], 'a' * 64), *base[1:]]
    with pytest.raises(ValueError, match='Different physical Index for identical events, request 0'):
        match_plans(corrupted, archive)
