import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / 'support'))
from owner_pack_tail import capture, ordered_topk


class Buffer:
    def __init__(self, array):
        self.array = array

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def zero_(self):
        self.array.fill(0)


def test_single_state_ordered_capture_and_off_on_pairs(tmp_path):
    layout = np.tile(np.arange(2050)[::-1], (16, 1))
    keys = np.ones((16, 4, 2050), dtype=np.uint16)
    keys[..., 1] = 2
    valid = np.ones_like(keys, dtype=bool)
    valid[..., -1] = False
    inputs = SimpleNamespace(logical_score_valid=valid, logical_score_keys=keys,
                             source_paths=('/capture',), source_round=1)
    expected = ordered_topk(inputs, layout, 2048)
    np.testing.assert_array_equal(expected[0, 0], [1, *range(2048, 1, -1)])
    ordered = Buffer(expected.copy())
    trace = Buffer(np.zeros((64, 848), dtype=np.int64))
    actions = []
    state = SimpleNamespace(
        input_layout=layout, update=SimpleNamespace(logical_output=ordered),
        forward=SimpleNamespace(query_owner_trace=trace, set_query_owner_capture=actions.append),
        rotating=SimpleNamespace(rows=Buffer(np.zeros((16, 256), dtype=np.uint32)),
                                 final_posts=Buffer(np.zeros((16, 4, 32), dtype=np.uint32)),
                                 arguments=[None]*9+[Buffer(expected.copy())]),
        reset=lambda: actions.append('reset'), validate=lambda inputs: actions.append('validate'),
        topk_set_digest='receipt', query_owners={'owners':64}, layout_receipts=[])
    config = SimpleNamespace(group=17, warmup=4, production_revision='abc')

    def replay():
        enabled = actions[-1]
        if enabled:
            assert not trace.array.any()
            trace.array[:, 16] = 45
        return 3 if enabled else 2

    assert capture(state, inputs, config, tmp_path, 1200, replay) == 10
    data = json.loads((tmp_path/'owner_tail.json').read_text())
    assert [s['first'] for s in data['samples']] == ['off', 'on', 'off', 'on', 'off']
    assert len(actions) == 30 and actions.count('reset') == actions.count('validate') == 10
    for sample in data['samples']:
        assert (sample['event_off_us'],sample['event_on_us']) == (2,3)
        assert sample['requests'][0]['queries'][1]['cross_q_duplicate'] == 2048
    np.testing.assert_array_equal(np.load(tmp_path/'owner_tail_topk.npz')['actual'][4], expected)
    ordered.array[..., [3,4]] = ordered.array[..., [4,3]]
    with pytest.raises(AssertionError, match='ordered logical token mismatch'):
        capture(state, inputs, config, tmp_path, 1200, replay)
    failure = json.loads((tmp_path/'owner_tail_order_failure.json').read_text())
    assert failure['rank'] == 3
    assert (tmp_path/'owner_tail_topk.npz').exists()
