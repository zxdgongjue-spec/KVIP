"""Exercise trace-slot binding and frozen G2 oracle reuse without an NPU."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from support import joined_tail_capture as capture


class HostTrace:
    def __init__(self):
        self.array = np.zeros((64, 96), dtype=np.int64)

    def zero_(self):
        self.array.fill(0)

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class Slot:
    def __init__(self):
        self.forward = self
        self.query_owner_trace = HostTrace()
        self.query_owner_trace_fields = ['entry'] + [str(k) for k in range(1, 96)]
        self.query_owners = dict(owners=64, queues=64, processed=1)
        self.layout_receipts, self.topk_set_digest = [], 'fixture'
        self.checked = 0

    def set_query_owner_capture(self, enabled):
        self.enabled = enabled

    def reset(self):
        self.enabled = False

    def validate(self, inputs):
        self.checked += 1


@pytest.mark.parametrize('wrong_slot', [False, True])
def test_capture_matches_owner_slot_to_three_li_graph(tmp_path, monkeypatch, wrong_slot):
    states = [Slot() for _ in range(20)]
    inputs = SimpleNamespace(source_round=1, source_paths=[])
    config = SimpleNamespace(warmup=4, group=2, production_revision='diag-fixture')
    monkeypatch.setattr(capture, 'workload', lambda state: [dict(prefetch=2, final=1)])
    monkeypatch.setattr(capture, 'fused_li_output', lambda state: [])
    monkeypatch.setattr(capture, 'check_scores', lambda *args: {'calls': 1, 'queries': 64, 'exact_queries': 0})

    def record(folder):
        samples = []
        for i, state in enumerate(states):
            if state.enabled or (wrong_slot and i == 1):
                state.query_owner_trace.array[:, 0] = 1000 + i
            samples.append(dict(invocation=i, owner_topk_kernel='v9_owner_topk_resolve_update',
                                intervals_us=dict(li=[[1, 2], [3, 5], [6, 10]], resolve_update=[[12, 24]]),
                                stages_us=dict(resolve_update=12), work_totals=dict(prefetch=2, final=1)))
        return dict(samples=samples, timing=dict(graph_calls=20, measured_graphs=1,
                    timing_protocol='one_profiled_graph', graph_event_us=500))

    args = (states, inputs, config, tmp_path, 50, record, ({'strong_serial': []}, {}))
    if wrong_slot:
        with pytest.raises(ValueError, match='ownership mismatch'):
            capture.capture(*args)
        assert not (tmp_path/'joined_tail.json').exists()
        return
    assert capture.capture(*args) == 80
    data = json.loads((tmp_path/'joined_tail.json').read_text())
    assert [p['name'] for p in data['passes']] == ['off_before', 'odd_on', 'even_on', 'off_after']
    assert len(data['trace_fields']) == 96 and all(s.checked == 4 for s in states)
    for call in range(20):
        records = [p['calls'][call] for p in data['passes']]
        traced, = [r for r in records if 'ticks' in r]
        assert all(r['call'] == call + 1 and r['tail_us'] == 12 for r in records)
        assert all(row[0] == 1000 + call for row in traced['ticks'])


def test_g2_reference_reuses_saved_serial_ids(tmp_path):
    folder = tmp_path/'g2/logical-ties/results/f/cases/diagnose/fused'
    folder.mkdir(parents=True)
    ids = np.broadcast_to(np.arange(2048, dtype=np.int32), (16, 4, 2048))
    inputs = SimpleNamespace(query_ends=np.tile(np.arange(2048, 2052), (16, 1)), geometry={'sparse_mode': 3})
    identity = {k: 'same' for k in ('inputs', 'host', 'npu', 'visible', 'operator', 'sparse_mode', 'torch', 'torch_npu', 'cann')}
    source = dict(origin='validated-cache', identity=identity)
    receipt = dict(group=2, calls=20, queries=1280, native_reference=source,
                   serial_set_digest=capture.topk_fingerprint(ids))
    (folder / 'score_topk_check.json').write_text(json.dumps(receipt))
    np.savez_compressed(folder / 'score_topk_ids.npz', strong_serial=ids)
    refs, meta = capture.archived_g2_serial(tmp_path, inputs, identity)
    assert np.array_equal(refs['strong_serial'], ids) and meta['origin'] == 'completed-archive'
    with pytest.raises(ValueError, match='native reference differs'):
        capture.archived_g2_serial(tmp_path, inputs, dict(identity, inputs='different'))
