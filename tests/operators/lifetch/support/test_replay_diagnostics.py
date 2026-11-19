"""Continuous profiling keeps per-path state and source-round identities."""
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

from diagnostics import collect_replay


def test_profile_sequence_resets_only_at_sequence_boundaries(tmp_path, monkeypatch):
    log, samples = [], []
    inputs = [SimpleNamespace(source_round=n, scheduler_round=n + 100,
                              identities=((2, 'req', 1),)) for n in (11, 12)]
    class Sequence:
        rounds = ((11,), (12,))
        def __iter__(self):
            return iter(inputs)
    class Profile:
        def __init__(self, **kwargs):
            self.callback = kwargs['on_trace_ready']
            self.steps = 0
        def __enter__(self):
            return self
        def __exit__(self, *args):
            assert self.steps == 4
            self.callback(self)
        def step(self):
            self.steps += 1
        def export_chrome_trace(self, path):
            from pathlib import Path
            Path(path).write_text('{"traceEvents": []}')
    def schedule(**kwargs):
        assert kwargs == dict(wait=0, warmup=0, active=4, repeat=1)
        return None
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace(profiler=SimpleNamespace(
        profile=Profile, schedule=schedule, ProfilerActivity=SimpleNamespace(CPU=0, NPU=1))))
    monkeypatch.setitem(sys.modules, 'torch.profiler', SimpleNamespace(record_function=lambda name: nullcontext()))
    stream = SimpleNamespace(synchronize=lambda: None)
    def event(**kwargs):
        return SimpleNamespace(record=lambda stream: None, synchronize=lambda: None,
                               elapsed_time=lambda other: 0.01)
    torch = SimpleNamespace(npu=SimpleNamespace(current_stream=lambda device: stream, Event=event))
    layout = np.zeros((2, 4), dtype=np.int64)
    class Tensor:
        def __getitem__(self, index):
            return SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: layout[index]))
    state = SimpleNamespace(device=0, rotating=object(), storage=SimpleNamespace(logical_ids=Tensor()))
    def reset():
        log.append('reset')
        layout.fill(0)
    def prepare(item):
        log.append(('prepare', item.source_round, int(layout[1, 0])))
    def replay():
        layout[1] += 1
    def validate(item):
        log.append(('validate', item.source_round, int(layout[1, 0])))
        # A numerical failure is retained; Q/Index state still advances.
        if item.source_round == 11:
            raise AssertionError('numerical fixture failure')
    state.reset, state.prepare = reset, prepare
    state.forward = SimpleNamespace(calibrate_counter_mhz=lambda: 1000,
        set_diagnostics=lambda enabled: log.append(('diagnostics', enabled)), diagnostic_cycles=lambda: [])
    monkeypatch.setattr('diagnostics.algorithm_before', lambda rotating: {})
    def algorithm_after(rotating, before):
        old = int(before['logical_ids'][0, 0])
        assert int(layout[1, 0]) == old + 1
        samples.append(old)
        return dict(totals=dict(prefetch=old, final=1), requests=[])
    monkeypatch.setattr('diagnostics.algorithm_after', algorithm_after)
    monkeypatch.setattr('diagnostics.account', lambda *args, **kwargs: dict(stages_us=dict(li=2)))
    config = SimpleNamespace(warmup=1, iterations=2, transport='urma', level=1)
    result = collect_replay(torch, state, SimpleNamespace(replay=replay), Sequence(), validate, config, tmp_path)
    assert log.count('reset') == 3  # One warmup sequence, two profiled sequences.
    assert [item for item in log if isinstance(item, tuple) and item[0] == 'prepare'] == [
        ('prepare', 11, 0), ('prepare', 12, 1)] * 3
    assert samples == [0, 1, 0, 1]
    assert result['correctness'] == 'FAIL' and result['samples'] == 4
    for source in (11, 12):
        data = json.loads((tmp_path / f'round-{source}/full_attribution.json').read_text())['samples']
        assert [sample['round'] for sample in data] == [0, 1]
        assert all(sample['source_round'] == source for sample in data)
        assert all(sample['scheduler_round'] == source + 100 for sample in data)
    assert log[-1] == ('diagnostics', False)
