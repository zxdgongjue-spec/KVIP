"""Independent Get preserves descriptors, restores outside events, and checks copies."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from performance import measure_serial_get


@pytest.mark.parametrize('count', [0, 2])
@pytest.mark.parametrize('width', [656, 2624])
@pytest.mark.parametrize('copy_payload', [False, True])
def test_get_graph_excludes_li_lidu_and_restores_each_sample(count, width, copy_payload):
    log = []
    active = []
    output = torch.arange(4 * width).to(torch.uint8).reshape(4, width)
    expected = output.clone()
    destinations = torch.tensor([1, 3])[:count]

    class Graph:
        def __init__(self):
            self.calls = []

        def replay(self):
            log.append('replay')
            for call in self.calls:
                call()

        def reset(self):
            log.append('destroy')

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self, stream):
            log.append('event')

        def synchronize(self):
            log.append('event_sync')

        def elapsed_time(self, other):
            return .012

    stream = SimpleNamespace(synchronize=lambda: log.append('sync'))

    @contextmanager
    def capture(graph, stream):
        active.append(graph)
        yield
        active.pop()

    def copy():
        log.append('get')
        if count:
            assert not torch.equal(output[destinations], expected[destinations])
        if copy_payload:
            output.index_copy_(0, destinations, expected[destinations])

    def fetch(*, scale_offsets):
        assert scale_offsets is False
        if active:
            active[-1].calls.append(copy)
        copy()

    def verify(selected):
        log.append('verify')
        assert selected == 'retained-topk'
        torch.testing.assert_close(output, expected)

    device = SimpleNamespace(npu=SimpleNamespace(
        current_stream=lambda: stream, Stream=lambda: stream, NPUGraph=Graph,
        Event=Event, graph=capture))
    native = SimpleNamespace(output=output, total=torch.tensor([count]), outputs=destinations,
                             fetch=fetch, verify=verify)
    state = SimpleNamespace(native=native, selected='retained-topk')
    result = measure_serial_get(device, state, warmup=1, iterations=3)
    assert result['correctness'] == ('FAIL' if count and not copy_payload else 'PASS')
    assert result['timing']['samples_us'] == [12.] * 3
    assert result['timing']['timing_protocol'] == 'independent_get_graph_events'
    torch.testing.assert_close(output, expected)
    for index, item in enumerate(log):
        if item == 'replay':
            assert log[index - 1] == 'event'
            assert log[index + 1:index + 4] == ['get', 'event', 'event_sync']
            assert log[index + 4] == 'verify'


@pytest.mark.parametrize('level', [1, 2, 3])
def test_independent_get_is_in_diagnostic_summary(tmp_path, level):
    from diagnostics import summarize
    from reporting import atomic_json, timing_summary

    atomic_json(tmp_path / 'serial/diagnose/round-1/serial_get_only.json', dict(
        source_round=1, timing=timing_summary([10., 20.]), correctness='PASS', failures=[]))
    result = summarize(tmp_path, SimpleNamespace(
        level=level, iterations=2, stages=('li',), transport='urma'))
    assert len(result['serial_get_only']) == 1
    for filename in (f'level{level}_summary.log', 'diagnostic_summary.log'):
        text = (tmp_path / filename).read_text()
        assert 'SERIAL-GET source_round=1' in text
        assert 'mean/p50/p99_us=15.00/15.00/19.90' in text


def test_serial_get_summary_and_csv_are_distinct_from_profiled_fetch(tmp_path):
    import csv
    from reporting import collect_results, atomic_json, save_summary, timing_summary

    config = SimpleNamespace(mode='diagnose', transport='urma', batches=(1,), miss_percent=10)
    atomic_json(tmp_path / 'case/report.json', dict(batch=1, correctness='PASS', timing=dict(
        native_chain=timing_summary([100]), prototype_with_update=timing_summary([80]),
        native_get_only=timing_summary([20, 30]))))
    rows = collect_results(tmp_path, config, 0)
    save_summary(tmp_path, rows)
    assert 'Serial Get mean' in (tmp_path / 'summary.log').read_text()
    with (tmp_path / 'results.csv').open() as stream:
        row, = csv.DictReader(stream)
    assert row['serial_get_only_mean_us'] == '25.0'
    assert row['serial_get_only_p50_us'] == '25.0'
    assert row['serial_get_only_samples'] == '2'


def test_batch_runner_samples_get_for_each_round_after_serial_graph(tmp_path, monkeypatch):
    import json
    from execution import measure_replay_batch
    from reporting import timing_summary
    from test_performance import fake_device

    device, log = fake_device()
    stream = device.npu.Stream()
    device.npu.current_stream = lambda device=None: stream
    inputs = [SimpleNamespace(source_round=n) for n in (1, 2)]

    class Sequence:
        rounds = ((1,), (2,))

        def __iter__(self):
            return iter(inputs)

    current = []
    state = SimpleNamespace(
        device=0, reset=lambda: log.append('reset'),
        prepare=lambda item: current.append(item.source_round),
        forward=lambda: log.append('full_native'),
        validate=lambda item: log.append(('validate', item.source_round)),
        native=SimpleNamespace(counts=torch.tensor([2, 2])))

    def profile(torch, graph, **kwargs):
        assert kwargs['serial'] and kwargs['calls'] == 1
        graph.replay()
        graph.completed()
        log.append(('profile', current[-1]))
        timing = dict(timing_summary([10]), stage_mean_us=dict(li=3, lidu_prepare=2, fetch=5),
                      inter_kernel_gap_mean_us=0, overlap_mean_us=0, outside_invocations_mean_us=0)
        return dict(timing=timing, samples=[dict(us=10, stages_us=timing['stage_mean_us'])],
                    accounting={}, trace='trace.json')

    def get(torch, selected, **kwargs):
        assert selected is state
        assert kwargs == dict(warmup=0, iterations=2)
        assert log[-1] == ('validate', current[-1])
        log.append(('get_only', current[-1]))
        return dict(timing=timing_summary([3, 4]), correctness='PASS', failures=[])

    monkeypatch.setattr('performance.profile_graph_batch', profile)
    monkeypatch.setattr('performance.measure_serial_get', get)
    config = SimpleNamespace(mode='diagnose', transport='urma', warmup=0, iterations=2)
    result = measure_replay_batch(device, [state], Sequence(), config, tmp_path, fused=False)
    assert result['get_only']['samples_us'] == [3, 4, 3, 4]
    assert result['correctness'] == 'PASS'
    assert [item for item in log if isinstance(item, tuple) and item[0] == 'get_only'] == [
        ('get_only', 1), ('get_only', 2)]
    for n in (1, 2):
        data = json.loads((tmp_path / f'diagnose/round-{n}/serial_get_only.json').read_text())
        assert data['source_round'] == n and data['timing']['samples'] == 2
