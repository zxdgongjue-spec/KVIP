"""CPU regressions for the unified LIFetch performance module."""
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from performance import TOPK, build_group_plan, measure_graph


def fake_device():
    events = []
    class Stream:
        def synchronize(self):
            events.append('sync')
    class Graph:
        def replay(self):
            events.append('replay')
        def reset(self):
            events.append('destroy')
    class Event:
        def __init__(self, **kwargs):
            pass
        def record(self, stream):
            events.append('event')
        def synchronize(self):
            events.append('event_sync')
        def elapsed_time(self, other):
            return .012
    @contextmanager
    def capture(graph, stream):
        events.append('capture')
        yield
        events.append('captured')
    return SimpleNamespace(npu=SimpleNamespace(
        current_stream=Stream, Stream=Stream, NPUGraph=Graph, Event=Event, graph=capture)), events


def test_restore_is_outside_events_and_warmup_excluded():
    torch, events = fake_device()
    result = measure_graph(torch, lambda: events.append('call'), lambda: events.append('restore'),
                           lambda: events.append('check'), warmup=2, iterations=3)
    assert result['timing']['samples_us'] == [12., 12., 12.]
    assert result['correctness'] == 'PASS'
    assert events.count('replay') == 5 and events[-1] == 'destroy'
    for index, event in enumerate(events):
        if event == 'replay':
            assert events[index-3:index] == ['restore', 'sync', 'event']
            assert events[index+1:index+4] == ['event', 'event_sync', 'check']


def test_numerical_failure_keeps_all_completed_timings():
    torch, events = fake_device()
    def incorrect():
        raise AssertionError('wrong KV bytes')
    result = measure_graph(torch, lambda: None, lambda: None, incorrect, warmup=1, iterations=2)
    assert result['correctness'] == 'FAIL'
    assert len(result['failures']) == 4
    assert result['timing']['samples'] == 2
    assert events[-1] == 'destroy'


def test_unsafe_device_completion_stops_replays():
    torch, events = fake_device()
    def incomplete():
        raise RuntimeError('completion incomplete')
    with pytest.raises(RuntimeError, match='completion incomplete'):
        measure_graph(torch, lambda: None, lambda: None, incomplete, warmup=1, iterations=2)
    assert 'replay' not in events


def test_continuous_sequence_restores_once_not_between_rounds():
    from performance import measure_sequence
    torch, events = fake_device()
    rounds = [SimpleNamespace(source_round=i, scheduler_round=i+100) for i in (1, 2)]
    graphs = {i: torch.npu.NPUGraph() for i in (1, 2)}
    saved = []
    result = measure_sequence(
        torch, rounds, graphs, lambda: events.append('restore'),
        lambda entry: events.append(f'prepare{entry.source_round}'),
        lambda entry: events.append(f'check{entry.source_round}'),
        warmup=1, iterations=2, save=saved.append)
    assert events.count('restore') == 3 and events.count('replay') == 6
    assert result['sequence_graph_sum']['samples_us'] == [24., 24.]
    assert result['rounds'][1]['samples_us'] == [12., 12.]
    assert len(saved) == 8  # Before/after validation, only measured rounds.
    for i, event in enumerate(events):
        if event == 'check1':
            assert events[i+1] == 'prepare2'


def test_sequence_persists_timing_before_unsafe_validation_failure():
    from performance import measure_sequence
    torch, events = fake_device()
    saved = []
    def unsafe(entry):
        raise RuntimeError('unknown device completion')
    with pytest.raises(RuntimeError, match='unknown device completion'):
        measure_sequence(torch, [SimpleNamespace(source_round=1, scheduler_round=101)],
                         {1: torch.npu.NPUGraph()}, lambda: None, lambda entry: None,
                         unsafe, warmup=0, iterations=2, save=saved.append)
    assert len(saved) == 1 and saved[0]['elapsed_us'] == 12.
    assert saved[0]['correctness'] == 'NOT_CHECKED'
    assert events.count('replay') == 1


def test_bench_validation_is_bounded_independent_of_iterations():
    torch, events = fake_device()
    result = measure_graph(torch, lambda: None, lambda: events.append('restore'),
                           lambda: events.append('check'), warmup=4, iterations=50,
                           validate_each=False)
    assert result['timing']['samples'] == 50
    assert events.count('replay') == 54
    assert events.count('check') == 2  # Before the timing loop and after its last replay.
    assert events.count('restore') == 56
    assert result['correctness'] == 'PASS'


def test_plan_matches_per_query_worker_order_and_conflicts():
    batch = 4
    posts = np.zeros((batch, 4, 32), dtype=np.int32)
    for b in range(batch):
        for q in range(4):
            posts[b, q, 4] = b * 8192 + q * TOPK
    for b in range(batch):
        posts[b, 0, 0], posts[b, 0, 5] = 2, 1
    posts[1, 1, 0] = posts[1, 1, 5] = 1
    blocks = np.arange(batch * 8192, dtype=np.int32)
    offsets = blocks + 3
    destinations = blocks + 7
    scratch, counts, queries, all_dest = build_group_plan(posts, blocks, offsets, destinations, batch)
    assert counts.tolist() == [[2, 2], [2, 2], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    assert scratch[0, :4].tolist() == [0, 16384, 1, 16385]
    assert scratch[1, :4].tolist() == [8192, 24576, 8193, 24577]
    stride = batch * TOPK
    assert scratch[0, stride:stride+4].tolist() == [3, 16387, 4, 16388]
    assert len(queries[0]) == 8 and len(queries[1]) == 1 and len(all_dest) == 9
    posts[0, 0, 5] = 3
    with pytest.raises(AssertionError, match='invalid descriptor'):
        build_group_plan(posts, blocks, offsets, destinations, batch)
