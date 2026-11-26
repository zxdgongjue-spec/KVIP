# SPDX-License-Identifier: Apache-2.0
"""Check that all ten graph invocations see independent, correctly restored state."""
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

import execution
from fixed_q4_packed_worker import FrozenPackedRound
from large_batch_input import FrozenRound
from reporting import timing_summary


class Buffer:
    def __init__(self, value=5):
        self.value = value

    def data_ptr(self):
        return id(self)

    def copy_(self, source):
        self.value = source.value

    def clone(self):
        return Buffer(self.value)


class State:
    def __init__(self, inputs, torch, runtime):
        self.initial, self.torch, self.device = inputs, torch, 0
        self.executions = 0
        fields = ("forward", "reverse", "cursors", "output", "preferred", "bitmap", "destinations", "counts")
        self.native = SimpleNamespace(**{name: Buffer() for name in fields},
                                     owner=SimpleNamespace(session=SimpleNamespace(runtime=runtime)))

    def reset(self):
        self.native.output.value = 5

    def forward(self):
        if self.torch.npu.graph_capturing:
            self.torch.npu.captured.append(self)
        else:
            assert self.native.output.value == 5
            self.native.output.value = 6
            self.executions += 1


class FakeNPU:
    def __init__(self):
        self.captured = []
        self.graph_capturing = False

    def current_stream(self, device):
        return SimpleNamespace(synchronize=lambda: None)

    def Stream(self):
        return self.current_stream(0)

    def stream(self, capture):
        return nullcontext()

    def NPUGraph(self):
        return SimpleNamespace(replay=lambda: [s.forward() for s in self.captured], reset=lambda: None)

    def graph(self, graph, stream):
        @contextmanager
        def capture():
            self.graph_capturing = True
            try:
                yield
            finally:
                self.graph_capturing = False
        return capture()


def test_packed_graph_restores_all_states_and_checks_after_the_same_timed_replay(monkeypatch, tmp_path):
    npu, runtime = FakeNPU(), object()
    torch = SimpleNamespace(npu=npu)
    inputs = SimpleNamespace(source_round=1)
    states = [State(inputs, torch, runtime) for _ in range(10)]
    frozen = FrozenPackedRound(states, inputs, None, False)
    measured = []
    checks = []
    def validate(state, actual, index):
        assert measured == [10] and state.native.output.value == 6
        assert state is states[index] and actual is inputs
        checks.append(index)
    frozen.validate = validate
    resources = SimpleNamespace(close=lambda: None)
    @contextmanager
    def prepare_get(runtime, stream, forward):
        forward()
        yield resources
    monkeypatch.setattr(execution, "capture_native_get", prepare_get)

    def profile(torch, graph, *, calls, **kwargs):
        assert npu.captured == states
        assert all(s.native.output.value == 5 for s in states)
        measured.append(calls)
        graph.replay()
        graph.completed()
        times = dict(timing_summary([1] * calls), mean_us=1.2, stage_mean_us={'li': 0.4},
                     inter_kernel_gap_mean_us=0, overlap_mean_us=0, outside_invocations_mean_us=0.2)
        return dict(timing=times, samples=[dict(us=1)] * calls, accounting={}, trace="fake")

    result = execution.measure_replay_batch(torch, states, FrozenRound(inputs, []),
        SimpleNamespace(mode="bench", warmup=4, transport="urma"), tmp_path, fused=False,
        profile=profile, frozen=frozen)
    assert checks == list(range(10))
    assert all(s.executions == 7 for s in states)  # Registration, GET prepare, four warmups, one timed graph.
    assert result['correctness'] == 'PASS' and result['all_rounds']['graph_calls'] == 10
    states[0].native.output = states[1].native.output
    with pytest.raises(ValueError, match="share mutable"):
        FrozenPackedRound(states, inputs, None, False)
