# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for timing scope and input preparation, not NPU timings."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kvip.diagnostics import call_preparation


@pytest.fixture
def call_mocks(monkeypatch):
    events = []
    controls = dict(fail=None)
    state = dict(uploaded=False, launched=False, completed=False, verified=0)

    def operation(name):
        events.append(name)
        if controls["fail"] == name:
            raise RuntimeError(f"{name} failed")

    class Preparation:
        metadata = {"candidate_bytes": 128, "dense_bytes": 1024}

        def poison(self):
            operation("poison")
            state.update(uploaded=False, launched=False, completed=False)

        def prepare_cpu(self):
            operation("prepare_cpu")

        def upload(self):
            operation("upload")
            state["uploaded"] = True

    def sync():
        operation("sync")
        if state["launched"]:
            if controls["fail"] == "completion":
                raise RuntimeError("completion failed")
            state["completed"] = True

    def replay():
        assert state["uploaded"]
        operation("replay")
        state["launched"] = True

    def verify():
        assert state["completed"]
        operation("verify")
        state["verified"] += 1
        return {"iteration": state["verified"], "bytes": 656}

    ticks = iter((1000, 2000, 5000, 10000, 13000, 19000, 30000, 35000, 46000))

    def clock():
        events.append("clock")
        return next(ticks)

    monkeypatch.setattr(call_preparation, "perf_counter_ns", clock)
    preparation = Preparation()
    bundle = SimpleNamespace(
        candidate_preparation=preparation,
        reset=lambda: operation("reset"),
        verify=verify,
    )
    return SimpleNamespace(
        events=events,
        controls=controls,
        preparation=preparation,
        bundle=bundle,
        args=SimpleNamespace(warmup=1, iterations=2, stage_iterations=0),
        stream=SimpleNamespace(synchronize=sync),
        replay=replay,
        retained={"bundle": bundle, "graphs": [object()]},
    )


def test_wall_timer_includes_cpu_upload_graph_completion_and_excludes_reset_oracle(call_mocks):
    mock = call_mocks
    report = call_preparation.measure_candidate_call(
        None, mock.stream, mock.bundle, mock.args, mock.replay, mock.retained
    )
    one = ["reset", "poison", "sync", "clock", "prepare_cpu", "clock", "upload", "replay", "sync", "clock", "verify"]
    assert mock.events == one * 3
    assert report["wall"]["samples_us"] == [9.0, 16.0]
    assert report["cpu_prepare"]["samples_us"] == [3.0, 5.0]
    assert report["reads_samples"] == [{"iteration": 2, "bytes": 656}, {"iteration": 3, "bytes": 656}]
    assert not report.get("h2d_diagnostic")


@pytest.mark.parametrize("failure", ("prepare_cpu", "upload", "replay", "completion", "verify"))
def test_failed_preparation_or_completion_keeps_owner_bundle_and_graphs(call_mocks, failure):
    mock = call_mocks
    mock.controls["fail"] = failure
    graphs = mock.retained["graphs"]
    with pytest.raises(RuntimeError, match=failure):
        call_preparation.measure_candidate_call(None, mock.stream, mock.bundle, mock.args, mock.replay, mock.retained)
    assert mock.retained["bundle"] is mock.bundle
    assert mock.bundle.candidate_preparation is mock.preparation
    assert mock.retained["graphs"] is graphs and len(graphs) == 1
    assert "close" not in mock.events and "graph_reset" not in mock.events


@pytest.mark.parametrize("fail_end_record", (False, True))
def test_h2d_events_exclude_cpu_and_graph_keep_events_alive_before_use(call_mocks, fail_end_record):
    mock = call_mocks
    mock.args.stage_iterations = 2
    previous = object()
    mock.retained["events"] = [previous]
    created = []

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing is True
            self.index = len(created)
            created.append(self)
            mock.events.append(f"event_create{self.index}")

        def record(self, stream):
            assert stream is mock.stream
            assert mock.retained["events"] == [previous, *created]
            assert len(created) == 2  # Both owners retained before either event is submitted.
            mock.events.append(f"event_record{self.index}")
            if fail_end_record and self.index == 1:
                raise RuntimeError("end record failed")

        def elapsed_time(self, end):
            assert self is created[0] and end is created[1]
            mock.events.append("elapsed")
            return 0.004

    fake_torch = SimpleNamespace(npu=SimpleNamespace(Event=Event))
    if fail_end_record:
        with pytest.raises(RuntimeError, match="end record failed"):
            call_preparation.measure_candidate_call(
                fake_torch, mock.stream, mock.bundle, mock.args, mock.replay, mock.retained
            )
        assert mock.events[-4:] == ["prepare_cpu", "event_record0", "upload", "event_record1"]
        assert mock.retained["bundle"] is mock.bundle and mock.retained["graphs"]
    else:
        report = call_preparation.measure_candidate_call(
            fake_torch, mock.stream, mock.bundle, mock.args, mock.replay, mock.retained
        )
        after_events = mock.events.index("event_create1") + 1
        one = [
            "reset",
            "poison",
            "sync",
            "prepare_cpu",
            "event_record0",
            "upload",
            "event_record1",
            "replay",
            "sync",
            "elapsed",
            "verify",
        ]
        assert mock.events[after_events:] == one * 3
        assert report["h2d_diagnostic"]["samples_us"] == [4.0, 4.0]
        assert report["wall"]["samples_us"] == [9.0, 16.0]
        assert len(report["reads_samples"]) == 2
        assert mock.events.count("clock") == 9  # No host-clock samples are mixed into the device diagnostic.
    assert mock.retained["events"] == [previous, *created]
    assert "close" not in mock.events and "graph_reset" not in mock.events


@pytest.fixture
def candidate_inputs():
    batch, resident, count = 3, 4096, 2048
    candidates = np.stack(
        (
            np.roll(np.arange(count, dtype=np.int32), 17),
            np.arange(count, 2 * count, dtype=np.int32),
            np.roll(np.arange(resident, dtype=np.int32)[::2], 31),
        )
    )
    ends = np.array([[4000, 4001, 4002, 4003], [0, 0, 0, 0], [4060, 4061, 4062, 4063]], dtype=np.int32)
    expected = np.full((batch, resident), -1, dtype=np.int32)
    for row in (0, 2):
        expected[row, candidates[row]] = np.arange(count)
    # A derived replay view may retain a legal inverse for its inert row.
    # Only active rows define the measured preparation's semantic contract.
    captured = expected.copy()
    captured[1, candidates[1]] = np.arange(count)
    arrays = {"pre.lru_slots": candidates, "pre.query_ends": ends, "pre.reserved_hot": captured}
    reads, allocations, copies = [], [], []

    def array(name):
        reads.append(name)
        return arrays[name]

    class CpuTorch:
        int32 = torch.int32
        locked = False

        def empty(self, shape, **kwargs):
            assert not self.locked, "Allocation inside measured preparation"
            assert kwargs["pin_memory"] is True and kwargs["device"] == "cpu"
            allocations.append(tuple(shape))
            return torch.empty(shape, **dict(kwargs, pin_memory=False))

    class Device:
        def __init__(self, name, shape):
            self.name = name
            self.values = torch.full(shape, -9, dtype=torch.int32)

        def fill_(self, value):
            self.values.fill_(value)

        def copy_(self, source, *, non_blocking):
            assert non_blocking is True
            copies.append(self.name)
            self.values.copy_(source)

    return SimpleNamespace(
        torch=CpuTorch(),
        archive=SimpleNamespace(array=array, manifest={"geometry": {"resident": resident}}),
        rotating=SimpleNamespace(
            lru_slots=Device("lru", candidates.shape), reserved_hot=Device("reserved", expected.shape)
        ),
        arrays=arrays,
        expected=expected,
        reads=reads,
        allocations=allocations,
        copies=copies,
    )


@pytest.mark.parametrize("compact", (False, True))
def test_candidate_preparation_stages_only_required_payload_and_reuses_preallocated_buffers(
    candidate_inputs, compact, monkeypatch
):
    inputs = candidate_inputs
    bundle = SimpleNamespace(rotating=inputs.rotating, tuning={"compact_reserved": compact})
    preparation = call_preparation.CandidatePreparation(inputs.torch, inputs.archive, bundle)
    assert inputs.allocations == ([(3, 2048)] if compact else [(3, 2048), (3, 4096)])
    assert preparation.metadata == dict(
        candidate_h2d_bytes=3 * 2048 * 4,
        dense_protection_h2d_bytes=0 if compact else 3 * 4096 * 4,
        dense_protection_generated_on_device=compact,
        batch_capacity=3,
        active_rows=2,
        group_count=1,
        includes_service_guard_row=False,
    )
    host = preparation.lru_host
    dense_host = preparation.reserved_host
    source_candidates = inputs.arrays["pre.lru_slots"].copy()

    def forbidden(*args, **kwargs):
        raise AssertionError("Archive IO, validation or allocation inside measured preparation")

    inputs.torch.locked = True
    for _ in range(2):
        preparation.poison()
        assert torch.all(inputs.rotating.lru_slots.values == -2)
        assert torch.all(inputs.rotating.reserved_hot.values == -2)
        with monkeypatch.context() as patch:
            patch.setattr(inputs.archive, "array", forbidden)
            patch.setattr(call_preparation, "reserved_hot_reference", forbidden)
            for name in ("empty", "full", "arange", "unique"):
                patch.setattr(call_preparation.np, name, forbidden)
            preparation.prepare_cpu()
            preparation.upload()
        assert np.array_equal(inputs.rotating.lru_slots.values.numpy(), source_candidates)
        if compact:
            assert torch.all(inputs.rotating.reserved_hot.values == -2)
        else:
            assert np.array_equal(inputs.rotating.reserved_hot.values.numpy(), inputs.expected)
        assert preparation.lru_host is host and preparation.reserved_host is dense_host
    assert inputs.copies == (["lru"] if compact else ["lru", "reserved"]) * 2
    assert np.array_equal(inputs.arrays["pre.lru_slots"], source_candidates)


@pytest.mark.parametrize("invalid", ("duplicate", "out_of_bounds", "inverse"))
def test_candidate_validation_fails_before_allocating_pinned_buffers(candidate_inputs, invalid):
    inputs = candidate_inputs
    if invalid == "duplicate":
        inputs.arrays["pre.lru_slots"][0, 0] = inputs.arrays["pre.lru_slots"][0, 1]
    elif invalid == "out_of_bounds":
        inputs.arrays["pre.lru_slots"][0, 0] = 4096
    else:
        inputs.arrays["pre.reserved_hot"][0].fill(-1)
    with pytest.raises(ValueError):
        call_preparation.CandidatePreparation(
            inputs.torch, inputs.archive, SimpleNamespace(rotating=inputs.rotating, tuning={})
        )
    assert not inputs.allocations
