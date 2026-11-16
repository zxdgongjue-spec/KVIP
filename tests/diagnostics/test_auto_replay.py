# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import pytest

from kvip.diagnostics import auto_replay
from kvip.diagnostics.auto_replay import AutoReplay, RowPermutation, requests_for, VALID_FEEDBACK
from kvip.diagnostics.pipeline_options import early_threshold_skipped, options_for, q_workers_for, validate_options


def archive():
    metadata = np.zeros((5, 8), np.int64)
    metadata[0] = metadata[2] = np.arange(8)
    metadata[1] = metadata[3] = 1
    metadata[4].view(np.int32)[:2] = [2, 1]
    arrays = {"pre.metadata": metadata, "pre.query_ends": np.arange(100, 108, dtype=np.int32),
              "pre.query": np.arange(8 * 3 * 4, dtype=np.uint8).reshape(8, 3, 4),
              "pre.weights": np.arange(24, dtype=np.uint16).reshape(8, 3),
              "pre.query_scale": np.arange(24, dtype=np.float32).reshape(8, 3),
              "pre.threshold_bounds": np.array([[5, 9], [10, 15]], np.int32),
              "post.score_keys": np.arange(8 * 10, dtype=np.uint16).reshape(2, 4, 10),
              "post.logical_output": np.arange(8 * 5, dtype=np.int32).reshape(8, 5)}
    value = SimpleNamespace(manifest={"geometry": {"batch": 2}}, directory="unused", array=arrays.__getitem__)
    return value, arrays


def test_repacking_preserves_tnd_query_mapping_and_owner_state():
    original, arrays = archive()
    permuted = RowPermutation(original, [1, 0])
    metadata = permuted.array("pre.metadata")
    np.testing.assert_array_equal(metadata[4].view(np.int32)[:2], [1, 2])
    np.testing.assert_array_equal(metadata[2], [4, 5, 6, 7, 0, 1, 2, 3])
    for name in ("pre.query", "pre.weights", "pre.query_scale", "pre.query_ends"):
        np.testing.assert_array_equal(permuted.array(name)[metadata[2]], arrays[name])
    np.testing.assert_array_equal(permuted.array("post.score_keys")[::-1], arrays["post.score_keys"])
    np.testing.assert_array_equal(permuted.array("pre.threshold_bounds"), [[10, 15], [5, 9]])
    assert permuted.array("post.logical_output") is arrays["post.logical_output"]
    np.testing.assert_array_equal(requests_for(permuted, generation=3, step=4), [[1, 3, 4, 4], [2, 3, 4, 4]])
    with pytest.raises(ValueError, match="complete batch"):
        RowPermutation(original, [0, 0])
    arrays["pre.metadata"][2] = [5, 4, 7, 6, 1, 0, 3, 2]
    np.testing.assert_array_equal(
        permuted.array("pre.query")[permuted.array("pre.metadata")[2]],
        arrays["pre.query"][arrays["pre.metadata"][2]])


def test_auto_options_are_explicit_and_runtime_mode_is_not_guessed_from_static_quota():
    for name in ("auto", "auto-end", "auto-tail", "auto-tail-end"):
        values = validate_options(options_for(name))
        assert values["auto_route"] and q_workers_for(name) == 4
        with pytest.raises(ValueError, match="auto_route requires"):
            validate_options(values | {"low_miss_skip": False})
        assert not early_threshold_skipped(values | {"admission_controls": [0, 64, 0, 0]})
        assert early_threshold_skipped(values | {"auto_low_active": True})


def test_low_tail_options_keep_identical_query_workers_and_manual_admission():
    from kvip.diagnostics.pipeline_options import admission_for

    values = validate_options(options_for("light-low-tail"))
    assert values["low_miss_tail"] and q_workers_for("light-low-tail") == 4
    assert admission_for("light-low-tail") == admission_for("light-low")
    with pytest.raises(ValueError, match="low_miss_tail requires"):
        validate_options(values | {"low_miss_skip": False})
    for name in ("light-high-tail", "light-high-tail-end"):
        high = validate_options(options_for(name))
        assert high == values
        assert q_workers_for(name) == 4
        assert admission_for(name) == admission_for("light-high")


class Tensor:
    def __init__(self, shape):
        self.value = np.zeros(shape, np.int64)

    def data_ptr(self):
        return self.value.ctypes.data


def driver(monkeypatch):
    state = SimpleNamespace(owners=3, config=Tensor((16,)), requests=Tensor((2, 4)),
                            feedback=Tensor((3, 16)), pending=Tensor((2, 16)))
    state.set_learning = lambda value: state.config.value.__setitem__(8, int(value))
    state.set_requests = lambda value: state.requests.value.__setitem__(slice(None), value)
    rotating = SimpleNamespace(auto_route=state, admission_controls=Tensor((2, 4)))
    rotating.admission_controls.value[:] = [512, 64, 0, 0]
    bundle = SimpleNamespace(rotating=rotating, tuning=options_for("auto"))
    monkeypatch.setattr(auto_replay, "cpu_array", lambda torch, tensor: tensor.value)
    value = AutoReplay(None, bundle, archive()[0])
    return value, state, bundle


def complete(state, bundle, *, learn, low=False):
    bundle.rotating.admission_controls.value[:] = [0 if low else 512, 64, 0, 0]
    state.pending.value.fill(0)
    for row, (owner, generation, step, q) in enumerate(state.requests.value):
        state.pending.value[row, :7] = [owner, generation, step, learn, int(not low), 0, q]
        state.pending.value[row, 9] = row + 3
        if learn:
            state.feedback.value[owner, :8] = [generation, step, row + 3, q, 1000, 656, int(not low), VALID_FEEDBACK]


def test_only_device_completion_updates_feedback_and_duplicates_cannot_relearn(monkeypatch):
    value, state, bundle = driver(monkeypatch)
    complete(state, bundle, learn=False)
    assert value.observe()["learned_owners"] == []
    assert not np.any(state.feedback.value)
    value.begin(step=1, learning=True)
    assert not np.any(state.feedback.value)  # Host preparation never seeds an oracle count.
    complete(state, bundle, learn=True)
    assert value.observe()["mode"] == "high"
    first = state.feedback.value.copy()
    value.begin(step=2, learning=True)
    np.testing.assert_array_equal(state.feedback.value, first)
    complete(state, bundle, learn=True, low=True)
    assert value.observe()["mode"] == "low"
    assert value.record(dict(prefetch=0, final=7, useful_transfers=7))["necessary_get"] == 7
    with pytest.raises(AssertionError, match="Get count"):
        value.record(dict(prefetch=0, final=7, useful_transfers=8))
    value.begin(step=2, learning=True)
    complete(state, bundle, learn=False)
    assert value.observe()["learned_owners"] == []
    state.feedback.value[1, 1] += 1
    with pytest.raises(AssertionError, match="Rejected or duplicate"):
        value.observe()


def test_capture_warmup_cannot_learn_and_control_addresses_cannot_change(monkeypatch):
    value, state, bundle = driver(monkeypatch)
    complete(state, bundle, learn=True)
    with pytest.raises(AssertionError, match="Capture/warmup"):
        value.observe()
    state.requests = Tensor((2, 4))
    with pytest.raises(AssertionError, match="captured state addresses"):
        value.begin(step=1, learning=True)


def test_prepare_never_reads_device_feedback_or_selects_a_mode(monkeypatch):
    value, state, bundle = driver(monkeypatch)
    previous = state.feedback.value.copy()
    monkeypatch.setattr(auto_replay, "cpu_array", lambda *args: pytest.fail("Preparation read the device"))
    value.begin(step=1, learning=True)
    np.testing.assert_array_equal(state.feedback.value, previous)
    np.testing.assert_array_equal(bundle.rotating.admission_controls.value, [[512, 64, 0, 0]] * 2)


def test_diagnostics_advance_measured_feedback_before_lifecycle_without_timing_them(monkeypatch):
    from kvip.diagnostics import replay_measure, native_profile

    log = []
    config = np.zeros(16, np.int64)
    config[5:7] = (128, 192)

    class Automatic:
        state = SimpleNamespace(config=config)
        step = 0
        learning = False

        def begin(self, *, step, learning):
            self.step, self.learning = step, learning
            log.append(("begin", step, learning))

        def record(self, reads):
            return dict(mode="high" if self.step == 1 else "low", observed_prefetch=0)

    automatic = Automatic()

    class Forward:
        batch, q_workers, resolver_cores = 2, 4, 3
        times = SimpleNamespace(zero_=lambda: None)
        diagnostics = False

        def __call__(self):
            pass

        def set_options(self, *, diagnostics):
            self.diagnostics = diagnostics

    forward = Forward()

    class Graph:
        def capture_begin(self, stream):
            pass

        def capture_end(self):
            pass

        def replay(self, stream):
            log.append(("replay", automatic.step, automatic.learning, forward.diagnostics))

        def reset(self):
            log.append("graph_reset")

    class Event:
        def __init__(self, **unused):
            pass

        def record(self, stream):
            assert not forward.diagnostics

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return [0.0, 0.010, 0.020, 0.030][automatic.step]

    def lifecycle(torch, stream, bundle, replay, last_step):
        assert last_step == 5
        assert log[-1] == ("replay", 5, True, True)
        log.append("lifecycle")
        return {"status": "pass"}

    monkeypatch.setattr(replay_measure, "cpu_array", lambda torch, value: config if value is config else None)
    monkeypatch.setattr(replay_measure, "check_auto_route", lifecycle)
    monkeypatch.setattr(native_profile, "decode_profile", lambda *args, **kwargs: {})
    bundle = SimpleNamespace(reset=lambda: None, verify=lambda: {}, read_counts=lambda: {},
                             forward=forward, lease=None, auto_replay=automatic)
    args = SimpleNamespace(mode="fused", transport="urma", warmup=0, iterations=3,
                           diagnostic_replays=2, check_auto_route=True)
    result = replay_measure.measure(SimpleNamespace(npu=SimpleNamespace(Event=Event)),
                                    SimpleNamespace(AclGraph=Graph),
                                    SimpleNamespace(synchronize=lambda: None), bundle, args, {})
    assert result["timing"]["samples_us"] == [10, 20, 30]
    assert result["timing"]["p95_us"] == 29
    assert result["auto_route"]["steady_by_mode"]["low"]["samples_us"] == [20, 30]
    assert result["auto_route"]["steady_timing"]["p50_us"] == 25
    assert [row["auto_route"]["mode"] for row in result["native_diagnostics"]["samples"]] == ["low", "low"]
    assert result["native_diagnostics"]["included_in_timing"] is False
    assert log[-2:] == ["lifecycle", "graph_reset"]
