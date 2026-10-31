import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from kvop.v9 import GraphFrameSession, SessionState


def fixture(**overrides):
    events = []
    stream = SimpleNamespace(device="npu:0", npu_stream=10)
    control = dict(
        valid=True, fail_end=False, current=stream, stream=stream, quarantined=[]
    )

    class Resource:
        pass

    resources = [Resource()]

    def end(stream):
        if control["fail_end"]:
            raise RuntimeError("end failed")
        events.append("end")

    runtime = SimpleNamespace(
        fused_graph_abi=lambda: 1,
        register_fused_graph_output=lambda: None,
        begin_fused_graph=lambda stream: events.append("begin"),
        end_fused_graph=end,
    )

    def quarantine(reason, error, keepalive):
        control["quarantined"].append(keepalive)
        events.append("quarantine")

    callbacks = dict(
        current_stream=lambda: control["current"],
        synchronize=lambda: events.append("sync"),
        check_completed=lambda: control["valid"],
        quarantine=quarantine,
        keepalive=resources,
    )
    callbacks.update(overrides)
    frame = GraphFrameSession(runtime, stream, **callbacks)
    return frame, events, control, resources


def test_graph_frame_orders_ownership_before_replay_and_checks_before_release():
    frame, events, _, _ = fixture()
    frame.begin(prepare=lambda: events.append("prepare"))
    events.append("replay")
    with pytest.raises(RuntimeError):
        frame.begin()
    with pytest.raises(RuntimeError):
        frame.close()
    frame.complete()
    assert events == ["prepare", "begin", "replay", "sync", "end"]
    assert frame.state is SessionState.ACCEPTED
    frame.begin()
    frame.complete()
    frame.close()


@pytest.mark.parametrize("failure", ["invalid", "end_error", "abort"])
def test_graph_uncertainty_preserves_resources_and_prevents_reuse(failure):
    frame, events, control, resources = fixture()
    frame.begin()
    if failure == "invalid":
        control["valid"] = False
    if failure == "end_error":
        control["fail_end"] = True
    with pytest.raises(RuntimeError, match="quarantined"):
        if failure == "abort":
            frame.abort(RuntimeError("model failed"))
        else:
            frame.complete()
    assert frame.state is SessionState.QUARANTINED
    assert frame.keepalive == tuple(resources)
    assert control["quarantined"] == [frame.keepalive]
    assert "quarantine" in events
    with pytest.raises(RuntimeError):
        frame.begin()
    with pytest.raises(RuntimeError):
        frame.close()


def test_wrong_stream_rejected_before_prepare_sync_check_or_release():
    frame, events, control, _ = fixture()
    wrong = SimpleNamespace(device="npu:0", npu_stream=11)
    control["current"] = wrong
    with pytest.raises(ValueError, match="acquired stream"):
        frame.begin(prepare=lambda: events.append("prepare"))
    assert events == [] and frame.state is SessionState.PREPARED
    control["current"] = control["stream"]
    frame.begin()
    control["current"] = wrong
    with pytest.raises(ValueError, match="acquired stream"):
        frame.complete()
    assert events == ["begin"] and frame.state is SessionState.IN_FLIGHT
    control["current"] = control["stream"]
    frame.complete()
    assert events == ["begin", "sync", "end"]


def test_preparation_cannot_switch_the_acquired_stream_before_begin():
    frame, events, control, _ = fixture()

    def prepare():
        control["current"] = SimpleNamespace(device="npu:0", npu_stream=11)

    with pytest.raises(RuntimeError, match="quarantined"):
        frame.begin(prepare=prepare)
    assert events == ["quarantine"]
    assert frame.state is SessionState.QUARANTINED


@pytest.mark.parametrize("result", [False, None, 1, np.bool_(True)])
def test_completion_requires_literal_host_attestation_for_dma_and_status(result):
    frame, events, _, _ = fixture(check_completed=lambda: result)
    frame.begin()
    with pytest.raises(RuntimeError, match="quarantined"):
        frame.complete()
    assert events == ["begin", "sync", "quarantine"]


@pytest.mark.parametrize(
    "callback",
    [
        "current_stream",
        "synchronize",
        "check_completed",
        "quarantine",
    ],
)
def test_graph_callbacks_must_be_explicit_callables(callback):
    with pytest.raises(ValueError, match="callbacks must be callable"):
        fixture(**{callback: None})


def test_resource_snapshot_survives_caller_clear_and_discarded_failed_frame():
    frame, events, control, resources = fixture()
    resource = weakref.ref(resources[0])
    resources.clear()
    assert resource() is not None and len(frame.keepalive) == 1
    frame.begin()
    with pytest.raises(RuntimeError, match="quarantined"):
        frame.abort(RuntimeError("model failed"))
    assert control["quarantined"][0] is frame.keepalive
    del frame
    gc.collect()
    assert resource() is control["quarantined"][0][0]
    assert events == ["begin", "quarantine"]


@pytest.mark.parametrize("failure", ["prepare", "begin", "synchronize", "check"])
def test_callback_failures_quarantine_without_releasing_or_losing_resources(failure):
    frame, events, control, resources = fixture()

    def fail(*args):
        raise RuntimeError("callback failed")

    if failure == "begin":
        frame.runtime.begin_fused_graph = fail
    if failure == "synchronize":
        frame.synchronize = fail
    if failure == "check":
        frame.check_completed = fail
    with pytest.raises(RuntimeError, match="quarantined"):
        frame.begin(prepare=fail if failure == "prepare" else None)
        frame.complete()
    assert "end" not in events and events[-1] == "quarantine"
    assert frame.state is SessionState.QUARANTINED
    assert control["quarantined"] == [tuple(resources)]


def test_quarantine_callback_error_does_not_make_frame_reusable():
    retained = []

    def quarantine(reason, error, keepalive):
        retained.append(keepalive)
        raise RuntimeError("retained by runtime")

    frame, _, _, resources = fixture(quarantine=quarantine)
    frame.begin()
    with pytest.raises(RuntimeError, match="retained by runtime"):
        frame.abort(RuntimeError("replay failed"))
    assert frame.state is SessionState.QUARANTINED
    assert retained == [tuple(resources)]
    with pytest.raises(RuntimeError):
        frame.close()
