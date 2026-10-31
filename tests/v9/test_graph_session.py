from types import SimpleNamespace

import pytest
from kvop.v9 import GraphFrameSession, SessionState


def fixture():
    events = []
    control = dict(valid=True, fail_end=False)
    resources = [object()]

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
    frame = GraphFrameSession(
        runtime,
        object(),
        synchronize=lambda: events.append("sync"),
        check_completed=lambda: control["valid"],
        quarantine=lambda reason, error: events.append("quarantine"),
        keepalive=resources,
    )
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
    assert frame.state is SessionState.QUARANTINED and frame.keepalive is resources
    assert "quarantine" in events
    with pytest.raises(RuntimeError):
        frame.begin()
    with pytest.raises(RuntimeError):
        frame.close()
