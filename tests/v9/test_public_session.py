from types import SimpleNamespace

import numpy as np
import pytest
from kvop.testing.owner import DeferredTransport, StubQueueOwner
from kvop.v9 import PreparedSession, SessionState, Status
from kvop.v9.access import OUTPUT_FIELDS


def session_fixture():
    stream = SimpleNamespace(device="npu:0", npu_stream=10)
    current = [stream]
    output = np.zeros((2, 656), np.uint8)
    source = np.full_like(output, 7)
    transport = DeferredTransport(source, output)
    owner = StubQueueOwner(output, stream, copier=transport)
    status = np.array([Status.OK], np.int32)
    control = dict(fail_launch=False, fail_completion=False)

    def launch():
        transport.submit(1, 0)
        if control["fail_launch"]:
            raise RuntimeError("launch failed")

    def completion():
        transport.complete(fail=control["fail_completion"])
        owner.mark_completed(completion_verified=True)
        return status.copy()

    launcher = SimpleNamespace(
        owner=owner,
        device="npu:0",
        outputs=dict.fromkeys(OUTPUT_FIELDS, output),
        torch=SimpleNamespace(npu=SimpleNamespace(current_stream=lambda _: current[0])),
        launch=launch,
        completion=completion,
    )
    return PreparedSession(launcher), owner, transport, current, status, control


def test_deferred_lifetime_and_output_without_extra_sync_or_reordering():
    session, owner, transport, _, _, _ = session_fixture()
    session.enqueue()
    assert owner.in_flight and np.all(owner.output == 0)
    with pytest.raises(RuntimeError, match="cannot enqueue"):
        session.enqueue()
    with pytest.raises(RuntimeError, match="cannot release"):
        session.close()
    result = session.complete()
    assert result.drained and result.success
    assert session.state is SessionState.ACCEPTED
    np.testing.assert_array_equal(owner.output[0], transport.source[1])
    np.testing.assert_array_equal(owner.output[1], 0)
    session.enqueue()
    session.complete()
    session.close()
    session.close()
    assert owner.closed


def test_wrong_stream_rejected_without_submitting_or_hiding_inflight_state():
    session, owner, transport, current, _, _ = session_fixture()
    original = current[0]
    current[0] = SimpleNamespace(device="npu:0", npu_stream=11)
    with pytest.raises(ValueError, match="acquired stream"):
        session.enqueue()
    assert not owner.in_flight and not transport.pending
    current[0] = original
    session.enqueue()
    current[0] = SimpleNamespace(device="npu:1", npu_stream=10)
    with pytest.raises(ValueError, match="acquired stream"):
        session.complete()
    assert owner.in_flight
    current[0] = original
    session.complete()
    session.close()


@pytest.mark.parametrize("failure", ["fail_launch", "fail_completion"])
def test_unknown_completion_keeps_storage_and_refuses_reuse_or_close(failure):
    session, owner, transport, _, _, control = session_fixture()
    control[failure] = True
    with pytest.raises(RuntimeError, match="quarantined"):
        session.enqueue()
        session.complete()
    assert session.state is SessionState.QUARANTINED
    assert owner.in_flight and owner.keepalive and transport.pending
    with pytest.raises(RuntimeError):
        session.close()
    with pytest.raises(RuntimeError):
        session.enqueue()


def test_drained_capacity_error_is_not_accepted_but_can_release():
    session, owner, _, _, status, _ = session_fixture()
    status[0] = Status.CAPACITY
    session.enqueue()
    result = session.complete()
    assert result.drained and not result.success and not owner.in_flight
    assert session.state is SessionState.RETIRED
    with pytest.raises(RuntimeError):
        session.enqueue()
    session.close()


def test_release_failure_quarantines_without_discarding_owner():
    session, owner, _, _, _, _ = session_fixture()
    owner.fail_release = True
    with pytest.raises(RuntimeError, match="quarantined"):
        session.close()
    assert session.state is SessionState.QUARANTINED and owner.keepalive
