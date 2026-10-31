from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import numpy as np
import pytest

from kvip.testing.owner import DeferredTransport, StubQueueOwner
from kvip.v9 import PreparedSession, SessionState, Status
from kvip.v9.access import OUTPUT_FIELDS


def session_fixture(*, prepared=True):
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
    session = PreparedSession(launcher) if prepared else launcher
    return session, owner, transport, current, status, control


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
    assert not owner.lease_active and owner.session_binding is None


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


def test_owner_cannot_report_completion_without_clearing_in_flight():
    session, owner, transport, _, status, _ = session_fixture()
    session.launcher.completion = lambda: status.copy()
    session.enqueue()
    with pytest.raises(RuntimeError, match="quarantined"):
        session.complete()
    assert owner.in_flight and transport.pending
    assert owner.session_binding is session
    assert any(item is session for item in owner.keepalive)


def test_owner_close_must_confirm_release_before_session_becomes_closed():
    session, owner, _, _, _, _ = session_fixture()
    owner.close = lambda: None
    with pytest.raises(RuntimeError, match="quarantined"):
        session.close()
    assert session.state is SessionState.QUARANTINED
    assert owner.lease_active and owner.in_flight
    assert owner.session_binding is session
    assert any(item is session for item in owner.keepalive)


def test_one_active_lease_cannot_bind_two_sessions():
    session, owner, transport, _, _, _ = session_fixture()
    with pytest.raises(ValueError, match="already bound"):
        PreparedSession(session.launcher)
    assert owner.session_binding is session
    assert not owner.in_flight and not transport.pending
    session.close()
    with pytest.raises(ValueError, match="active, open lease"):
        PreparedSession(session.launcher)
    assert owner.closed and not owner.in_flight and not transport.pending


def test_concurrent_preparation_has_exactly_one_lease_binding():
    launcher, owner, _, _, _, _ = session_fixture(prepared=False)
    start = Barrier(2)

    def prepare():
        start.wait(timeout=5)
        try:
            return PreparedSession(launcher)
        except ValueError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: prepare(), range(2)))
    sessions = [item for item in results if isinstance(item, PreparedSession)]
    failures = [item for item in results if isinstance(item, ValueError)]
    assert len(sessions) == len(failures) == 1
    assert "already bound" in str(failures[0])
    assert owner.session_binding is sessions[0]
    sessions[0].close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("closed", True),
        ("lease_active", False),
        ("lease_generation", 0),
        ("lease_generation", True),
        ("lease_generation", 1.0),
        ("lease_stream", None),
    ],
)
def test_preparation_requires_explicit_valid_lease_metadata(field, value):
    launcher, owner, transport, _, _, _ = session_fixture(prepared=False)
    setattr(owner, field, value)
    with pytest.raises(ValueError):
        PreparedSession(launcher)
    assert owner.session_binding is None and not owner.in_flight
    assert not transport.pending


@pytest.mark.parametrize(
    "field",
    [
        "closed",
        "lease_active",
        "lease_generation",
        "lease_stream",
        "session_lock",
        "session_binding",
    ],
)
def test_preparation_does_not_infer_missing_owner_metadata(field):
    launcher, owner, _, _, _, _ = session_fixture(prepared=False)
    delattr(owner, field)
    with pytest.raises(ValueError, match="QueueOwnerProtocol"):
        PreparedSession(launcher)


def test_explicit_stream_must_match_acquired_lease_stream():
    launcher, owner, _, current, _, _ = session_fixture(prepared=False)
    wrong = SimpleNamespace(device="npu:0", npu_stream=11)
    with pytest.raises(ValueError, match="acquired lease stream"):
        PreparedSession(launcher, stream=wrong)
    assert owner.session_binding is None
    # A separate Python wrapper for the same native stream remains valid.
    same = SimpleNamespace(device=current[0].device, npu_stream=current[0].npu_stream)
    session = PreparedSession(launcher, stream=same)
    session.enqueue()
    session.complete()
    session.close()


@pytest.mark.parametrize("accepted", [False, True])
def test_reacquisition_invalidates_old_session_without_releasing_new_lease(accepted):
    old, owner, transport, current, _, _ = session_fixture()
    if accepted:
        old.enqueue()
        old.complete()
    generation = owner.lease_generation
    owner.release()
    owner.acquire(current[0])
    assert owner.lease_generation == generation + 1
    replacement = PreparedSession(old.launcher)
    for action in (old.enqueue, old.close):
        with pytest.raises(RuntimeError, match="generation has changed"):
            action()
    assert owner.lease_active and not owner.closed
    assert owner.session_binding is replacement and not transport.pending
    replacement.enqueue()
    replacement.complete()
    replacement.close()


def test_closed_session_close_stays_idempotent_after_owner_replacement():
    old, owner, _, _, _, _ = session_fixture()
    old.close()
    replacement_owner = StubQueueOwner(owner.output, owner.lease_stream)
    old.launcher.owner = replacement_owner
    replacement = PreparedSession(old.launcher)
    old.close()
    assert not replacement_owner.closed and replacement_owner.lease_active
    assert replacement_owner.session_binding is replacement
    replacement.close()


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("close", "closed or inactive"),
        ("release", "closed or inactive"),
        ("binding", "binding"),
        ("stream", "acquired stream has changed"),
    ],
)
def test_external_owner_changes_are_rejected_before_submission(mutation, match):
    session, owner, transport, _, _, _ = session_fixture()
    if mutation == "close":
        owner.close()
    elif mutation == "release":
        owner.release()
    elif mutation == "binding":
        owner.session_binding = None
    else:
        owner.lease_stream = SimpleNamespace(device="npu:0", npu_stream=99)
    with pytest.raises(RuntimeError, match=match):
        session.enqueue()
    assert not transport.pending and not owner.in_flight


@pytest.mark.parametrize("failure", ["fail_launch", "fail_completion"])
def test_quarantine_keeps_binding_and_prohibits_external_release_or_reacquire(failure):
    session, owner, transport, current, _, control = session_fixture()
    retained = tuple(owner.keepalive)
    control[failure] = True
    with pytest.raises(RuntimeError, match="quarantined"):
        session.enqueue()
        session.complete()
    assert owner.session_binding is session and owner.lease_active
    assert tuple(owner.keepalive) == retained and transport.pending
    for action in (owner.release, owner.close, lambda: owner.acquire(current[0])):
        with pytest.raises(RuntimeError):
            action()
