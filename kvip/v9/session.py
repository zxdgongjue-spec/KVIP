"""Explicit eager lifecycle around the compatible low-level ABI13 launcher.

Graph capture/replay continues to use its frame-level owner and device guard;
an eager session is deliberately not captured as a Python state machine.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from threading import RLock
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .access import OUTPUT_FIELDS
from .contract import Status
from .transport import validate_graph_transport


@runtime_checkable
class QueueOwnerProtocol(Protocol):
    """One explicitly acquired lease, serialized by an owner-local reentrant lock.

    Acquisition increments lease_generation and publishes lease_stream/active;
    release clears active. Both transitions use session_lock. An active lease
    has at most one session_binding. quarantine must retain all asynchronous
    resources even if the consumer discards its session after the exception.
    These requirements apply to PreparedSession, not the low-level launcher.
    """

    output: Any
    device: Any
    keepalive: list
    in_flight: bool
    abi_sizes: dict
    core_status: Any
    closed: bool
    lease_active: bool
    lease_generation: int
    lease_stream: Any
    session_binding: Any
    session_lock: Any

    def build_device_args(self, library): ...
    def mark_completed(self, *, completion_verified: bool): ...
    def quarantine(self, reason, error): ...
    def close(self): ...


class SessionState(str, Enum):
    PREPARED = "prepared"
    IN_FLIGHT = "in_flight"
    ACCEPTED = "accepted"
    RETIRED = "retired"
    QUARANTINED = "quarantined"
    CLOSED = "closed"


@dataclass(frozen=True)
class SelectionResult(Mapping):
    """Immutable field binding; tensor contents remain caller-owned outputs."""

    _fields: Mapping

    @classmethod
    def from_tensors(cls, tensors):
        return cls(MappingProxyType({name: tensors[name] for name in OUTPUT_FIELDS}))

    def __getitem__(self, key):
        return self._fields[key]

    def __iter__(self):
        return iter(self._fields)

    def __len__(self):
        return len(self._fields)


@dataclass(frozen=True)
class CompletionResult:
    drained: bool
    success: bool
    status: Any
    selection: SelectionResult


def stream_identity(stream):
    handle = getattr(stream, "npu_stream", None)
    return (
        (str(getattr(stream, "device", "")), int(handle))
        if handle is not None
        else id(stream)
    )


class PreparedSession:
    """Own stream/lifetime checks, without adding kernels, copies, or syncs.

    Construct after a checked, idle lease and ContractLaunch preparation.
    The owner remains responsible for the actual lease release/quarantine.
    This session owns in_flight transitions for every subsequent submission.
    One session owns the lease until close. Reacquisition invalidates old
    sessions; they must never close or submit through the replacement lease.
    """

    def __init__(self, launcher, *, stream=None):
        self.launcher = launcher
        self.owner = launcher.owner
        if not isinstance(self.owner, QueueOwnerProtocol):
            raise ValueError("owner must implement QueueOwnerProtocol")
        with self.owner.session_lock:
            if self.owner.closed is not False or self.owner.lease_active is not True:
                raise ValueError("session preparation requires an active, open lease")
            if self.owner.in_flight is not False:
                raise ValueError("session preparation requires an idle acquired owner")
            generation = self.owner.lease_generation
            if (
                not isinstance(generation, Integral)
                or isinstance(generation, bool)
                or generation <= 0
            ):
                raise ValueError("lease generation must be a positive integer")
            if self.owner.lease_stream is None:
                raise ValueError("owner must expose the acquired lease stream")
            if self.owner.session_binding is not None:
                raise ValueError("active lease is already bound to a session")
            self.stream = self.owner.lease_stream if stream is None else stream
            if stream_identity(self.stream) != stream_identity(self.owner.lease_stream):
                raise ValueError(
                    "session stream differs from the acquired lease stream"
                )
            self._stream_identity = stream_identity(self.stream)
            self._lease_generation = int(generation)
            self.selection = SelectionResult.from_tensors(launcher.outputs)
            self.state = SessionState.PREPARED
            self.owner.session_binding = self
            self.owner.keepalive.append(self)

    def _check_lease(self):
        if self.owner.closed is not False or self.owner.lease_active is not True:
            raise RuntimeError("session lease is closed or inactive")
        generation = self.owner.lease_generation
        if (
            not isinstance(generation, Integral)
            or isinstance(generation, bool)
            or generation != self._lease_generation
        ):
            raise RuntimeError("session lease generation has changed")
        if self.owner.session_binding is not self:
            raise RuntimeError("session no longer owns the active lease binding")
        if (
            self.owner.lease_stream is None
            or stream_identity(self.owner.lease_stream) != self._stream_identity
        ):
            raise RuntimeError("owner acquired stream has changed")

    def _check_stream(self):
        current = self.launcher.torch.npu.current_stream(self.launcher.device)
        if stream_identity(current) != self._stream_identity:
            raise ValueError("session must execute and complete on its acquired stream")

    def _quarantine(self, reason, error):
        self.state = SessionState.QUARANTINED
        self.owner.in_flight = True
        self.owner.quarantine(reason, error)
        raise RuntimeError(f"quarantined: {reason}: {error}")

    def enqueue(self):
        with self.owner.session_lock:
            if self.state not in (SessionState.PREPARED, SessionState.ACCEPTED):
                raise RuntimeError(f"cannot enqueue a {self.state.value} session")
            self._check_lease()
            self._check_stream()
            if self.owner.in_flight:
                raise RuntimeError("owner has another submission in flight")
            self.owner.in_flight = True
            self.state = SessionState.IN_FLIGHT
            try:
                self.launcher.launch()
            except BaseException as error:
                self._quarantine("enqueue completion unknown", error)
        return self.selection

    def complete(self):
        with self.owner.session_lock:
            if self.state is not SessionState.IN_FLIGHT:
                raise RuntimeError("completion requires one in-flight submission")
            self._check_lease()
            self._check_stream()
            try:
                status = self.launcher.completion()
                self._check_lease()
                if self.owner.in_flight:
                    raise RuntimeError("launcher did not verify owner completion")
                success = bool(np.all(np.asarray(status) == Status.OK))
            except BaseException as error:
                self._quarantine("completion unverified", error)
            self.state = SessionState.ACCEPTED if success else SessionState.RETIRED
        return CompletionResult(True, success, status, self.selection)

    def close(self):
        with self.owner.session_lock:
            if self.state is SessionState.CLOSED:
                return
            if (
                self.state in (SessionState.IN_FLIGHT, SessionState.QUARANTINED)
                or self.owner.in_flight
            ):
                raise RuntimeError(
                    "cannot release in-flight or quarantined session resources"
                )
            self._check_lease()
            try:
                self.owner.close()
                if (
                    self.owner.closed is not True
                    or self.owner.lease_active is not False
                ):
                    raise RuntimeError(
                        "owner did not close and release the active lease"
                    )
            except BaseException as error:
                self._quarantine("lease release failed", error)
            if self.owner.session_binding is self:
                self.owner.session_binding = None
            self.state = SessionState.CLOSED


class GraphFrameSession:
    """Public frame-level lease boundary; callbacks remain outside captured work.

    Layer registrations belong to runtime shutdown. This object controls only
    begin/end and checked completion of one captured model frame at a time.
    current_stream returns the consumer's active stream. synchronize and
    check_completed must refer to the acquired stream, and run outside capture.
    A literal True from check_completed attests both drained DMA and successful
    row status; other values are conservatively quarantined. quarantine receives
    (reason, error, keepalive_tuple) and must retain that snapshot independently
    of this frame's lifetime until completion is proved or the process exits.
    """

    def __init__(
        self,
        runtime,
        stream,
        *,
        current_stream,
        synchronize,
        check_completed,
        quarantine,
        keepalive,
    ):
        validate_graph_transport(runtime)
        if stream is None:
            raise ValueError("Graph frame requires an acquired stream")
        if not all(
            callable(callback)
            for callback in (current_stream, synchronize, check_completed, quarantine)
        ):
            raise ValueError("Graph frame callbacks must be callable")
        self.runtime, self.stream = runtime, stream
        self.current_stream = current_stream
        self.synchronize = synchronize
        self.check_completed = check_completed
        self.quarantine = quarantine
        self.keepalive = tuple(keepalive)
        self._stream_identity = stream_identity(stream)
        self._lock = RLock()
        self.state = SessionState.PREPARED

    def _check_stream(self):
        if stream_identity(self.current_stream()) != self._stream_identity:
            raise ValueError(
                "Graph frame must execute and complete on its acquired stream"
            )

    def _fail(self, reason, error):
        self.state = SessionState.QUARANTINED
        self.quarantine(reason, error, self.keepalive)
        raise RuntimeError(f"Graph frame quarantined: {reason}: {error}")

    def begin(self, *, prepare=None):
        with self._lock:
            if self.state not in (SessionState.PREPARED, SessionState.ACCEPTED):
                raise RuntimeError(f"cannot begin a {self.state.value} Graph frame")
            self._check_stream()
            try:
                if prepare is not None:
                    prepare()
                self._check_stream()
                self.runtime.begin_fused_graph(self.stream)
            except BaseException as error:
                self._fail("frame preparation/acquisition failed", error)
            self.state = SessionState.IN_FLIGHT

    def complete(self):
        with self._lock:
            if self.state is not SessionState.IN_FLIGHT:
                raise RuntimeError("Graph completion requires an in-flight frame")
            self._check_stream()
            try:
                self.synchronize()
                self._check_stream()
                if self.check_completed() is not True:
                    raise RuntimeError("Graph completion or row status is unverified")
                self.runtime.end_fused_graph(self.stream)
            except BaseException as error:
                self._fail("Graph completion/release failed", error)
            self.state = SessionState.ACCEPTED

    def abort(self, error):
        with self._lock:
            if self.state is SessionState.IN_FLIGHT:
                self._fail("Graph execution failed", error)

    def close(self):
        with self._lock:
            if self.state in (SessionState.IN_FLIGHT, SessionState.QUARANTINED):
                raise RuntimeError("cannot release uncertain Graph frame resources")
            # Do not unregister runtime-owned layer output/queue registrations.
            self.state = SessionState.CLOSED
