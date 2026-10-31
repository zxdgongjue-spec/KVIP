"""Explicit eager lifecycle around the compatible low-level ABI13 launcher.

Graph capture/replay continues to use its frame-level owner and device guard;
an eager session is deliberately not captured as a Python state machine.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .access import OUTPUT_FIELDS
from .contract import Status
from .transport import validate_graph_transport


@runtime_checkable
class QueueOwnerProtocol(Protocol):
    output: Any
    device: Any
    keepalive: list
    in_flight: bool
    abi_sizes: dict
    core_status: Any

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
    """

    def __init__(self, launcher, *, stream=None):
        self.launcher = launcher
        self.owner = launcher.owner
        if not isinstance(self.owner, QueueOwnerProtocol):
            raise ValueError("owner must implement QueueOwnerProtocol")
        if self.owner.in_flight:
            raise ValueError("session preparation requires an idle acquired owner")
        self.stream = (
            stream
            if stream is not None
            else launcher.torch.npu.current_stream(launcher.device)
        )
        lease_stream = getattr(
            self.owner, "stream", getattr(self.owner, "lease_stream", self.stream)
        )
        if stream_identity(self.stream) != stream_identity(lease_stream):
            raise ValueError("session stream differs from the acquired lease stream")
        self._stream_identity = stream_identity(self.stream)
        self.selection = SelectionResult.from_tensors(launcher.outputs)
        self.state = SessionState.PREPARED
        self.owner.keepalive.append(self)

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
        if self.state not in (SessionState.PREPARED, SessionState.ACCEPTED):
            raise RuntimeError(f"cannot enqueue a {self.state.value} session")
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
        if self.state is not SessionState.IN_FLIGHT:
            raise RuntimeError("completion requires one in-flight submission")
        self._check_stream()
        try:
            status = self.launcher.completion()
            if self.owner.in_flight:
                raise RuntimeError("launcher did not verify owner completion")
            success = bool(np.all(np.asarray(status) == Status.OK))
        except BaseException as error:
            self._quarantine("completion unverified", error)
        self.state = SessionState.ACCEPTED if success else SessionState.RETIRED
        return CompletionResult(True, success, status, self.selection)

    def close(self):
        if self.state is SessionState.CLOSED:
            return
        if (
            self.state in (SessionState.IN_FLIGHT, SessionState.QUARANTINED)
            or self.owner.in_flight
        ):
            raise RuntimeError(
                "cannot release in-flight or quarantined session resources"
            )
        try:
            self.owner.close()
        except BaseException as error:
            self._quarantine("lease release failed", error)
        self.state = SessionState.CLOSED


class GraphFrameSession:
    """Public frame-level lease boundary; callbacks remain outside captured work.

    Layer registrations belong to runtime shutdown. This object controls only
    begin/end and checked completion of one captured model frame at a time.
    check_completed must return a host bool after the supplied synchronization.
    """

    def __init__(
        self, runtime, stream, *, synchronize, check_completed, quarantine, keepalive
    ):
        validate_graph_transport(runtime)
        self.runtime, self.stream = runtime, stream
        self.synchronize = synchronize
        self.check_completed = check_completed
        self.quarantine = quarantine
        self.keepalive = keepalive
        self.state = SessionState.PREPARED

    def _fail(self, reason, error):
        self.state = SessionState.QUARANTINED
        self.quarantine(reason, error)
        raise RuntimeError(f"Graph frame quarantined: {reason}: {error}")

    def begin(self, *, prepare=None):
        if self.state not in (SessionState.PREPARED, SessionState.ACCEPTED):
            raise RuntimeError(f"cannot begin a {self.state.value} Graph frame")
        try:
            if prepare is not None:
                prepare()
            self.runtime.begin_fused_graph(self.stream)
        except BaseException as error:
            self._fail("frame preparation/acquisition failed", error)
        self.state = SessionState.IN_FLIGHT

    def complete(self):
        if self.state is not SessionState.IN_FLIGHT:
            raise RuntimeError("Graph completion requires an in-flight frame")
        try:
            self.synchronize()
            if self.check_completed() is not True:
                raise RuntimeError("Graph completion or row status is unverified")
            self.runtime.end_fused_graph(self.stream)
        except BaseException as error:
            self._fail("Graph completion/release failed", error)
        self.state = SessionState.ACCEPTED

    def abort(self, error):
        if self.state is SessionState.IN_FLIGHT:
            self._fail("Graph execution failed", error)

    def close(self):
        if self.state in (SessionState.IN_FLIGHT, SessionState.QUARANTINED):
            raise RuntimeError("cannot release uncertain Graph frame resources")
        # Do not unregister runtime-owned layer output/queue registrations.
        self.state = SessionState.CLOSED
