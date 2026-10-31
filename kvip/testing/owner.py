"""Deferred CPU owner for public-session tests, never a production provider."""

from threading import RLock


class DeferredTransport:
    def __init__(self, source, output):
        self.source = source
        self.output = output
        self.pending = []
        self.poisoned = False

    def submit(self, source_row, target_row):
        if self.poisoned:
            raise RuntimeError("transport quarantined")
        if not 0 <= source_row < len(self.source) or not 0 <= target_row < len(
            self.output
        ):
            raise ValueError("invalid local GET descriptor")
        self.pending.append((source_row, target_row))

    def complete(self, *, fail=False):
        if self.poisoned or fail:
            self.poisoned = True
            raise RuntimeError("completion unknown; allocations retained")
        for source, target in self.pending:
            self.output[target] = self.source[source]
        self.pending.clear()


class StubQueueOwner:
    transport = "host-contract-test"

    def __init__(self, output, stream, *, copier=None):
        self.output, self.stream = output, stream
        self.device = getattr(output, "device", None)
        self.abi_sizes = {}
        self.core_status = None
        self.copier = copier
        self.keepalive = [output, copier]
        self.in_flight = False
        self.closed = False
        # PreparedSession's explicit lease protocol. Real owners initialize the
        # inactive state, then publish these fields on successful acquisition.
        # This CPU owner is constructed with its first lease already acquired.
        self.session_lock = RLock()
        self.session_binding = None
        self.lease_active = True
        self.lease_generation = 1
        self.lease_stream = stream
        self.poisoned = False
        self.fail_release = False

    def acquire(self, stream):
        with self.session_lock:
            if self.closed or self.lease_active or self.in_flight or self.poisoned:
                raise RuntimeError("cannot acquire an active or uncertain host owner")
            self.stream = self.lease_stream = stream
            self.lease_generation += 1
            self.session_binding = None
            self.lease_active = True

    def release(self):
        with self.session_lock:
            if self.in_flight or self.poisoned or self.fail_release:
                raise RuntimeError("cannot release uncertain host resources")
            self.lease_active = False
            self.session_binding = None

    def build_device_args(self, library):
        raise RuntimeError("host owner cannot pack production device queue arguments")

    def mark_completed(self, *, completion_verified):
        with self.session_lock:
            if self.closed or not self.lease_active:
                raise RuntimeError("completion requires an active host lease")
            if not completion_verified or (
                self.copier is not None and self.copier.pending
            ):
                self.quarantine("unverified host completion", "")
            self.in_flight = False

    def quarantine(self, reason, error):
        with self.session_lock:
            self.poisoned = True
            raise RuntimeError(f"host owner quarantined: {reason}: {error}")

    def close(self):
        with self.session_lock:
            if self.closed:
                return
            self.release()
            self.closed = True
            self.keepalive.clear()
