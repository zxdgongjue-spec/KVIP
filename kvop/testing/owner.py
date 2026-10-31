"""Deferred CPU owner for public-session tests, never a production provider."""


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
        self.poisoned = False
        self.fail_release = False

    def build_device_args(self, library):
        raise RuntimeError("host owner cannot pack production device queue arguments")

    def mark_completed(self, *, completion_verified):
        if not completion_verified or (self.copier is not None and self.copier.pending):
            self.quarantine("unverified host completion", "")
        self.in_flight = False

    def quarantine(self, reason, error):
        self.poisoned = True
        raise RuntimeError(f"host owner quarantined: {reason}: {error}")

    def close(self):
        if self.in_flight or self.poisoned or self.fail_release:
            raise RuntimeError("cannot release uncertain host resources")
        self.closed = True
        self.keepalive.clear()
