"""Deferred host-copy stub for testing lifetime and failure contracts only."""


class HostTransport:
    def __init__(self, source, output):
        self.source, self.output = source, output
        self.pending = []
        self.closed = self.poisoned = False

    def submit(self, source_row, target_row):
        if self.closed or self.poisoned:
            raise RuntimeError("transport unavailable")
        if not 0 <= source_row < len(self.source) or not 0 <= target_row < len(self.output):
            raise ValueError("invalid GET descriptor")
        self.pending.append((source_row, target_row))

    def complete(self, *, fail=False):
        if self.poisoned:
            raise RuntimeError("transport quarantined")
        if fail:
            self.poisoned = True
            raise RuntimeError("completion unknown; allocations retained")
        for source, target in self.pending:
            self.output[target] = self.source[source]
        self.pending.clear()

    def close(self):
        if self.pending or self.poisoned:
            raise RuntimeError("cannot release before verified completion")
        self.closed = True
