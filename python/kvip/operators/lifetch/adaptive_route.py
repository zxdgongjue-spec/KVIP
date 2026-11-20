# SPDX-License-Identifier: Apache-2.0
"""Persistent URMA route feedback for one Index group and ordered execution stream."""


class AdaptiveRoute:
    def __init__(self, torch, device, batch, owners, *, policy="zero", requests=None):
        if policy not in ("zero", "early") or not 1 <= batch <= 16 or owners < batch:
            raise ValueError("Adaptive URMA requires zero/early policy and enough owners for each row")
        self.torch, self.device, self.batch, self.owners = torch, device, batch, owners
        self.policy = 1 if policy == "zero" else 2
        self.requests = requests if requests is not None else torch.zeros((batch, 4), dtype=torch.int64, device=device)
        if (tuple(self.requests.shape) != (batch, 4) or self.requests.dtype != torch.int64
                or self.requests.device != torch.device(device) or not self.requests.is_contiguous()):
            raise ValueError("Request identity requires contiguous int64 [batch,4] on the bound device")
        # Different request workers commit independently. Keep their writes on separate lines.
        self._feedback = torch.zeros((owners, 16), dtype=torch.int64, device=device)
        self.feedback = self._feedback[:, :4]
        # Timer words and per-request counts have independent device writers.
        observations = torch.zeros((97, 16), dtype=torch.int64, device=device)
        self.mode = observations[0].view(torch.int32)[:1]
        self.used_high_bound = observations[0].view(torch.int32)[1:2]
        self.probe_ticks = observations[1:, 0]
        # Retired read bound and optional timing have separate cache-line writers.
        self.high_bound = torch.zeros((2, 16), dtype=torch.int64, device=device)
        self.demand = torch.zeros((batch, 32), dtype=torch.int32, device=device)[:, 0]

    def invalidate_high_bound(self):
        """Invalidate derived state before externally restoring or fault-injecting feedback on this stream."""
        self.high_bound[0, :1].zero_()

    def set_requests(self, rows):
        """Stage [owner, generation, step, Q count] outside capture. Zero generation disables learning."""
        import numpy as np

        rows = np.asarray(rows, dtype=np.int64)
        if rows.shape != (self.batch, 4):
            raise ValueError("Expected one four-field request record per packed row")
        active = rows[rows[:, 1] != 0]
        if (np.any(rows < 0) or np.any(rows[:, 3] > 4) or np.any(active[:, 0] >= self.owners)
                or len(np.unique(active[:, 0])) != len(active)):
            raise ValueError("Invalid or repeated active owner/generation/step/query record")
        self.requests.copy_(self.torch.from_numpy(rows.copy()), non_blocking=True)
