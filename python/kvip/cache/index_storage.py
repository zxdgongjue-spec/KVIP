# SPDX-License-Identifier: Apache-2.0
"""Persistent Index rows, ownership and committed DDR addresses.

Paged-cache import lives in adapters.vllm.index_import. The generic state has no
vLLM metadata or indexer dependency.
"""


class IndexCacheStorage:
    def __init__(self, torch, owners, history, resident, device):
        self.torch, self.history, self.device = torch, history, device
        self.keys = torch.zeros((owners, history, 128), dtype=torch.uint8, device=device)
        self.scales = torch.ones((owners, history), dtype=torch.float32, device=device)
        self.addresses = torch.zeros((owners, history), dtype=torch.int64, device=device)
        self.identity = torch.arange(history, dtype=torch.int64, device=device)
        self.logical_ids = self.identity.expand(owners, -1).clone()
        self.hot_valid = torch.zeros((owners, resident), dtype=torch.int32, device=device)
        self.requests = {}
        self.imported = set()
        self.published = {}

    @property
    def nbytes(self):
        return sum(
            t.numel() * t.element_size()
            for t in (
                self.keys,
                self.scales,
                self.addresses,
                self.identity,
                self.logical_ids,
                self.hot_valid,
            )
        )

    def prepare(self, plans):
        """Invalidate recycled owner metadata before queuing this round."""
        for plan in plans:
            if self.requests.get(plan.owner) == plan.request:
                continue
            self.requests[plan.owner] = plan.request
            self.imported.discard(plan.owner)
            self.published[plan.owner] = 0
            self.logical_ids[plan.owner].copy_(self.identity)
            self.hot_valid[plan.owner].zero_()
            self.addresses[plan.owner].zero_()

    def publish(self, owner, page_addresses, *, block_size, token_bytes):
        """Append new committed pages; never rewrite already-permuted entries."""
        count = len(page_addresses)
        previous = self.published[owner]
        if count < previous or count * block_size > self.history:
            raise ValueError("Committed Index address range shrank or exceeds bound history")
        if count == previous:
            return
        pages = self.torch.tensor(page_addresses[previous:], dtype=self.torch.int64, device=self.device)
        offsets = self.identity[:block_size] * token_bytes
        self.addresses[owner, previous * block_size : count * block_size].copy_(
            (pages[:, None] + offsets[None, :]).reshape(-1)
        )
        self.published[owner] = count
