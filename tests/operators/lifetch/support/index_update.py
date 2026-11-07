# SPDX-License-Identifier: Apache-2.0
"""Consume producer change records; no TopK scan, victim choice or KV copy."""

from types import SimpleNamespace
from kvip.operators.lifetch.index_update import DirectIndexUpdate

import numpy as np


def permutation_records(lru, changes, history):
    """CPU oracle only: snapshot source rows before applying the permutation."""
    active = [(int(dst), int(src)) for dst, src in zip(lru, changes) if src >= 0]
    destinations = {dst for dst, _ in active}
    sources = {src for _, src in active}
    if len(destinations) != len(active) or len(sources) != len(active):
        raise ValueError("Producer changes must have unique sources and destinations")
    if any(not 0 <= x < history for pair in active for x in pair):
        raise ValueError("Producer change outside Index storage")
    outgoing = [dst for dst, _ in active if dst not in sources]
    freed = [src for _, src in active if src not in destinations]
    return active + list(zip(freed, outgoing))


class DirectCacheUpdate(DirectIndexUpdate):
    def __init__(
        self,
        torch,
        keys,
        scales,
        addresses,
        output,
        batch,
        history,
        hot,
        *,
        library,
        topk,
        locations,
        rotating,
    ):
        self.torch, self.device = torch, keys.device
        self.batch, self.hot, self.history = batch, hot, history
        self.max_swaps = rotating.changes.shape[1]
        self.rotating = rotating
        self.identity = torch.arange(history, device=self.device).expand(batch, -1)
        self.logical = self.identity.clone()
        self.last_slots = locations.reshape(batch, -1)
        self.status = torch.zeros(batch, dtype=torch.int32, device=self.device)
        self.inputs = (
            keys,
            scales,
            addresses,
            self.logical,
            rotating.change_slots,
            rotating.changes,
            self.status,
        )
        row_owners = torch.arange(batch, dtype=torch.int32, device=self.device)
        self.unpack_indices = torch.arange(batch * 4, dtype=torch.int64, device=self.device)
        self.output_valid = torch.ones(batch * 4, dtype=torch.int64, device=self.device)
        super().__init__(
            torch,
            library,
            keys=keys.view(batch, history, 128),
            scales=scales.view(batch, history),
            addresses=addresses,
            logical_ids=self.logical,
            row_owners=row_owners,
            rotating=rotating,
            status=self.status,
            query_inputs=SimpleNamespace(unpack_indices=self.unpack_indices, output_valid=self.output_valid),
        )

    def set_order(self, orders):
        """Forward input preparation only; consume the existing service LRU."""
        candidates = np.asarray(orders, dtype=np.int32)[:, :2048].copy()
        if candidates.shape != (self.batch, 2048):
            raise ValueError("Expected 2048 existing LRU candidates per request")
        reserved = np.full((self.batch, self.hot), -1, dtype=np.int32)
        for row, slots in enumerate(candidates):
            if np.any(slots < 0) or np.any(slots >= self.hot) or len(np.unique(slots)) != 2048:
                raise ValueError("LRU candidates must be distinct hot slots")
            reserved[row, slots] = np.arange(2048, dtype=np.int32)
        self.rotating.lru_slots.copy_(self.torch.from_numpy(candidates))
        self.rotating.reserved_hot.copy_(self.torch.from_numpy(reserved))

    def __call__(self, topk=None, locations=None):
        super().__call__()

    def check_capacity(self):
        status = self.status.cpu().tolist()
        if any(status):
            raise RuntimeError(f"Invalid producer Index permutation: {status}")
