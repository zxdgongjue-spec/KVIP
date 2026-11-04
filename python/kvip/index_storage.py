# SPDX-License-Identifier: Apache-2.0
"""Service Index rows whose physical order follows the fused hot cache.

Prefill/prefix-cache keep their original paged Index storage. This storage
imports a prefix once, then only the current query writes. Committed DDR
addresses move with keys/scales/logical IDs in DirectIndexUpdate.
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
            raise ValueError("Committed Index address range shrank or exceeds compiled history")
        if count == previous:
            return
        pages = self.torch.tensor(page_addresses[previous:], dtype=self.torch.int64, device=self.device)
        offsets = self.identity[:block_size] * token_bytes
        self.addresses[owner, previous * block_size : count * block_size].copy_(
            (pages[:, None] + offsets[None, :]).reshape(-1)
        )
        self.published[owner] = count

    def _sources(self, cache):
        keys, scales = cache
        if keys.element_size() != 1 or keys.shape[-1] != 128 or scales.dtype != self.torch.float32:
            raise ValueError("LIFetch Index import requires C8 keys and float32 scales")
        source_keys = keys.view(self.torch.uint8).view(-1, 128)
        source_scales = scales.view(-1)
        if source_keys.shape[0] != source_scales.numel():
            raise ValueError("Paged Index key and scale capacities differ")
        return source_keys, source_scales

    def ingest_prefill(self, cache, metadata, plans, positions):
        """Eager prefill: import restored history once, then current writes."""
        source_keys, source_scales = self._sources(cache)
        for row, plan in enumerate(plans):
            owner = plan.owner
            if self.requests.get(owner) != plan.request:
                raise RuntimeError("Index owner must be prepared before ingestion")
            if owner not in self.imported and plan.start:
                # Cached prefix data may never have passed through this worker's
                # prefill callback. Read it once from the restored paged cache.
                tokens = self.identity[: plan.start]
                pages = self.torch.index_select(metadata.block_table[row], 0, tokens // metadata.block_size)
                sources = pages.long() * metadata.block_size + tokens % metadata.block_size
                self.keys[owner, : plan.start].copy_(self.torch.index_select(source_keys, 0, sources))
                self.scales[owner, : plan.start].copy_(self.torch.index_select(source_scales, 0, sources))
            self.imported.add(owner)
            sources = metadata.slot_mapping[plan.begin : plan.stop].long()
            targets = positions[plan.begin : plan.stop].long()
            self.keys[owner].index_copy_(0, targets, self.torch.index_select(source_keys, 0, sources))
            self.scales[owner].index_copy_(0, targets, self.torch.index_select(source_scales, 0, sources))

    def ingest_decode(self, cache, metadata, inputs, row_owners):
        """Fixed-shape graph writes using device owner/position metadata.

        Only B*Q rows move, regardless of history length. Invalid graph rows
        write distinct positions in reserved owner zero, avoiding duplicate
        index-copy destinations and leaving every live request untouched.
        """
        source_keys, source_scales = self._sources(cache)
        valid = inputs.query_ends > 0
        sources = self.torch.index_select(metadata.slot_mapping, 0, inputs.gather_indices).long()
        sources.masked_fill_(~valid, 0)
        targets = row_owners.long()[:, None] * self.history + inputs.query_ends.view(inputs.batch, 4).long() - 1
        targets = self.torch.where(valid, targets.reshape(-1), self.identity[: valid.numel()])
        self.keys.view(-1, 128).index_copy_(0, targets, self.torch.index_select(source_keys, 0, sources))
        self.scales.view(-1).index_copy_(0, targets, self.torch.index_select(source_scales, 0, sources))
