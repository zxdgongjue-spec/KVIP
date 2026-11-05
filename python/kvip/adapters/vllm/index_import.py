"""Import vLLM paged Index rows into persistent LIFetch storage."""

from kvip.cache.index_storage import IndexCacheStorage


class VllmIndexCacheStorage(IndexCacheStorage):
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


# Legacy public name, implemented by the vLLM adapter subclass.
IndexCacheStorage = VllmIndexCacheStorage
