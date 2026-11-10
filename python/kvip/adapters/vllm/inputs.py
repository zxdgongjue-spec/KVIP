# SPDX-License-Identifier: Apache-2.0
"""Stable Q4 input views for the original LIFetch kernel.

Scheduler row packing is uploaded once before forward and shared by all Index
layers. Positions come from the runner's device tensor (including MTP correction),
not the CPU's possibly stale computed-token count. No host read is needed.
"""

import numpy as np

QUERIES_PER_REQUEST = 4
INDEX_WIDTH = 128


def reserved_hot_reference(candidates, query_ends, resident):
    """Diagnostic CPU reference for init's compact LRU inverse mapping.

    Inactive packed rows are inert and canonicalized to -1 in snapshots. The
    service uses its existing init kernel, never this CPU reconstruction.
    """
    candidates, query_ends = np.asarray(candidates), np.asarray(query_ends)
    if (
        candidates.ndim != 2
        or candidates.dtype != np.int32
        or query_ends.size != candidates.shape[0] * QUERIES_PER_REQUEST
        or type(resident) is not int
        or resident < candidates.shape[1]
    ):
        raise ValueError("Reserved-hot reference requires int32 candidates, Q4 bounds and sufficient hot slots")
    ends = query_ends.reshape(candidates.shape[0], QUERIES_PER_REQUEST)
    if np.any(ends < 0):
        raise ValueError("Reserved-hot query bounds must be nonnegative")
    result = np.full((candidates.shape[0], resident), -1, dtype=np.int32)
    ordinals = np.arange(candidates.shape[1], dtype=np.int32)
    for row in np.flatnonzero(np.any(ends > 0, axis=1)):
        slots = candidates[row]
        if np.any(slots < 0) or np.any(slots >= resident) or len(np.unique(slots)) != len(slots):
            raise ValueError("Active LRU candidates must be unique in-range hot slots")
        result[row, slots] = ordinals
    return result


def committed_candidates(candidates, committed, resident):
    """Keep Index permutations inside the committed prefix, before forward.

    The v9 allocator consumes candidates sequentially. With fewer than TOPK
    committed rows it can fetch at most that many distinct rows: remaining
    entries only retain the fixed ABI shape and cannot become destinations.
    ABC and future Index rows must stay at their logical positions.
    """
    limit = min(committed, resident)
    if limit == resident:
        return candidates
    eligible = candidates[candidates < limit]
    missing = np.setdiff1d(np.arange(limit, dtype=np.int32), eligible, assume_unique=True)
    return np.concatenate((eligible, missing, candidates[candidates >= limit]))[: len(candidates)]


class QueryInputs:
    def __init__(self, torch, batch, device, *, pin_memory=True, history=0, index_heads=64):
        if index_heads not in (32, 64):
            raise ValueError("LIFetch supports 32 or 64 Indexer heads")
        self.index_heads = index_heads
        self.torch, self.batch, self.device = torch, batch, device
        capacity = batch * QUERIES_PER_REQUEST
        # Fields share one upload, including output padding validity. Graph
        # padding must not inherit the first real request's selected slots.
        self.metadata = torch.zeros((5, capacity), dtype=torch.int64, device=device)
        self.host = [torch.zeros((5, capacity), dtype=torch.int64, pin_memory=pin_memory) for _ in range(2)]
        self.host_events = [None, None]
        self.turn = 0
        self.gather_indices, self.query_valid, self.unpack_indices, self.output_valid = self.metadata[:4].unbind(0)
        # The kernel owner ABI is packed int32, while Torch gather indices are
        # int64. Reserve the last row as bytes so both share the same H2D copy.
        self.row_owners = self.metadata[4].view(torch.int32)[:batch]
        self.query = torch.empty((capacity, self.index_heads, INDEX_WIDTH), dtype=torch.float8_e4m3fn, device=device)
        self.weights = torch.empty((capacity, self.index_heads), dtype=torch.bfloat16, device=device)
        self.scale = torch.empty((capacity, self.index_heads), dtype=torch.float32, device=device)
        self.position_scratch = torch.empty(capacity, dtype=torch.int64, device=device)
        self.query_ends = torch.zeros(capacity, dtype=torch.int32, device=device)
        self.token_count = 0
        self.block_offsets = torch.arange(history // 128, dtype=torch.int32, device=device)
        self.block_table = torch.zeros((batch, history // 128), dtype=torch.int32, device=device)
        self.block_bases = torch.zeros((batch, 1), dtype=torch.int32, device=device)

    @property
    def nbytes(self):
        return sum(
            t.numel() * t.element_size()
            for t in (
                self.metadata,
                self.query,
                self.weights,
                self.scale,
                self.position_scratch,
                self.query_ends,
                self.block_offsets,
                self.block_table,
                self.block_bases,
            )
        )

    def prepare(self, query_counts, *, query_starts=None, query_owners=None):
        """Input preparation, outside capture; no device-dependent branching.

        The caller invokes this after the previous forward has been enqueued on
        the same stream. The event protects only reuse of pinned H2D staging.
        """
        counts = tuple(query_counts)
        if len(counts) > self.batch or any(type(n) is not int or not 0 <= n <= 4 for n in counts):
            raise ValueError("LIFetch rows require zero to four queries per request")
        starts = None if query_starts is None else tuple(query_starts)
        if starts is not None and (len(starts) != len(counts) or any(type(n) is not int or n < 0 for n in starts)):
            raise ValueError("Decode query starts must match request rows and be nonnegative")
        owners = None if query_owners is None else tuple(query_owners)
        if owners is not None and (
            len(owners) != len(counts) or any(type(n) is not int or not 0 <= n < (1 << 31) for n in owners)
        ):
            raise ValueError("Decode owners must match request rows and fit nonnegative int32")
        ring = self.turn % 2
        event = self.host_events[ring]
        if event is not None and not event.query():
            event.synchronize()
        values = self.host[ring].numpy()
        values.fill(0)
        if owners is not None:
            self.host[ring][4].view(self.torch.int32)[: len(owners)].numpy()[...] = owners
        begin = 0
        for row, count in enumerate(counts):
            start = row * QUERIES_PER_REQUEST
            source = begin if starts is None else starts[row]
            values[0, start : start + count] = np.arange(source, source + count)
            values[1, start : start + count] = 1
            values[2, begin : begin + count] = np.arange(start, start + count)
            values[3, begin : begin + count] = 1
            begin += count
        self.token_count = begin
        self.metadata.copy_(self.host[ring], non_blocking=True)
        if self.block_offsets.numel():
            self.torch.mul(self.row_owners[:, None], self.block_offsets.numel(), out=self.block_bases)
            self.torch.add(self.block_bases, self.block_offsets, out=self.block_table)
        if self.device.type != "cpu":
            event = self.host_events[ring]
            if event is None:
                event = self.host_events[ring] = self.torch.npu.Event()
            event.record(self.torch.npu.current_stream(self.device))
        self.turn += 1

    def bind_positions(self, positions):
        """Enqueue once per forward, after the runner corrects MTP positions."""
        self.torch.index_select(positions, 0, self.gather_indices, out=self.position_scratch)
        self.position_scratch.add_(1).mul_(self.query_valid)
        self.query_ends.copy_(self.position_scratch)
        return self.query_ends.view(self.batch, QUERIES_PER_REQUEST)

    def pack(self, query, weights, scale):
        """Graph-safe gather into fixed buffers; native projection stays unchanged."""
        if weights.ndim != 2 or weights.shape[1] != self.index_heads:
            raise ValueError("LIFetch weights require [tokens,index_heads]")
        tokens = weights.shape[0]
        # AscendSFAIndexer._quantize_li_tensor flattens token/head dimensions.
        # Restore views here, without changing the native Indexer return ABI.
        if tuple(query.shape) == (tokens * self.index_heads, INDEX_WIDTH):
            query = query.view(tokens, self.index_heads, INDEX_WIDTH)
        if tuple(query.shape) != (tokens, self.index_heads, INDEX_WIDTH) or query.dtype != self.query.dtype:
            raise ValueError("LIFetch requires FP8 e4m3 Index queries with the configured head count and width 128")
        if scale is None or scale.dtype != self.scale.dtype or weights.dtype != self.weights.dtype:
            raise ValueError("LIFetch requires float32 query scales and BF16 weights")
        if tuple(scale.shape) in ((tokens * self.index_heads,), (tokens * self.index_heads, 1)):
            scale = scale.view(tokens, self.index_heads)
        if tuple(scale.shape) != (tokens, self.index_heads):
            raise ValueError("LIFetch requires one scale per token/head")
        # aclnnIndexSelect supports uint8, not FP8. Reinterpret bytes without
        # conversion; the persistent FP8 output and its graph address stay intact.
        self.torch.index_select(
            query.view(self.torch.uint8), 0, self.gather_indices, out=self.query.view(self.torch.uint8)
        )
        self.torch.index_select(weights, 0, self.gather_indices, out=self.weights)
        self.torch.index_select(scale, 0, self.gather_indices, out=self.scale)
        return self.query, self.weights, self.scale
