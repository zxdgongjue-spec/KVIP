"""Boundary-only import/export of the operator-owned hot/paged-cold cache.

No persistent canonical payload or temporary payload is retained. Stable Q4
writes use the device writer; these Torch operations run only on ownership
transitions. Callers must prove exclusive writable ownership of the pages.
"""

from numbers import Integral

if __package__:
    from .index_cache import allocate_cache
else:
    from index_cache import allocate_cache


class OwnedIndexCache:
    def __init__(
        self,
        torch,
        geometry,
        key,
        scale,
        table,
        *,
        pool,
        length,
        budget,
        forward,
        reverse,
        page_ids=None,
    ):
        self.torch, self.g = torch, geometry
        self.key, self.scale = key, scale
        self.active = True
        self.poisoned = False
        self.page_ids = self._page_ids(table, page_ids, length)
        self.table = table.clone()
        self.pool, self.length, self.budget = int(pool), int(length), int(budget)
        self.buffer = allocate_cache(torch, geometry, key.device, owned=True)
        self.header = self.buffer[:128].view(torch.int32)
        s, c = geometry.resident_capacity, geometry.candidate_capacity
        self.hot_key = self.buffer[128 : 128 + s * 128].reshape(s, 128)
        self.hot_scale = self.buffer[128 + s * 128 : 128 + s * 132].view(torch.float32)
        self.ids = self.buffer[128 + s * 132 : 128 + s * 132 + (s + c) * 4].view(
            torch.int32
        )
        begin = 128 + s * 132 + (s + c) * 8
        self.positions = self.buffer[begin : begin + c * 4].view(torch.int32)
        tokens = torch.arange(length, device=key.device, dtype=torch.int64)
        slots = torch.arange(budget, device=key.device, dtype=torch.int64)
        resident = reverse[pool, :budget].to(torch.int64)
        valid = (resident >= 0) & (resident < length)
        valid &= forward[pool, resident.clamp(0, c - 1)].to(torch.int64) == slots
        hot = resident[valid]
        membership = torch.zeros(length, dtype=torch.bool, device=key.device)
        membership[hot] = True
        cold = tokens[~membership]
        nh, nc = hot.numel(), cold.numel()
        # Gather ALL sources before writing either bank: compacted cold pages
        # overlap the logical input and cannot be populated in a streaming loop.
        # Gather each payload row once, directly into its destination bank
        # order. Avoid a full logical payload gather followed by hot/cold gathers.
        hot_source, cold_source = self.physical(hot), self.physical(cold)
        flat_key, flat_scale = key.view(torch.uint8).reshape(-1, 128), scale.reshape(-1)
        hot_key, hot_scale = flat_key[hot_source], flat_scale[hot_source]
        cold_key, cold_scale = flat_key[cold_source], flat_scale[cold_source]
        self.hot_key[:nh].copy_(hot_key)
        self.hot_scale[:nh].copy_(hot_scale)
        destinations = self.physical(
            torch.arange(nc, device=key.device, dtype=torch.int64)
        )
        flat_key.index_copy_(0, destinations, cold_key)
        flat_scale.index_copy_(0, destinations, cold_scale)
        self.ids.fill_(-1)
        self.ids[:nh].copy_(hot.to(torch.int32))
        self.ids[s : s + nc].copy_(cold.to(torch.int32))
        self.positions.fill_(-1)
        self.positions[hot] = (
            -torch.arange(nh, device=key.device, dtype=torch.int32) - 2
        )
        self.positions[cold] = torch.arange(nc, device=key.device, dtype=torch.int32)
        self.header[:8].copy_(
            torch.tensor(
                [1, length, nh, nc, pool, geometry.layer_id, budget, 0],
                dtype=torch.int32,
                device=key.device,
            )
        )

    def physical(self, positions):
        return self.table[positions // 128].to(self.torch.int64) * 128 + positions % 128

    def _require_active(self):
        if not self.active or self.poisoned:
            raise RuntimeError(
                "owned Index is retired or has an incomplete page transition"
            )

    def _page_ids(self, table, page_ids, length):
        """Validate host ownership metadata without reading a device page table."""
        torch = self.torch
        if (
            table.ndim != 1
            or table.dtype != torch.int32
            or table.device != self.key.device
            or not table.is_contiguous()
            or table.numel() * 128 < length
        ):
            raise ValueError(
                "page table must be a covering contiguous same-device int32 row"
            )
        if page_ids is None:
            if table.device.type != "cpu":
                return None
            # CPU fixtures have no separate scheduler metadata. Never do this on NPU.
            page_ids = table.tolist()
        if not isinstance(page_ids, (tuple, list)) or any(
            not isinstance(p, Integral) or isinstance(p, bool) for p in page_ids
        ):
            raise ValueError("page_ids must be host integer ownership metadata")
        pages = tuple(int(p) for p in page_ids)
        if not (length + 127) // 128 <= len(pages) <= table.numel():
            raise ValueError("host page metadata does not cover the owned length")
        if len(set(pages)) != len(pages) or any(
            p < 0 or p >= self.g.index_blocks for p in pages
        ):
            raise ValueError("page IDs must be unique and within the Index pool")
        if table.device.type == "cpu" and tuple(table[: len(pages)].tolist()) != pages:
            raise ValueError("host page metadata differs from the CPU table")
        return pages

    def grow(self, length, table=None, *, page_ids=None):
        """Append eligible rows; existing physical page mappings are immutable.

        NPU table refresh requires scheduler-owned CPU page_ids. A page replacement
        must use rebind_pages at an explicit checked ownership boundary instead.
        """
        self._require_active()
        if (
            not isinstance(length, Integral)
            or isinstance(length, bool)
            or not self.length <= length <= self.g.candidate_capacity
        ):
            raise ValueError("Index growth is outside the owned candidate capacity")
        if table is not None:
            pages = self._page_ids(table, page_ids, length)
            if self.page_ids is None or pages is None:
                raise ValueError(
                    "table refresh requires host page_ids from the ownership boundary"
                )
            if pages[: len(self.page_ids)] != self.page_ids:
                raise ValueError(
                    "grow cannot replace existing pages; use checked rebind_pages"
                )
            if pages != self.page_ids:
                # Stable metadata already owns an identical private page table.
                # Do not add a device copy for same-length checks or in-page growth.
                self.table = table.clone()
                self.page_ids = pages
        elif page_ids is not None:
            raise ValueError("page_ids requires a refreshed page table")
        elif length > self.table.numel() * 128 or (
            self.page_ids is not None and length > len(self.page_ids) * 128
        ):
            raise ValueError("existing page table does not cover Index growth")
        if length == self.length:
            return
        torch = self.torch
        tokens = torch.arange(
            self.length, length, device=self.key.device, dtype=torch.int64
        )
        source = self.physical(tokens)
        # Clone both planes before writing: a very short/empty hot bank can
        # make the appended cold destination overlap the incoming tail range.
        keys = self.key.view(torch.uint8).reshape(-1, 128)[source]
        scales = self.scale.reshape(-1)[source]
        positions = (
            torch.arange(
                length - self.length, device=self.key.device, dtype=torch.int64
            )
            + self.header[3]
        )
        physical = self.physical(positions)
        try:
            self.key.view(torch.uint8).reshape(-1, 128).index_copy_(0, physical, keys)
            self.scale.reshape(-1).index_copy_(0, physical, scales)
            self.ids[self.g.resident_capacity + positions] = tokens.to(torch.int32)
            self.positions[tokens] = positions.to(torch.int32)
            self.header[20] = 0
            self.header[3].add_(length - self.length)
            self.header[1] = length
            self.length = length
        except BaseException:
            self.poisoned = True
            self.header[20] = 0
            raise

    def logical(self):
        """Temporary logical snapshot; callers must not retain it as a mirror."""
        self._require_active()
        torch = self.torch
        tokens = torch.arange(self.length, device=self.key.device, dtype=torch.int64)
        pos = self.positions[: self.length].to(torch.int64)
        hot = pos <= -2
        hp = (-pos - 2).clamp(0, self.g.resident_capacity - 1)
        cp = self.physical(pos.clamp(0, self.g.candidate_capacity - 1))
        keys = torch.where(
            hot[:, None],
            self.hot_key[hp],
            self.key.view(torch.uint8).reshape(-1, 128)[cp],
        )
        scales = torch.where(hot, self.hot_scale[hp], self.scale.reshape(-1)[cp])
        return tokens, keys, scales

    def rebind_pages(
        self, table, *, completion_verified, exclusive_pages, logical_length=None
    ):
        """Boundary-only page replacement after checked completion.

        Retain hot payload/IDs, relocate the cold bank, and preserve the explicit
        logical tail before overwriting any destination. Caller proves exclusive
        ownership and valid physical pages. Steady grow/write never calls this.
        """
        self._require_active()
        npu = getattr(self.torch, "npu", None)
        capturing = getattr(npu, "is_current_stream_capturing", None)
        if self.key.device.type != "cpu" and callable(capturing) and capturing():
            raise RuntimeError("page replacement is forbidden during capture")
        if completion_verified is not True or exclusive_pages is not True:
            raise ValueError(
                "page replacement requires checked completion and exclusive pages"
            )
        torch = self.torch
        end = self.length if logical_length is None else logical_length
        if (
            not isinstance(end, int)
            or isinstance(end, bool)
            or not self.length <= end <= table.numel() * 128
        ):
            raise ValueError("invalid logical tail length")
        if (
            table.ndim != 1
            or table.dtype != torch.int32
            or table.device != self.table.device
        ):
            raise ValueError("replacement page table must be a same-device int32 row")
        if end > self.table.numel() * 128:
            raise ValueError("old page table does not cover the logical tail")
        # An explicit ownership boundary may validate page values on the host;
        # this method is never called from enqueue or Graph capture/replay.
        pages = table[: (end + 127) // 128].detach().cpu().tolist()
        if len(set(pages)) != len(pages) or any(
            p < 0 or p >= self.g.index_blocks for p in pages
        ):
            raise ValueError(
                "replacement pages must be unique and within the Index pool"
            )
        flat_key = self.key.view(torch.uint8).reshape(-1, 128)
        flat_scale = self.scale.reshape(-1)
        tokens, keys, scales = self.logical()
        positions = self.positions[tokens].to(torch.int64)
        cold = positions >= 0
        tail = torch.arange(self.length, end, device=self.key.device, dtype=torch.int64)
        tail_keys = flat_key[self.physical(tail)]
        tail_scales = flat_scale[self.physical(tail)]
        # Resolve and validate all destination indices before publishing a new
        # table or writing. Invalid pages fail without abandoning the old owner.
        targets = table[positions[cold] // 128].long() * 128 + positions[cold] % 128
        tail_targets = table[tail // 128].long() * 128 + tail % 128
        try:
            flat_key.index_copy_(0, targets, keys[cold])
            flat_scale.index_copy_(0, targets, scales[cold])
            flat_key.index_copy_(0, tail_targets, tail_keys)
            flat_scale.index_copy_(0, tail_targets, tail_scales)
        except BaseException:
            # Writes may already have happened. Never expose this as rollback.
            self.poisoned = True
            self.header[20] = 0
            raise
        self.table = table.clone()
        self.page_ids = tuple(pages)
        self.header[20] = 0

    def restore_logical(self):
        """Complete before a native reader/writer, then discard this owner."""
        tokens, keys, scales = self.logical()
        physical = self.physical(tokens)
        try:
            self.key.view(self.torch.uint8).reshape(-1, 128).index_copy_(
                0, physical, keys
            )
            self.scale.reshape(-1).index_copy_(0, physical, scales)
            self.header[20] = 0
            self.header[0] = 0
            self.active = False
        except BaseException:
            self.poisoned = True
            self.header[20] = 0
            raise

    def write_logical(self, tokens, keys, scales):
        """Boundary/test writer; production Q4 uses one device routing kernel."""
        self._require_active()
        torch = self.torch
        tokens = tokens.to(device=self.key.device, dtype=torch.int64)
        pos = self.positions[tokens].to(torch.int64)
        hot = pos <= -2
        try:
            self.hot_key.index_copy_(0, -pos[hot] - 2, keys[hot])
            self.hot_scale.index_copy_(0, -pos[hot] - 2, scales[hot])
            cold = ~hot
            physical = self.physical(pos[cold])
            self.key.view(torch.uint8).reshape(-1, 128).index_copy_(
                0, physical, keys[cold]
            )
            self.scale.reshape(-1).index_copy_(0, physical, scales[cold])
        except BaseException:
            self.poisoned = True
            self.header[20] = 0
            raise
