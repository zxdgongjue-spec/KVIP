"""Boundary-only import/export of the operator-owned hot/paged-cold cache.

No persistent canonical payload or temporary payload is retained. Stable Q4
writes use the device writer; these Torch operations run only on ownership
transitions. Callers must prove exclusive writable ownership of the pages.
"""

if __package__:
    from .index_cache import allocate_cache
else:
    from index_cache import allocate_cache


class OwnedIndexCache:
    def __init__(self, torch, geometry, key, scale, table, *, pool, length, budget, forward, reverse):
        self.torch, self.g = torch, geometry
        self.key, self.scale = key, scale
        self.table = table.clone()
        self.pool, self.length, self.budget = int(pool), int(length), int(budget)
        self.buffer = allocate_cache(torch, geometry, key.device, owned=True)
        self.header = self.buffer[:128].view(torch.int32)
        s, c = geometry.resident_capacity, geometry.candidate_capacity
        self.hot_key = self.buffer[128 : 128 + s * 128].reshape(s, 128)
        self.hot_scale = self.buffer[128 + s * 128 : 128 + s * 132].view(torch.float32)
        self.ids = self.buffer[128 + s * 132 : 128 + s * 132 + (s + c) * 4].view(torch.int32)
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
        destinations = self.physical(torch.arange(nc, device=key.device, dtype=torch.int64))
        flat_key.index_copy_(0, destinations, cold_key)
        flat_scale.index_copy_(0, destinations, cold_scale)
        self.ids.fill_(-1)
        self.ids[:nh].copy_(hot.to(torch.int32))
        self.ids[s : s + nc].copy_(cold.to(torch.int32))
        self.positions.fill_(-1)
        self.positions[hot] = -torch.arange(nh, device=key.device, dtype=torch.int32) - 2
        self.positions[cold] = torch.arange(nc, device=key.device, dtype=torch.int32)
        self.header[:8].copy_(
            torch.tensor([1, length, nh, nc, pool, geometry.layer_id, budget, 0], dtype=torch.int32, device=key.device)
        )

    def physical(self, positions):
        return self.table[positions // 128].to(self.torch.int64) * 128 + positions % 128

    def grow(self, length, table=None):
        """Promote only the newly eligible tail rows to cold; no old-bank scan."""
        if not self.length <= length <= self.g.candidate_capacity:
            raise ValueError("Index growth is outside the owned candidate capacity")
        if table is not None:
            self.table = table.clone()
        if length == self.length:
            return
        torch = self.torch
        tokens = torch.arange(self.length, length, device=self.key.device, dtype=torch.int64)
        source = self.physical(tokens)
        # Clone both planes before writing: a very short/empty hot bank can
        # make the appended cold destination overlap the incoming tail range.
        keys = self.key.view(torch.uint8).reshape(-1, 128)[source]
        scales = self.scale.reshape(-1)[source]
        positions = torch.arange(length - self.length, device=self.key.device, dtype=torch.int64) + self.header[3]
        physical = self.physical(positions)
        self.key.view(torch.uint8).reshape(-1, 128).index_copy_(0, physical, keys)
        self.scale.reshape(-1).index_copy_(0, physical, scales)
        self.ids[self.g.resident_capacity + positions] = tokens.to(torch.int32)
        self.positions[tokens] = positions.to(torch.int32)
        self.header[20] = 0  # Invalidate normalized resident/H0 snapshot on layout growth.
        self.header[3].add_(length - self.length)
        self.header[1] = length
        self.length = length

    def logical(self):
        """Temporary logical snapshot; callers must not retain it as a mirror."""
        torch = self.torch
        tokens = torch.arange(self.length, device=self.key.device, dtype=torch.int64)
        pos = self.positions[: self.length].to(torch.int64)
        hot = pos <= -2
        hp = (-pos - 2).clamp(0, self.g.resident_capacity - 1)
        cp = self.physical(pos.clamp(0, self.g.candidate_capacity - 1))
        keys = torch.where(hot[:, None], self.hot_key[hp], self.key.view(torch.uint8).reshape(-1, 128)[cp])
        scales = torch.where(hot, self.hot_scale[hp], self.scale.reshape(-1)[cp])
        return tokens, keys, scales

    def restore_logical(self):
        """Complete before a native reader/writer, then discard this owner."""
        tokens, keys, scales = self.logical()
        physical = self.physical(tokens)
        self.key.view(self.torch.uint8).reshape(-1, 128).index_copy_(0, physical, keys)
        self.scale.reshape(-1).index_copy_(0, physical, scales)
        self.header[20] = 0
        self.header[0] = 0

    def write_logical(self, tokens, keys, scales):
        """Boundary/test writer; production Q4 uses one device routing kernel."""
        torch = self.torch
        tokens = tokens.to(device=self.key.device, dtype=torch.int64)
        pos = self.positions[tokens].to(torch.int64)
        hot = pos <= -2
        self.hot_key.index_copy_(0, -pos[hot] - 2, keys[hot])
        self.hot_scale.index_copy_(0, -pos[hot] - 2, scales[hot])
        cold = ~hot
        physical = self.physical(pos[cold])
        self.key.view(torch.uint8).reshape(-1, 128).index_copy_(0, physical, keys[cold])
        self.scale.reshape(-1).index_copy_(0, physical, scales[cold])
