# SPDX-License-Identifier: Apache-2.0
"""Graph-stable LI buffers and UBmem copy counters, allocated before profiling."""

from kvip.adapters.vllm.inputs import QueryInputs
from kvip.operators.lifetch.geometry import final_aivs, li_geometry


class UbmemQueryInputs(QueryInputs):
    def prepare(self, query_counts, *, query_starts=None, query_owners=None):
        counts = tuple(query_counts)
        owners = None if query_owners is None else tuple(query_owners)
        if owners is not None:
            active = [owner for owner, count in zip(owners, counts) if count > 0]
            if len(active) != len(set(active)):
                raise ValueError("UBmem active rows must use distinct stable owners")
        return super().prepare(counts, query_starts=query_starts, query_owners=owners)


def rotation_spec(batch, history, resident, aics, windows):
    shape = li_geometry(batch, "auto", windows, aics)
    final = final_aivs(batch, "auto", aics)
    spec = {
        "progress": ((batch, 4, 32), "int32"),
        "rows": ((batch, 256), "int32"),
        "mapping": ((batch, history), "int32"),
        "locations": ((batch, 4, 2048), "int32"),
        "owners": ((batch * 32 + batch * final * 320,), "int32"),
        "blocks": ((batch * 8192,), "int32"),
        "offsets": ((batch * 8192,), "int32"),
        "destinations": ((batch * 8192,), "int32"),
        "count": ((1,), "int32"),
        "lru_slots": ((batch, 2048), "int32"),
        "reserved_hot": ((batch, resident), "int32"),
        "changes": ((batch, 2048), "int32"),
        "row_status": ((batch, 32), "int32"),
        "events": ((shape["intermediate_fetches"], shape["service_aivs_per_window"], 128), "uint8"),
        "fallback_addresses": ((batch, 8192), "int64"),
        "ubmem_status": ((batch, 16), "uint64"),
    }
    return spec


class UbmemWorkspace:
    def __init__(self, torch, *, batch, history, resident, aics, windows, device, index_heads=64, pin_memory=True):
        geometry = li_geometry(batch, "auto", windows, aics)
        self.inputs = UbmemQueryInputs(
            torch, batch, device, pin_memory=pin_memory, history=history, index_heads=index_heads
        )
        self.query_lengths = torch.full((batch,), 4, dtype=torch.int32, device=device)
        self.key_lengths = torch.full((batch,), history, dtype=torch.int32, device=device)
        self.topk = torch.full((batch, 4, 2048), -1, dtype=torch.int32, device=device)
        self.dummy = torch.zeros(1, dtype=torch.uint8, device=device)
        self.tail_bounds = torch.zeros((batch, 2), dtype=torch.int32, device=device)
        self.compute = torch.empty(
            max(16 * 1024 * 1024, geometry["rotating_aic"] * 4 * history * 4 + 1024 * 1024),
            dtype=torch.uint8,
            device=device,
        )
        prefix_bytes = 4 * 2048 * 6 + 1024
        tail_bytes = 4 * history * 2 + history * 8 + 8192 * 64 + 4 * 128
        self.prefix_storage = torch.zeros(batch * (prefix_bytes + tail_bytes), dtype=torch.uint8, device=device)
        self.prefix_reset = self.prefix_storage[: batch * (prefix_bytes + 4 * history * 2)]
        self.scratch = {
            name: torch.zeros(size, dtype=getattr(torch, dtype), device=device)
            for name, (size, dtype) in rotation_spec(batch, history, resident, aics, windows).items()
        }

    @property
    def nbytes(self):
        tensors = (
            self.query_lengths,
            self.key_lengths,
            self.topk,
            self.dummy,
            self.tail_bounds,
            self.compute,
            self.prefix_storage,
            *self.scratch.values(),
        )
        return self.inputs.nbytes + sum(t.numel() * t.element_size() for t in tensors)

    def arguments(self, storage):
        if storage.history != self.inputs.block_table.shape[1] * 128:
            raise ValueError("Index storage differs from the UBmem LI workspace")
        return (
            self.inputs.query,
            storage.keys,
            self.inputs.weights,
            self.inputs.scale,
            storage.scales,
            self.query_lengths,
            self.key_lengths,
            self.inputs.block_table,
            self.dummy,
            self.topk,
            self.dummy,
            self.dummy,
            self.compute,
            self.prefix_storage,
        )
