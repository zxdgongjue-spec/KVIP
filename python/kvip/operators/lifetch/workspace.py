# SPDX-License-Identifier: Apache-2.0
"""Graph-stable original LI work buffers shared by sequential layer groups."""

from kvip.operators.lifetch import geometry as tiling_geometry
from kvip.adapters.vllm.inputs import QueryInputs


def rotation_spec(batch, history, resident, final, rounds, lanes, status_bytes):
    """Single source of truth for the persistent-address scratch ABI."""
    shapes = dict(
        progress=(batch, 4, 32),
        rows=(batch, 256),
        mapping=(batch, history),
        locations=(batch, 4, 2048),
        owners=(batch * 32 + batch * final * 320,),
        blocks=(batch * 8192,),
        offsets=(batch * 8192,),
        destinations=(batch * 8192,),
        count=(1,),
        lru_slots=(batch, 2048),
        reserved_hot=(batch, resident),
        changes=(batch, 10240),
        change_slots=(batch, 10240),
        change_count=(batch,),
        row_status=(batch, 32),
    )
    spec = {name: (shape, "int32") for name, shape in shapes.items()}
    spec["events"] = ((max(1, rounds), lanes, 128), "uint8")
    spec["final_status"] = ((batch * 2, status_bytes), "uint8")
    spec["fallback_addresses"] = ((batch, 8192), "int64")
    return spec


class LIFetchWorkspace:
    def __init__(self, torch, *, batch, history, aics, windows, device, pin_memory=True, index_heads=64):
        if history % 256:
            raise ValueError("Original LIFetch history must align to 256 tokens")
        geometry = tiling_geometry.li_geometry(batch, "auto", windows, aics)
        self.inputs = QueryInputs(torch, batch, device, pin_memory=pin_memory, history=history, index_heads=index_heads)
        self.query_lengths = torch.full((batch,), 4, dtype=torch.int32, device=device)
        self.key_lengths = torch.full((batch,), history, dtype=torch.int32, device=device)
        self.topk = torch.full((batch, 4, 2048), -1, dtype=torch.int32, device=device)
        self.dummy = torch.zeros(1, dtype=torch.uint8, device=device)
        self.tail_bounds = torch.zeros((batch, 2), dtype=torch.int32, device=device)
        # These sizes follow the original batch_inputs / prefix-storage ABI.
        self.compute = torch.empty(
            max(16 * 1024 * 1024, geometry["rotating_aic"] * 4 * history * 4 + 1024 * 1024),
            dtype=torch.uint8,
            device=device,
        )
        prefix_bytes = 4 * 2048 * 6 + 1024 + 4 * history * 2
        tail_bytes = history * 8 + 8192 * 64 + 4 * 128
        self.prefix_storage = torch.zeros(batch * (prefix_bytes + tail_bytes), dtype=torch.uint8, device=device)
        self.prefix_reset = self.prefix_storage[: batch * prefix_bytes]
        self.scratch = {}

    def allocate_rotation(self, torch, *, batch, history, resident, aics, windows, status_bytes=1216):
        """Reserve reusable resolver work memory before HBM profiling.

        Groups execute serially on the same stream. Their ABI bindings point
        at these buffers; only persistent Index residency is group-specific.
        """
        if self.scratch:
            raise RuntimeError("Rotation workspace already allocated")
        geometry = tiling_geometry
        shape = geometry.li_geometry(batch, "auto", windows, aics)
        final = geometry.final_aivs(batch, "auto", aics)
        spec = rotation_spec(
            batch,
            history,
            resident,
            final,
            shape["intermediate_fetches"],
            shape["service_aivs_per_window"],
            status_bytes,
        )
        self.scratch = {
            name: torch.zeros(size, dtype=getattr(torch, dtype), device=self.topk.device)
            for name, (size, dtype) in spec.items()
        }

    @property
    def nbytes(self):
        scratch_bytes = sum(t.numel() * t.element_size() for t in self.scratch.values())
        return (
            self.inputs.nbytes
            + scratch_bytes
            + sum(
                t.numel() * t.element_size()
                for t in (
                    self.query_lengths,
                    self.key_lengths,
                    self.topk,
                    self.dummy,
                    self.compute,
                    self.prefix_storage,
                    self.tail_bounds,
                )
            )
        )

    def arguments(self, storage):
        if storage.history != int(self.block_history):
            raise ValueError("Index storage differs from the bound LI workspace")
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

    @property
    def block_history(self):
        return self.inputs.block_table.shape[1] * 128
