# SPDX-License-Identifier: Apache-2.0
"""Graph-stable original LI work buffers shared by sequential layer groups."""

from .fused_build import load_original
from .fused_inputs import FusedQueryInputs


class FusedWorkspace:
    def __init__(self, torch, *, batch, history, aics, windows, device, pin_memory=True, index_heads=64):
        if history % 256:
            raise ValueError("Original LIFetch history must align to 256 tokens")
        geometry = load_original("geometry").li_geometry(batch, "auto", windows, aics)
        self.inputs = FusedQueryInputs(
            torch, batch, device, pin_memory=pin_memory, history=history, index_heads=index_heads
        )
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
        geometry = load_original("geometry")
        shape = geometry.li_geometry(batch, "auto", windows, aics)
        final = geometry.final_aivs(batch, "auto", aics)
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
            changes=(batch, 2048),
            row_status=(batch, 32),
        )
        self.scratch = {
            name: torch.zeros(size, dtype=torch.int32, device=self.topk.device) for name, size in shapes.items()
        }
        self.scratch["events"] = torch.zeros(
            (max(1, shape["intermediate_fetches"]), shape["service_aivs_per_window"], 128),
            dtype=torch.uint8,
            device=self.topk.device,
        )
        self.scratch["final_status"] = torch.zeros(
            (batch * 2, status_bytes), dtype=torch.uint8, device=self.topk.device
        )
        self.scratch["fallback_addresses"] = torch.zeros((batch, 8192), dtype=torch.int64, device=self.topk.device)

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
            raise ValueError("Index storage differs from the compiled LI workspace")
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
