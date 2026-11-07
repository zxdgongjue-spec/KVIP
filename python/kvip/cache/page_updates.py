# SPDX-License-Identifier: Apache-2.0
"""Publish appended DDR pages across all requests/groups in one upload/launch."""


class PageUpdates:
    def __init__(self, torch, addresses, valid, token_to_slot, *, fused=(), widths=(), token_bytes=656, block_size=128):
        self.torch = torch
        self.addresses, self.valid, self.token_to_slot = addresses, valid, token_to_slot
        self.block_size = block_size
        self.fused = tuple(fused)
        self.widths, self.token_bytes = tuple(widths), token_bytes
        self.on_npu = addresses.device.type == "npu"
        # Each row contains [flat page index, DDR address, old map offset/-1].
        shape = (addresses.numel(), 3)
        self.device = torch.empty(shape, dtype=torch.int64, device=addresses.device)
        self.host = [torch.empty(shape, dtype=torch.int64, pin_memory=self.on_npu) for _ in range(2)]
        self.fused_table = torch.tensor(
            [(t.data_ptr(), width * token_bytes) for t, width in zip(self.fused, self.widths)] or [(0, 0)],
            dtype=torch.int64,
            device=addresses.device,
        )

    @property
    def nbytes(self):
        return self.device.numel() * self.device.element_size() + self.fused_table.numel() * 8

    def submit(self, rows, ring):
        if not rows:
            return
        count = len(rows)
        host = self.host[ring][:count]
        host.numpy()[:] = rows
        self.device[:count].copy_(host, non_blocking=self.on_npu)
        if self.on_npu:
            from kvip.cache._page_updates_kernel import publish_pages_kernel

            publish_pages_kernel[(count,)](
                self.device,
                self.addresses,
                self.valid,
                self.token_to_slot,
                self.fused_table,
                GROUPS=self.addresses.shape[1],
                PAGES=self.addresses.shape[2],
                BLOCK=self.block_size,
                FUSED=bool(self.fused),
                HISTORY=self.fused[0].shape[1] if self.fused else 0,
            )
            return
        # CPU reference; the production path above has one copy and one launch.
        for index, address, clear in rows:
            owner, rem = divmod(index, self.addresses.shape[1] * self.addresses.shape[2])
            group, page = divmod(rem, self.addresses.shape[2])
            self.addresses.view(-1)[index] = address
            self.valid.view(-1)[index] = 1
            if clear >= 0:
                self.token_to_slot.view(-1)[clear : clear + self.block_size] = -1
            if self.fused:
                offsets = self.torch.arange(self.block_size, dtype=self.torch.int64)
                self.fused[group][owner, page * self.block_size : (page + 1) * self.block_size] = (
                    address + offsets * self.widths[group] * self.token_bytes
                )
