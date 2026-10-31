"""CPU-only, exact nine-argument KVAF consumer fake with local payload copies."""

from dataclasses import dataclass

from kvip.v9.api import LayoutSpec


@dataclass(frozen=True)
class TimingContext:
    capturing: bool = False


class FakeNpuRuntime:
    """Deterministic test provider; registration is an explicit address->CPU block map.

    No Torch import occurs until the caller supplies it. Never dereference an
    arbitrary address or perform device scalar readback. zero_task checks the
    invocation only and leaves output unchanged.
    """

    def __init__(
        self,
        *,
        torch,
        batch_get_token_bytes,
        batch_get_tokens_per_block,
        device_id=0,
        npu_core_count=1,
        batch_get_kernel_object_path="test-only",
        batch_put_kernel_object_path="test-only",
        jettys_per_core=2,
        timing=None,
        zero_task=False,
    ):
        layout = LayoutSpec()
        if (batch_get_token_bytes, batch_get_tokens_per_block) != (
            layout.token_bytes,
            layout.block_tokens,
        ):
            raise ValueError("fake layout must match the public byte/block contract")
        self.torch = torch
        self.layout = layout
        self.timing = TimingContext() if timing is None else timing
        self.zero_task = zero_task
        self.blocks = {}
        self.calls = 0

    def register_block(self, address, block):
        if not isinstance(address, int) or address <= 0 or address in self.blocks:
            raise ValueError("registered block needs a unique positive address")
        self._tensor(block, self.torch.uint8)
        if tuple(block.shape) != (self.layout.block_tokens, self.layout.token_bytes):
            raise ValueError("registered payload must be one complete CPU block")
        self.blocks[address] = block

    def _tensor(self, value, dtype):
        if (
            value.device.type != "cpu"
            or not value.is_contiguous()
            or value.dtype != dtype
        ):
            raise ValueError(
                "fake requires contiguous CPU tensors of the declared dtype"
            )

    def batch_get(
        self,
        address_table,
        valid_table,
        indices,
        offsets,
        total_miss_count,
        output_indices,
        output_rows,
        stream,
        synchronize_batch_get,
    ):
        if self.timing.capturing:
            raise RuntimeError("CPU fake must not run in capture/replay")
        torch, layout = self.torch, self.layout
        for value, dtype in (
            (address_table, torch.int64),
            (valid_table, torch.uint8),
            (indices, torch.int32),
            (offsets, torch.int32),
            (total_miss_count, torch.int32),
            (output_indices, torch.int32),
            (output_rows, torch.uint8),
        ):
            self._tensor(value, dtype)
        if (
            tuple(total_miss_count.shape) != (1,)
            or address_table.ndim != 1
            or valid_table.shape != address_table.shape
        ):
            raise ValueError("invalid directory/count shape")
        if any(value.ndim != 1 for value in (indices, offsets, output_indices)):
            raise ValueError("descriptors must be flat global prefixes")
        if output_rows.ndim != 2 or output_rows.shape[1] != layout.token_bytes:
            raise ValueError("output must be byte rows")
        if type(synchronize_batch_get) is not bool:
            raise ValueError("synchronize_batch_get must be bool")
        count = int(total_miss_count[0])  # CPU-only, never a device hot-path read.
        if count < 0 or count > min(
            indices.numel(), offsets.numel(), output_indices.numel()
        ):
            raise ValueError("count exceeds global descriptor capacity")
        copies = []
        for i in range(count):
            directory, offset, target = (
                int(indices[i]),
                int(offsets[i]),
                int(output_indices[i]),
            )
            if not 0 <= directory < address_table.numel() or not int(
                valid_table[directory]
            ):
                raise ValueError("invalid directory index")
            if (
                offset < 0
                or offset % layout.token_bytes
                or offset >= layout.block_tokens * layout.token_bytes
            ):
                raise ValueError("unaligned or out-of-block byte offset")
            if not 0 <= target < output_rows.shape[0]:
                raise ValueError("physical output row out of range")
            address = int(address_table[directory]) & ((1 << 64) - 1)
            if address not in self.blocks:
                raise ValueError("unregistered source block")
            copies.append((target, self.blocks[address][offset // layout.token_bytes]))
        # Validate the complete invocation before any writes; clones permit aliases
        # between registered test sources and the local output without order effects.
        if not self.zero_task:
            copies = [(target, source.clone()) for target, source in copies]
            for target, source in copies:
                output_rows[target].copy_(source)
        self.calls += 1
