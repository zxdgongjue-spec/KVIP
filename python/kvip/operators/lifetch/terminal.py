"""Allocate immutable four-Q lease bindings before graph capture."""

import ctypes
import struct

from kvip.operators.lifetch.terminal_queues import ENTITY_BYTES, TEMPLATE_WORDS, bind_queues


class TerminalLeaseBinding:
    """Own capture-lifetime templates and row-major copies of real KVAF args.

    This does not launch the terminal kernel. Live row-owner offsets must be
    applied by that kernel; output_address here is the base of the acquired
    output allocation, not a frozen per-request destination.
    """

    def __init__(self, rotating):
        owner, torch = rotating.owner, rotating.torch
        self.owner = owner
        self.rotating = rotating
        lease = owner.lease
        if lease is None or lease.closed or lease.capturing_startup or lease.capture_fence_failed:
            raise ValueError("Terminal binding requires a live URMA lease outside capture")
        # Setup only: complete earlier use before copying immutable entities.
        stream = lease.check_stream()
        stream.synchronize()
        batch = rotating.runtime_shape.batch
        size = owner.abi_sizes
        if size["args"] != 112 or size["entity"] != ENTITY_BYTES or size["queue"] != 64:
            raise ValueError("Unsupported KVAF argument/entity/queue ABI")
        count = batch * 5
        args = owner.device_args[:count].cpu().contiguous()
        raw = args.numpy().tobytes()
        if len(raw) != count * size["args"]:
            raise ValueError("Insufficient independent terminal queue views")
        fields = [struct.unpack_from("<10Q8I", raw, i * size["args"]) for i in range(count)]
        entity_base = fields[0][8]
        if any(f[8] != entity_base + i * ENTITY_BYTES for i, f in enumerate(fields)):
            raise ValueError("Expected contiguous live entity table")
        snapshot = lease.library.kvip_transport_snapshot
        snapshot.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
        snapshot.restype = ctypes.c_uint32
        entities = torch.empty(count * ENTITY_BYTES, dtype=torch.uint8, device=owner.device)
        code = snapshot(entity_base, entities.data_ptr(), entities.numel(), stream.npu_stream)
        if code:
            raise RuntimeError(f"Terminal entity snapshot failed: {code}")
        entity_bytes = entities.cpu().numpy().tobytes()
        views = [
            dict(
                entity_address=f[8],
                queue_address=f[9],
                output_address=f[6],
                entity_bytes=entity_bytes[i * ENTITY_BYTES : (i + 1) * ENTITY_BYTES],
            )
            for i, f in enumerate(fields)
        ]
        token_bytes = rotating.runtime_shape.tokenBytes
        if any(f[13] != token_bytes for f in fields):
            raise ValueError("Queue token size differs from LIFetch geometry")
        self.bindings = bind_queues(batch, views, token_bytes)
        templates = b"".join(struct.pack("<32I", *binding.template) for binding in self.bindings)
        selected_args = b"".join(
            raw[binding.view * size["args"] : (binding.view + 1) * size["args"]] for binding in self.bindings
        )
        # Allocation, host transfer and snapshot happen once, outside measured
        # graphs. SQ/CQ consumer/producer indices are never copied into templates.
        self.templates = (
            torch.tensor(list(templates), dtype=torch.uint8, device=owner.device)
            .view(torch.int32)
            .reshape(batch, 4, TEMPLATE_WORDS)
        )
        self.args = torch.tensor(list(selected_args), dtype=torch.uint8, device=owner.device).reshape(
            batch, 4, size["args"]
        )
        self.entity_base = entity_base
        self.queue_views = tuple(binding.view for binding in self.bindings)

    def attach_workspace(self, forward):
        """Allocate private request-local publication state before capture.

        This method does not replace forward.function or change LI bindings.
        Launch integration must explicitly consume the returned binding.
        """
        if forward.rotating is not self.rotating:
            raise ValueError("Terminal workspace and forward must share a binding")
        if hasattr(self, "workspace"):
            raise ValueError("Terminal workspace is already attached")
        shape = self.rotating.runtime_shape
        if min(shape.batch, 16) * 4 > shape.aicCount * 2:
            raise ValueError("Four concurrent AIVs per request are required")
        torch, device = self.rotating.torch, self.rotating.device
        self.flags = torch.zeros((shape.batch, 32, 32), dtype=torch.int32, device=device)
        self.times = torch.zeros((shape.batch, 4, 32), dtype=torch.int64, device=device)
        self.logical = torch.empty((shape.batch, 4, 2048), dtype=torch.int32, device=device)
        # Private extension list and four per-Q protection masks. Lifetime
        # matches the captured binding; allocation is outside the graph.
        self.capacity = torch.zeros(
            (shape.batch, 10240 + 4 * ((shape.resident + 31) // 32)), dtype=torch.int32, device=device
        )
        # Large-batch scheduler layout is four 128-byte lines (TerminalRowSchedule).
        self.row_schedule = torch.zeros((4, 32), dtype=torch.int32, device=device) if shape.batch > 16 else None
        self.li_ready = torch.empty((shape.batch, 8, 32), dtype=torch.int32, device=device) if shape.batch > 16 else None
        blob = struct.pack(
            "<6Q2I9Q",
            forward.workspace.data_ptr(),
            self.args.data_ptr(),
            self.templates.data_ptr(),
            self.flags.data_ptr(),
            self.times.data_ptr(),
            self.logical.data_ptr(),
            1,
            0,
            self.capacity.data_ptr(),
            0 if self.row_schedule is None else self.row_schedule.data_ptr(),
            0 if self.li_ready is None else self.li_ready.data_ptr(),
            *([0] * 6),
        )
        if len(blob) != 128:
            raise AssertionError("Terminal runtime ABI packing changed")
        self.workspace = torch.tensor(list(blob), dtype=torch.uint8, device=device)
        self.forward = forward
        return self.workspace
