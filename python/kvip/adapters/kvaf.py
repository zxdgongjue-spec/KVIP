# SPDX-License-Identifier: Apache-2.0
"""One stream-ordered KVAF lease for fused decode and ordinary Get graphs."""

import ctypes
from kvip.runtime.loader import load_lifetch_library, load_native_get
from contextlib import contextmanager


class LeasedRuntime:
    """Preserve Runtime calls while routing Get through its existing lease.

    Descriptor bindings are cached by storage address, shape and dtype, not
    Python tensor identity: service callers often pass fresh views each round.
    Cached plans retain the underlying storage for the lease lifetime.
    """

    def __init__(self, lease):
        self.lease = lease
        self.bindings = {}

    def __getattr__(self, name):
        return getattr(self.lease.runtime, name)

    def batch_get(self, *args, synchronize=False):
        if len(args) == 6:
            table, *descriptors = args
            args = (table.addresses, table.valid, *descriptors)
        if len(args) != 7:
            raise TypeError("BatchGet requires a table and five descriptors, or seven tensors")
        self.lease.check_stream()
        key = tuple((t.data_ptr(), tuple(t.shape), t.dtype) for t in args)
        plan = self.bindings.get(key)
        if plan is None:
            plan = self.lease.prepare_get(*args)
            self.bindings[key] = plan
        plan()
        if synchronize:
            self.lease.check_stream().synchronize()

    def shutdown(self):
        # Release must fence outstanding DMA before Runtime destroys queues.
        self.lease.close()
        self.bindings.clear()
        self.lease.runtime.shutdown()


class GetLaunch:
    """Prepared native Get: replay only enqueues a kernel, with no H2D or wait."""

    def __init__(self, owner, arguments, tensors):
        self.owner, self.arguments, self.tensors = owner, arguments, tensors

    def __call__(self):
        owner = self.owner
        stream = owner.check_stream()
        code = owner.get(self.arguments, stream.npu_stream)
        if code:
            raise RuntimeError(f"Leased BatchGet launch failed: {code}")


class KvafGetLease:
    """Setup/teardown owner, not a second Runtime or DDR allocation.

    Register outputs and prepare descriptor bindings before graph capture.
    The caller keeps graph lifetimes inside this object's lifetime. Queue
    cursors belong to KVAF and must never be cleared between replays.
    """

    def __init__(self, torch, runtime, library, output, stream):
        self.torch, self.runtime, self.stream = torch, runtime, stream
        self.library = load_lifetch_library(library)
        self.sizes = {}
        for name in ("args", "status", "entity", "queue"):
            fn = getattr(self.library, f"refactor_fetch_{name}_bytes")
            fn.argtypes, fn.restype = [], ctypes.c_uint32
            self.sizes[name] = int(fn())
        self.pack = self.library.refactor_fetch_pack_args
        self.pack.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32]
        self.pack.restype = ctypes.c_uint32
        self.native_library, self.get = load_native_get(runtime)
        self.context = runtime.acquire_fused_batch_get_context(output, stream)
        self.device = output.device
        self.outputs = {self.output_key(output): (output, self.context)}
        self.plans = []
        self.closed = False
        self.capturing_startup = False
        self.capture_fence_failed = False

    @staticmethod
    def output_key(output):
        return output.data_ptr(), output.numel() * output.element_size()

    def check_stream(self):
        if self.closed:
            raise RuntimeError("Fused Get lease has been released")
        stream = self.torch.npu.current_stream(self.device)
        if stream.npu_stream != self.stream.npu_stream and not self.capturing_startup:
            raise RuntimeError("Fused and ordinary Get must use the lease stream")
        return stream

    @contextmanager
    def capture_scope(self):
        """Startup-only auxiliary streams, fenced before normal inference.

        vLLM captures on a separate stream, then replays on its execution
        stream. No wait or event is inserted into the decode hot path.
        """
        self.check_stream()
        if self.capturing_startup:
            raise RuntimeError("Nested fused capture scopes are unsupported")
        self.torch.npu.synchronize(self.device)
        self.capturing_startup = True
        try:
            yield
        finally:
            try:
                self.torch.npu.synchronize(self.device)
            except BaseException:
                self.capture_fence_failed = True
                raise
            finally:
                self.capturing_startup = False

    def register_output(self, output):
        self.check_stream()
        key = self.output_key(output)
        if key not in self.outputs:
            view = self.runtime.register_fused_batch_get_output(self.context, output)
            self.outputs[key] = output, view
        return self.outputs[key][1]

    def serialize(self, fields):
        blob = ctypes.create_string_buffer(self.sizes["args"])
        values = (ctypes.c_uint64 * len(fields))(*fields)
        code = self.pack(blob, len(blob), values, len(fields))
        if code:
            raise ValueError(f"Invalid KVAF argument fields: {code}")
        return blob.raw

    def prepare_get(self, addresses, valid, indices, offsets, count, destinations, output, *, max_inflight=2048):
        self.check_stream()
        tensors = addresses, valid, indices, offsets, count, destinations, output
        if any(t.device != self.device or not t.is_contiguous() for t in tensors):
            raise ValueError("Leased Get requires contiguous tensors on the Runtime device")
        if indices.numel() != offsets.numel() or indices.numel() != destinations.numel() or count.numel() != 1:
            raise ValueError("Get descriptor capacities differ or count is not scalar")
        if addresses.numel() != valid.numel():
            raise ValueError("Get address and validity tables differ")
        if addresses.dtype not in (self.torch.int64, self.torch.uint64) or valid.dtype != self.torch.uint8:
            raise ValueError("Get address/valid tables must be int64 or uint64 / uint8")
        if any(t.dtype != self.torch.int32 for t in (indices, offsets, count, destinations)):
            raise ValueError("Get descriptors and count must be int32")
        context = self.register_output(output)
        if type(max_inflight) is not int or max_inflight < 0 or (0 < max_inflight < 2 * context["worker_count"]):
            raise ValueError("Get inflight budget must be zero or at least twice the worker count")
        fields = [t.data_ptr() for t in tensors[:6]] + [
            output.data_ptr(),
            context["device_core_status"],
            context["device_channel_entity"],
            context["device_queue_state"],
            addresses.numel(),
            indices.numel(),
            context["output_bytes"] // context["token_bytes"],
            context["token_bytes"],
            context["tokens_per_block"],
            context["worker_count"],
            context["jettys_per_core"],
            max_inflight,
        ]
        # Immutable host launch arguments; all mutable counts/descriptors remain
        # device pointers. ACL copies this struct at enqueue/capture time, so
        # replay observes current device data without H2D or a host count read.
        # Use KVAF's compiled kernel and launch attributes, not a recompiled
        # copy inheriting LI's manual-sync compiler flags.
        arguments = ctypes.create_string_buffer(self.serialize(fields))
        plan = GetLaunch(self, arguments, tensors)
        self.plans.append(plan)
        return plan

    def close(self):
        if not self.closed:
            if self.capture_fence_failed:
                raise RuntimeError("Fused capture stream completion failed; transport remains retained")
            # Native release fences this stream and validates persistent queue
            # status. On failure keep every tensor and registration alive.
            self.runtime.release_fused_batch_get_context(self.context, self.stream)
            self.closed = True
            self.plans.clear()
            self.outputs.clear()


class KvafQueueView:
    """Original LIFetch owner interface backed by a service Runtime lease.

    The original launcher takes one single-jetty argument view per physical AIV.
    This adapter only slices the existing channel/queue arrays; it does not
    construct another Runtime, allocate DDR, reset queues, or import a fixture.
    """

    def __init__(self, lease, output, addresses, valid):
        lease.check_stream()
        self.lease, self.output = lease, output
        self.addresses, self.valid = addresses, valid
        self.device, self.abi_sizes = lease.device, lease.sizes
        self.context = lease.register_output(output)
        if self.context["jettys_per_core"] != 2:
            raise ValueError("LIFetch requires two jettys per KVAF worker")
        if addresses.device != self.device or valid.device != self.device:
            raise ValueError("Fused queue address tables must be on the Runtime device")
        if not addresses.is_contiguous() or not valid.is_contiguous() or addresses.numel() != valid.numel():
            raise ValueError("Fused queue address/valid tables must be contiguous with matching capacity")
        torch = lease.torch
        if addresses.dtype not in (torch.int64, torch.uint64) or valid.dtype != torch.uint8:
            raise ValueError("Fused queue address/valid tables must be int64 or uint64 / uint8")
        queues = self.context["worker_count"] * 2
        self.core_status = torch.zeros((queues, self.abi_sizes["status"]), dtype=torch.uint8, device=self.device)
        self.descriptors = torch.zeros((4, 8192), dtype=torch.int32, device=self.device)
        self.device_args = None
        lease.plans.append(self)

    def build_device_args(self, library):
        """Allocate/serialize once before graph capture; retained for all replays."""
        self.lease.check_stream()
        if self.device_args is not None:
            return self.device_args
        context, sizes = self.context, self.abi_sizes
        stride = self.descriptors.stride(0) * self.descriptors.element_size()
        blobs = []
        for queue in range(context["worker_count"] * 2):
            fields = [
                self.addresses.data_ptr(),
                self.valid.data_ptr(),
                *[self.descriptors.data_ptr() + row * stride for row in range(4)],
                self.output.data_ptr(),
                self.core_status.data_ptr() + queue * sizes["status"],
                context["device_channel_entity"] + queue * sizes["entity"],
                context["device_queue_state"] + queue * sizes["queue"],
                self.addresses.numel(),
                8192,
                context["output_bytes"] // context["token_bytes"],
                context["token_bytes"],
                context["tokens_per_block"],
                1,
                1,
                0,
            ]
            blobs.append(list(self.lease.serialize(fields)))
        self.device_args = self.lease.torch.tensor(blobs, dtype=self.lease.torch.uint8, device=self.device)
        return self.device_args
