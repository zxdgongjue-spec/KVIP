# SPDX-License-Identifier: Apache-2.0
"""Streaming LIFetch workspace and launch, allocated before graph capture."""

import ctypes

import numpy as np


class StreamingLIFetchForward:
    def __init__(self, rotating, update, torch, prefetch_limit, *, fused_final=False, q_workers=None):
        if rotating.device != update.device:
            raise ValueError("Fetch and Index update must use the same device")
        if update.bindings.numel() != 28:
            raise ValueError("LIFetch requires the 28-field Index update/output binding ABI")
        self.rotating, self.update, self.torch = rotating, update, torch
        shape = rotating.runtime_shape
        self.batch, self.history, self.resident = shape.batch, shape.history, shape.resident
        self.shape_pointer = ctypes.byref(shape)
        if not 1 <= self.batch <= 16:
            raise ValueError(f"LIFetch supports BS1..16; received BS={self.batch}; no legacy fallback")
        if not (
            2048 <= self.resident <= 131072
            and self.resident < self.history <= 210944
            and self.history % 256 == 0
            and self.resident % 256 == 0
            and shape.indexHeads in (32, 64)
        ):
            raise ValueError("Unsupported LIFetch history/resident or Index head geometry")
        if not 0 <= prefetch_limit <= 2048:
            raise ValueError("LIFetch prefetch budget must be in 0..2048")
        keys, scales, addresses, logical = update.storage[:4]
        b, device = self.batch, rotating.device
        self.fused_final = fused_final
        if q_workers not in (1, 2, 4) or not fused_final:
            raise ValueError("LIFetch requires fused completion and 1/2/4 Fetch workers per query")
        self.q_workers = q_workers
        self.prefetch_limit = prefetch_limit
        self.legacy_layout = False
        self.parallel_finalize = False
        self.prepare_metadata = False
        geometry_fn = rotating.library.kvip_streaming_resolver_cores
        geometry_fn.argtypes, geometry_fn.restype = [], ctypes.c_uint32
        flags_fn = rotating.library.kvip_streaming_flag_count
        flags_fn.argtypes, flags_fn.restype = [], ctypes.c_uint32
        self.resolver_cores, flag_count = int(geometry_fn()), int(flags_fn())
        if self.resolver_cores != 3 or flag_count != 16:
            raise ValueError("Rebuild KVIP: the public LIFetch pipeline requires the shared three-role ABI")
        self.function = rotating.library.kvip_lifetch_streaming_forward
        if b * self.resolver_cores + 4 * q_workers > shape.aicCount * 2:
            raise ValueError("LIFetch resolver and Fetch workers exceed the concurrently resident AIV budget")
        if shape.finalWorkers != 4:
            raise ValueError("Streaming LIFetch requires four query publication records per request")
        # Layer groups execute on the same lease stream. Reuse temporary
        # tensors across groups, retaining a distinct immutable Args per group.
        pool = getattr(rotating, "streaming_scratch", {})

        def allocate(name, shape, dtype, zero=False):
            key = (name, tuple(shape), dtype, str(device))
            if key not in pool:
                factory = torch.zeros if zero else torch.empty
                pool[key] = factory(shape, dtype=dtype, device=device)
            return pool[key]

        self.flags = allocate("flags", (b, flag_count, 32), torch.int32, True)
        self.events_src = allocate("event_sources", (b, 10240), torch.int32)
        self.events_dst = allocate("event_destinations", (b, 10240), torch.int32)
        self.worker_status = allocate("worker_status", (b, 32), torch.int32, True)
        self.times = allocate("times", (b * 256 + (self.resolver_cores * b + 4 * q_workers) * 32,), torch.int64, True)
        self.query_bitmaps = allocate("query_bitmaps", (b, 4, (self.resident + 31) // 32), torch.int32)
        self.query_errors = allocate("query_errors", (b, 4, 32), torch.int32, True)
        self.dense_shadow = allocate("dense_shadow", (b, 3 * self.history), torch.int64)
        self.packed_fetch = allocate(
            "packed_fetch", (4 * q_workers, 9 * b * 2048 + ((3 * b + 31) // 32) * 32 + 64), torch.int32
        )
        pointers = [
            rotating.context.data_ptr(),
            keys.data_ptr(),
            scales.data_ptr(),
            addresses.data_ptr(),
            logical.data_ptr(),
            rotating.hot_valid.data_ptr(),
            self.flags.data_ptr(),
            self.events_src.data_ptr(),
            self.events_dst.data_ptr(),
            self.worker_status.data_ptr(),
            self.times.data_ptr(),
        ]
        geometry = [self.history, self.resident, prefetch_limit | ((q_workers or 0) << 16), 2048 + 8192]
        trailing = [self.query_bitmaps.data_ptr(), self.query_errors.data_ptr(), update.bindings.data_ptr()]
        blob = np.concatenate(
            (
                np.asarray(pointers, dtype="<u8").view("u1"),
                np.asarray(geometry, dtype="<u4").view("u1"),
                np.asarray(trailing, dtype="<u8").view("u1"),
            )
        )
        blob = np.concatenate(
            (
                blob,
                np.asarray([self.dense_shadow.data_ptr(), self.packed_fetch.data_ptr()], dtype="<u8").view("u1"),
                np.zeros(112, dtype="u1"),
            )
        )
        if len(blob) != 256:
            raise AssertionError(f"LIFetch workspace ABI size={len(blob)}, expected 256")
        self.workspace = torch.from_numpy(blob.copy()).to(device)
        self.bindings = torch.tensor(
            [x.data_ptr() for x in rotating.arguments] + [self.workspace.data_ptr()],
            dtype=torch.uint64,
            device=device,
        )
        # Standalone LI calls use the most recently attached workspace. Each
        # fused call keeps its own LI and resolver workspace paired.
        rotating.bindings = self.bindings
        self.function.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_uint32]
        self.function.restype = ctypes.c_uint32

    def __call__(self):
        stream = (
            self.rotating.lease.check_stream()
            if hasattr(self.rotating, "lease")
            else self.torch.npu.current_stream(self.rotating.device)
        )
        args = [
            self.bindings.data_ptr(),
            self.update.bindings.data_ptr(),
            self.workspace.data_ptr(),
            stream.npu_stream,
            self.shape_pointer,
        ]
        args.append(self.q_workers)
        result = self.function(*args)
        if result:
            raise RuntimeError(f"streaming LIFetch launch failed: {result}")

    def set_options(self, *, diagnostics=False, deferred_fetch=False):
        """Startup/profile-only update; forward never allocates or copies."""
        value = (
            self.prefetch_limit
            | ((self.q_workers or 0) << 16)
            | (0x20000000 if self.legacy_layout else 0)
            | (0x10000000 if self.parallel_finalize else 0)
            | (0x08000000 if self.prepare_metadata else 0)
            | (0x04000000 if getattr(self, "ubmem_overlap_fetch", False) else 0)
            | (0x80000000 if diagnostics else 0)
            | (0x40000000 if deferred_fetch else 0)
        )
        encoded = np.asarray([value], dtype="<u4").view("u1")
        self.workspace[96:100].copy_(self.torch.from_numpy(encoded.copy()))
