"""Graph-safe V9 launches. Allocations/reset/readback stay outside timing."""

import ctypes
from kvip.runtime.loader import load_lifetch_library

import numpy as np

from kvip.operators.lifetch.geometry import final_aivs, li_geometry, prefetch_budget
from kvip.operators.lifetch.tiling import RuntimeShape
from kvip.operators.lifetch.workspace import rotation_spec


class LIFetchBinding:
    def __init__(
        self,
        torch,
        library,
        arguments,
        owner,
        *,
        batch,
        history,
        resident,
        prefetch_limit=512,
        window_limit=None,
        li_tiling="auto",
        intermediate_fetches=32,
        aic_count=32,
        service_aics=0,
        final_workers=0,
        final_strategy="replicated",
        token_bytes=656,
        query_ends=None,
        hot_valid=None,
        row_owners=None,
        token_addresses=None,
        tail_bounds=None,
        output_stride=None,
        scratch=None,
        parallel_final=False,
        async_final=False,
        cold_only_prefetch=False,
        threshold_only=False,
        simt_init=False,
        staged_outputs=False,
    ):
        if resident > 131072:
            raise ValueError("LIFetch hot bitmap supports resident <= 131072")
        self.torch, self.owner = torch, owner
        self.library = load_lifetch_library(library)
        self.device = arguments[0].device
        if (
            len(arguments) != 14
            or self.device.type != "npu"
            or any(x.device != self.device or not x.is_contiguous() for x in arguments)
        ):
            raise ValueError("V9 requires 14 contiguous input buffers on the same NPU")
        self.runtime_shape = RuntimeShape.prepare(
            self.library,
            batch=batch,
            history=history,
            resident=resident,
            index_heads=arguments[0].numel() // (batch * 4 * 128),
            aic_count=aic_count,
            windows=intermediate_fetches,
            split=li_tiling == "auto",
            service_aics=service_aics,
            final_workers=final_workers,
            token_bytes=token_bytes,
        )
        self.shape_pointer = ctypes.byref(self.runtime_shape)
        if type(parallel_final) is not bool:
            raise ValueError("parallel_final must be a boolean")
        self.parallel_final = parallel_final
        if parallel_final or hasattr(self.library, "v9_parallel_final"):
            mode = self.library.v9_parallel_final
            mode.argtypes, mode.restype = [ctypes.c_void_p], ctypes.c_uint32
            if int(mode(self.shape_pointer)) != int(parallel_final):
                raise ValueError("V9 binary contract mismatch: parallel_final")
        if type(async_final) is not bool or (async_final and not parallel_final):
            raise ValueError("async_final must be a boolean and requires parallel_final")
        self.async_final = async_final
        if async_final or hasattr(self.library, "v9_async_final"):
            mode = self.library.v9_async_final
            mode.argtypes, mode.restype = [ctypes.c_void_p], ctypes.c_uint32
            if int(mode(self.shape_pointer)) != int(async_final):
                raise ValueError("V9 binary contract mismatch: async_final")
        for name, expected in (
            ("cold_only_prefetch", cold_only_prefetch),
            ("threshold_only", threshold_only),
            ("simt_init", simt_init),
            ("staged_outputs", staged_outputs),
        ):
            if type(expected) is not bool:
                raise ValueError(f"{name} must be a boolean")
            setattr(self, name, expected)
            if expected or hasattr(self.library, "v9_" + name):
                mode = getattr(self.library, "v9_" + name)
                mode.argtypes, mode.restype = [ctypes.c_void_p], ctypes.c_uint32
                if int(mode(self.shape_pointer)) != int(expected):
                    raise ValueError(f"V9 binary contract mismatch: {name}")
        for name, expected in (
            ("token_bytes", token_bytes),
            ("final_replicated", int(final_strategy == "replicated")),
            ("final_workers", final_aivs(batch, li_tiling, aic_count, final_workers)),
            ("aic_count", aic_count),
            ("abi_version", 11),
            (
                "service_lanes",
                li_geometry(batch, li_tiling, intermediate_fetches, aic_count, service_aics)["service_aivs_per_window"],
            ),
            (
                "rounds",
                li_geometry(batch, li_tiling, intermediate_fetches, aic_count, service_aics)["intermediate_fetches"],
            ),
            ("li_cores", li_geometry(batch, li_tiling, intermediate_fetches, aic_count, service_aics)["rotating_aic"]),
            ("batch", batch),
            ("history", history),
            ("resident", resident),
            ("simulation", 0),
            ("context_bytes", 320),
            ("row_bytes", 1024),
        ):
            fn = getattr(self.library, "v9_" + name)
            fn.argtypes, fn.restype = [ctypes.c_void_p], ctypes.c_uint32
            if fn(self.shape_pointer) != expected:
                raise ValueError(f"V9 binary contract mismatch: {name}")
        prefetch_limit, window_limit = prefetch_budget(prefetch_limit, window_limit, intermediate_fetches)
        geometry = li_geometry(batch, li_tiling, intermediate_fetches, aic_count, service_aics)
        if geometry["logical_aic_tasks"] > aic_count or not geometry["intermediate_fetches"]:
            raise ValueError("Direct hot-first mode requires a window and concurrently resident LI tasks")
        contexts = owner.build_device_args(library)
        required_queues = batch * (final_aivs(batch, li_tiling, aic_count, final_workers) + 1)
        if contexts.shape[0] < required_queues:
            raise ValueError("V9 requires exclusive prefetch and per-final-AIV queues")
        self.rounds = li_geometry(batch, li_tiling, intermediate_fetches, aic_count, service_aics)[
            "intermediate_fetches"
        ]
        self.service_lanes = li_geometry(batch, li_tiling, intermediate_fetches, aic_count, service_aics)[
            "service_aivs_per_window"
        ]

        self.final_workers = final_aivs(batch, li_tiling, aic_count, final_workers)
        spec = rotation_spec(
            batch, history, resident, self.final_workers, self.rounds, self.service_lanes, owner.abi_sizes["status"]
        )
        for name, (shape, dtype_name) in spec.items():
            dtype = getattr(torch, dtype_name)
            value = torch.zeros(shape, dtype=dtype, device=self.device) if scratch is None else scratch[name]
            if (
                value.device != self.device
                or value.dtype != dtype
                or tuple(value.shape) != shape
                or not value.is_contiguous()
            ):
                raise ValueError(f"Shared LIFetch scratch mismatch: {name}")
            setattr(self, name, value)
        self.final_posts = self.owners[batch * 32 + batch * self.final_workers * 288 :].reshape(
            batch, self.final_workers, 32
        )
        self.fallback_valid = torch.ones((batch, 8192), dtype=torch.uint8, device=self.device)
        if final_strategy != "replicated":
            raise ValueError("Direct LRU slots use the existing replicated resolver")
        self.reserved_hot.fill_(-1)

        def shared_input(value, shape, default):
            if value is None:
                return torch.full(shape, default, dtype=torch.int32, device=self.device)
            if (
                value.device != self.device
                or value.dtype != torch.int32
                or tuple(value.shape) != shape
                or not value.is_contiguous()
            ):
                raise ValueError("Shared LIFetch metadata must be contiguous int32 with the bound shape on this NPU")
            return value

        # Shared across layer groups for query bounds; per-group for residency.
        # Neither field is cleared by the per-round workspace reset.
        self.query_ends = shared_input(query_ends, (batch, 4), history)
        owner_count = batch if hot_valid is None else hot_valid.shape[0]
        self.hot_valid = shared_input(hot_valid, (owner_count, resident), 1)
        self.row_owners = None if row_owners is None else shared_input(row_owners, (batch,), 0)
        self.tail_bounds = None if tail_bounds is None else shared_input(tail_bounds, (batch, 2), 0)
        self.token_addresses = token_addresses
        if token_addresses is not None and (
            token_addresses.device != self.device
            or token_addresses.dtype not in (torch.int64, torch.uint64)
            or tuple(token_addresses.shape) != (owner_count, history)
            or not token_addresses.is_contiguous()
        ):
            raise ValueError("Token addresses must match the stable-owner Index storage")
        self.output_stride = resident if output_stride is None else output_stride
        tail_tokens = self.output_stride - resident
        if tail_tokens < 0 or tail_tokens % 128:
            raise ValueError("Output stride must preserve hot and optional ABC slots")
        if tail_bounds is not None and tail_tokens < 384:
            raise ValueError("Serving ABC bounds require at least 384 dedicated output slots")
        self.prefetch_status = owner.core_status[: batch * 2]
        pointers = [
            self.progress.data_ptr(),
            self.rows.data_ptr(),
            self.mapping.data_ptr(),
            self.locations.data_ptr(),
            self.events.data_ptr(),
            arguments[13].data_ptr() + batch * (4 * 2048 * 6 + 1024),
            arguments[9].data_ptr(),
            self.owners.data_ptr(),
            contexts.data_ptr(),
            self.blocks.data_ptr(),
            self.offsets.data_ptr(),
            self.destinations.data_ptr(),
            self.count.data_ptr(),
            self.row_status.data_ptr(),
            self.final_status.data_ptr(),
        ]
        blob = np.concatenate(
            (
                np.array(pointers, dtype="<u8").view("u1"),
                np.array([prefetch_limit, window_limit], dtype="<u4").view("u1"),
                np.array(
                    [
                        self.lru_slots.data_ptr(),
                        self.reserved_hot.data_ptr(),
                        self.changes.data_ptr(),
                        self.query_ends.data_ptr(),
                        self.fallback_addresses.data_ptr(),
                        self.fallback_valid.data_ptr(),
                        self.hot_valid.data_ptr(),
                        0 if self.row_owners is None else self.row_owners.data_ptr(),
                        0 if token_addresses is None else token_addresses.data_ptr(),
                        0 if token_addresses is None else token_addresses.stride(0),
                        self.output_stride,
                        0 if self.tail_bounds is None else self.tail_bounds.data_ptr(),
                    ],
                    dtype="<u8",
                ).view("u1"),
            )
        )
        blob = np.concatenate(
            (
                blob,
                np.frombuffer(bytes(self.runtime_shape), dtype=np.uint8),
                np.array([self.change_slots.data_ptr(), self.change_count.data_ptr()], dtype="<u8").view("u1"),
            )
        )
        self.context = torch.from_numpy(blob.copy()).to(self.device)
        self.arguments = tuple(arguments) + (self.context,)
        self.bindings = torch.tensor([x.data_ptr() for x in self.arguments], dtype=torch.uint64, device=self.device)
        self.functions = {}
        for name in ("init", "li", "wait", "resolve", "fetch", "probe"):
            fn = getattr(self.library, "v9_" + name)
            fn.argtypes, fn.restype = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p], ctypes.c_uint32
            self.functions[name] = fn
        self.reset()

    def reset(self):
        for tensor in (
            self.progress,
            self.rows,
            self.events,
            self.owners,
            self.count,
            self.row_status,
            self.final_status,
            self.prefetch_status,
        ):
            tensor.zero_()
        self.changes.fill_(-1)
        self.mapping.fill_(-1)
        self.locations.fill_(-1)

    def _call(self, name):
        stream = self.torch.npu.current_stream(self.device)
        result = self.functions[name](self.bindings.data_ptr(), stream.npu_stream, self.shape_pointer)
        if result:
            raise RuntimeError(f"v9_{name}_launch_failed:{result}")

    def launch_compute(self):
        self._call("init")
        self._call("li")

    def wait(self):
        self._call("wait")

    def launch_final_submit(self):
        self._call("resolve")

    def finish(self):
        self._call("fetch")

    def launch(self):
        self.launch_compute()
        self.launch_final_submit()
        self.finish()
