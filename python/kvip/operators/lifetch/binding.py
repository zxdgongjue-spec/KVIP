"""Graph-safe V9 launches. Allocations/reset/readback stay outside timing."""

import ctypes
from kvip.runtime.loader import load_lifetch_library

import numpy as np

from kvip.operators.lifetch.geometry import final_aivs, li_geometry, prefetch_budget
from kvip.operators.lifetch.tiling import RuntimeShape
from kvip.operators.lifetch.workspace import rotation_spec


def _admission_values(value, batch):
    """Validate on the CPU; accepting device inputs would hide a readback."""
    if getattr(getattr(value, "device", None), "type", "cpu") != "cpu":
        raise ValueError("URMA admission controls must be a CPU int32[B,4] table")
    values = np.asarray(value)
    if values.dtype != np.int32 or values.shape != (batch, 4) or not values.flags.c_contiguous:
        raise ValueError("URMA admission controls must be contiguous CPU int32[B,4]")
    bounds = ((0, 2048), (1, 480), (0, 65535), (0, 1))
    if any(np.any(values[:, i] < low) or np.any(values[:, i] > high) for i, (low, high) in enumerate(bounds)):
        raise ValueError("URMA admission bounds: total=0..2048, window=1..480, key_steps=0..65535, carry=0/1")
    return values.copy()


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
        threshold_bounds=None,
        admission_controls=None,
        output_stride=None,
        scratch=None,
    ):
        if resident > 131072:
            raise ValueError("LIFetch hot bitmap supports resident <= 131072")
        if not 1 <= batch <= 16:
            raise ValueError(f"LIFetch supports BS1..16; received BS={batch}; no legacy fallback")
        self.torch, self.owner, self.batch = torch, owner, batch
        admission_values = None if admission_controls is None else _admission_values(admission_controls, batch)
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
        for name, expected in (
            ("token_bytes", token_bytes),
            ("final_replicated", int(final_strategy == "replicated")),
            ("final_workers", final_aivs(batch, li_tiling, aic_count, final_workers)),
            ("aic_count", aic_count),
            ("abi_version", 13),
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
            ("context_bytes", 336),
            ("row_bytes", 1024),
        ):
            fn = getattr(self.library, "v9_" + name)
            fn.argtypes, fn.restype = [ctypes.c_void_p], ctypes.c_uint32
            if fn(self.shape_pointer) != expected:
                raise ValueError(f"V9 binary contract mismatch: {name}")
        prefetch_limit, window_limit = prefetch_budget(prefetch_limit, window_limit, intermediate_fetches)
        self.prefetch_limit = prefetch_limit
        geometry = li_geometry(batch, li_tiling, intermediate_fetches, aic_count, service_aics)
        if geometry["logical_aic_tasks"] > aic_count or not geometry["intermediate_fetches"]:
            raise ValueError("Direct hot-first mode requires a window and concurrently resident LI tasks")
        contexts = owner.build_device_args(library)
        required_queues = max(batch * (final_aivs(batch, li_tiling, aic_count, final_workers) + 1), 2 * batch + 16)
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
        self.threshold_bounds = None if threshold_bounds is None else shared_input(threshold_bounds, (batch, 2), 0)
        self._admission_storage = (
            None if admission_values is None else torch.from_numpy(admission_values).to(self.device)
        )
        self.admission_controls = self._admission_storage
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
                np.array(
                    [
                        self.change_slots.data_ptr(),
                        self.change_count.data_ptr(),
                        0 if self.threshold_bounds is None else self.threshold_bounds.data_ptr(),
                        0 if self.admission_controls is None else self.admission_controls.data_ptr(),
                    ],
                    dtype="<u8",
                ).view("u1"),
            )
        )
        if blob.size != 336:
            raise AssertionError("LIFetch Context ABI changed")
        self.context = torch.from_numpy(blob.copy()).to(self.device)
        self.arguments = tuple(arguments) + (self.context,)
        self.bindings = torch.tensor([x.data_ptr() for x in self.arguments], dtype=torch.uint64, device=self.device)
        self.functions = {}
        for name in ("init", "li"):
            fn = getattr(self.library, "v9_" + name)
            fn.argtypes, fn.restype = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p], ctypes.c_uint32
            self.functions[name] = fn
        self.reset()

    def set_admission_controls(self, values):
        """Update optional URMA int32[B,4] controls outside graph capture.

        Columns are total prefetch limit, per-window quota, upward sortable-key
        steps, and carry (0/1). ``None`` restores the original admission path.
        Pass a contiguous NumPy array or CPU tensor. No NPU readback is used.
        Storage is retained across updates; allocate it before graph capture.
        Queue updates on the graph's stream after its preceding use completes.
        CPU validation and H2D belong to caller preparation time, not a free
        part of kernel timing. Never change this table while a graph uses it.
        """
        capturing = getattr(getattr(self.torch, "npu", None), "is_current_stream_capturing", None)
        if callable(capturing) and capturing():
            raise RuntimeError("Set URMA admission controls outside graph capture")
        checked = None if values is None else _admission_values(values, self.batch)
        previous = 0 if self.admission_controls is None else self.admission_controls.data_ptr()
        if checked is None:
            self.admission_controls = None
        else:
            source = self.torch.from_numpy(checked)
            if self._admission_storage is None:
                self._admission_storage = source.to(self.device)
            else:
                self._admission_storage.copy_(source)
            self.admission_controls = self._admission_storage
        address = 0 if self.admission_controls is None else self.admission_controls.data_ptr()
        if address != previous:
            pointer = np.array([address], dtype="<u8").view("u1")
            self.context[328:336].copy_(self.torch.from_numpy(pointer))
        return self

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
