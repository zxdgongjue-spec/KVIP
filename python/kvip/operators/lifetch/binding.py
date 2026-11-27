"""Graph-safe V9 launches. Allocations/reset/readback stay outside timing."""

import ctypes
from kvip.runtime.loader import load_lifetch_library

import numpy as np

from kvip.operators.lifetch.geometry import (
    LARGE_BATCHES,
    MAX_SEQUENCE_TOKENS,
    final_aivs,
    li_geometry,
    prefetch_budget,
    validate_history,
)
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
        prefetch_rank_start=1024,
        window_limit=None,
        carry_window_credit=False,
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
    ):
        if not isinstance(prefetch_rank_start, int) or not 1 <= prefetch_rank_start <= 2048:
            raise ValueError("prefetch_rank_start must be an integer in 1..2048")
        self.prefetch_rank_start = prefetch_rank_start
        validate_history(history, resident)
        if history > MAX_SEQUENCE_TOKENS and query_ends is None:
            raise ValueError("The padded 512K Q4 workspace requires explicit exclusive query_ends")
        if not 1 <= batch <= 16 and not (batch in LARGE_BATCHES and aic_count == 32 and li_tiling == "auto"):
            raise ValueError(f"Unsupported LIFetch geometry: batch={batch}, AICs={aic_count}, tiling={li_tiling}")
        self.torch, self.owner = torch, owner
        self.library = load_lifetch_library(library)
        self.device = arguments[0].device
        if (
            len(arguments) != 14
            or self.device.type != "npu"
            or any(x.device != self.device or not x.is_contiguous() for x in arguments)
        ):
            raise ValueError("V9 requires 14 contiguous input buffers on the same NPU")
        index_heads = arguments[0].numel() // (batch * 4 * 128)
        if batch > 16 and index_heads != 32:
            raise ValueError("Large-batch Q4 URMA currently requires 32 Index heads")
        self.runtime_shape = RuntimeShape.prepare(
            self.library,
            batch=batch,
            history=history,
            resident=resident,
            index_heads=index_heads,
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
            ("abi_version", 12),
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
            ("context_bytes", 352),
            ("row_bytes", 1024),
        ):
            fn = getattr(self.library, "v9_" + name)
            fn.argtypes, fn.restype = [ctypes.c_void_p], ctypes.c_uint32
            if fn(self.shape_pointer) != expected:
                raise ValueError(f"V9 binary contract mismatch: {name}")
        prefetch_limit, window_limit = prefetch_budget(prefetch_limit, window_limit, intermediate_fetches)
        self.prefetch_limit = prefetch_limit
        if carry_window_credit and not (batch > 16 and self.runtime_shape.rounds == 4
                                        and window_limit == 256 and prefetch_limit == 1024):
            raise ValueError("Carry-forward requires large Q4, four windows and a 256/1024 budget")
        self.large_four_window_credit = carry_window_credit
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
            if name == "mapping_journal" and scratch is not None and name not in scratch:
                # Keep the sparse snapshot journal paired with its shared mapping.
                scratch[name] = torch.zeros(shape, dtype=dtype, device=self.device)
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
        covered_start = batch * (512 + 4 * 2048 * 6 + 1024)
        covered_bytes = batch * ((history + 255) // 256) * 32
        self.covered = self.window_topk_storage[covered_start:covered_start + covered_bytes]
        self.rank_progress = self.window_topk_storage[-batch*4*32*8:].view(torch.int64).view(batch,4,32)
        self.incremental = self.window_topk_storage[:batch*512].view(torch.int64).view(batch,4,16)
        blob = np.concatenate((blob, np.array([self.incremental.data_ptr(), prefetch_rank_start,
                               self.mapping_journal.data_ptr() if self.mapping_journal is not None else 0, 0], dtype="<u8").view("u1")))
        self.context = torch.from_numpy(blob.copy()).to(self.device)
        self.arguments = tuple(arguments) + (self.context,)
        self.bindings = torch.tensor([x.data_ptr() for x in self.arguments] + [0], dtype=torch.uint64, device=self.device)
        self.functions = {}
        for name in ("init", "li"):
            fn = getattr(self.library, "v9_" + name)
            fn.argtypes, fn.restype = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p], ctypes.c_uint32
            self.functions[name] = fn
        self.reset()

    def reset(self):
        for tensor in (
            self.covered,
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
        if self.mapping_journal is not None:
            self.mapping_journal.zero_()
        self.locations.fill_(-1)

    def _call(self, name):
        stream = self.torch.npu.current_stream(self.device)
        result = self.functions[name](self.bindings.data_ptr(), stream.npu_stream, self.shape_pointer)
        if result:
            raise RuntimeError(f"v9_{name}_launch_failed:{result}")
