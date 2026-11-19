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
        threshold_bounds=None,
        output_stride=None,
        scratch=None,
    ):
        if resident > 131072:
            raise ValueError("LIFetch hot bitmap supports resident <= 131072")
        if not 1 <= batch <= 16:
            raise ValueError(f"LIFetch supports BS1..16; received BS={batch}; no legacy fallback")
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
        if self.library.kvip_cold_only_windows() != 1:
            raise ValueError("Cold-only Fetch window ABI required")
        self.ranked_port = bool(self.library.kvip_ranked_port())
        if self.ranked_port and (batch != 16 or history != 131328 or resident != 16384 or intermediate_fetches != 8 or threshold_bounds is None or query_ends is None):
            raise ValueError("Ranked port requires the validated BS16 Q4 eight-window frozen input")
        option = getattr(self.library, "kvip_incremental_topk", None)
        self.incremental_topk = bool(option()) if option is not None else False
        self.fetch_rank = any(bool(fn and fn()) for fn in
                              (getattr(self.library, "kvip_fetch_bitmap", None),
                               getattr(self.library, "kvip_fetch_dense", None)))
        packets_option = getattr(self.library, "kvip_fetch_packets", None)
        self.fetch_packets_enabled = bool(packets_option and packets_option())
        if self.fetch_rank and (batch != 16 or resident != 16384 or history % 128 or
                self.runtime_shape.rounds != 9 or not self.runtime_shape.splitQ or
                self.runtime_shape.indexHeads != 32 or not self.runtime_shape.batchedRows or
                threshold_bounds is None or query_ends is None or window_limit != 64 or prefetch_limit != 512):
            raise ValueError("Fetch ranking requires the BS16 Q4 8-cold-window 64/512 replay geometry")
        probe = getattr(self.library, "kvip_window_probe", None)
        window_probe = bool(probe and probe())
        mask_option = getattr(self.library, "kvip_seed_mask", None)
        seed_mask = bool(mask_option and mask_option())
        if self.incremental_topk and (resident != 16384 or self.runtime_shape.indexHeads != 32 or
                self.runtime_shape.rounds != 9 or not self.runtime_shape.splitQ):
            raise ValueError("Incremental TopK requires split Q4, H32, resident16384 and eight cold windows")
        trunk_option = getattr(self.library, "kvip_native_trunk", None)
        if trunk_option and trunk_option() and history - resident > 8 * 16384:
            raise ValueError("Eight Fetch windows exceed the native 16K score buffer")
        if self.incremental_topk and (threshold_bounds is None or query_ends is None):
            raise ValueError("Incremental TopK requires a validated resident+recent seed")
        self.shape_pointer = ctypes.byref(self.runtime_shape)
        context_bytes = (
            328 + 16 * (self.ranked_port + self.fetch_rank)
            + 8 * (self.incremental_topk + window_probe + seed_mask + self.fetch_packets_enabled)
        )
        for name, expected in (
            ("token_bytes", token_bytes),
            ("final_replicated", int(final_strategy == "replicated")),
            ("final_workers", final_aivs(batch, li_tiling, aic_count, final_workers)),
            ("aic_count", aic_count),
            ("abi_version", 12 + 128*self.ranked_port + 16*self.fetch_packets_enabled + 8*self.fetch_rank + self.incremental_topk + 2 * window_probe + 4 * seed_mask),
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
            ("context_bytes", context_bytes),
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
        if batch == 16 and self.rounds == 9 and window_limit == 64:
            # The verification pass records the token encoded in each WQE.
            # Final cache planning may repurpose its victim slot afterwards.
            event_bytes = self.events.numel()
            storage = torch.empty(event_bytes + batch * self.rounds * 64 * 4,
                                  dtype=torch.uint8, device=self.device)
            self.events = storage[:event_bytes].view_as(self.events)
            self.fetch_wqe_ids = storage[event_bytes:].view(torch.int32).reshape(batch, self.rounds, 64)
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
                    ],
                    dtype="<u8",
                ).view("u1"),
            )
        )
        if self.ranked_port:
            self.ranked_state = torch.empty((batch, 4, 16), dtype=torch.int64, device=self.device)
            self.ranked_offers = torch.empty((self.rounds, batch, 4, 2048), dtype=torch.int32, device=self.device)
            blob = np.r_[blob, np.array([self.ranked_state.data_ptr(), self.ranked_offers.data_ptr()], dtype="<u8").view("u1")]
        if self.fetch_rank:
            self.fetch_ranks = torch.empty((batch, 4, self.rounds, 3104), dtype=torch.int32, device=self.device)
            self.fetch_audit = torch.empty((batch, self.rounds, 96), dtype=torch.int32, device=self.device)
            blob = np.r_[blob, np.array([self.fetch_ranks.data_ptr(), self.fetch_audit.data_ptr()], dtype="<u8").view("u1")]
        if self.fetch_packets_enabled:
            self.fetch_packets = torch.empty((batch, 4, history // 128, 48), dtype=torch.int16, device=self.device)
            blob = np.r_[blob, np.array([self.fetch_packets.data_ptr()], dtype="<u8").view("u1")]
        if self.incremental_topk:
            # Optional experimental ABI: baseline allocations/layout are unchanged.
            self.incremental_lists = torch.empty((batch * 4, self.rounds, 2080),
                                                  dtype=torch.int32, device=self.device)
            blob = np.r_[blob, np.array([self.incremental_lists.data_ptr()], dtype="<u8").view("u1")]
        if window_probe:
            self.window_times = torch.zeros((batch * 4, self.rounds, 8), dtype=torch.int64, device=self.device)
            blob = np.r_[blob, np.array([self.window_times.data_ptr()], dtype="<u8").view("u1")]
        if seed_mask:
            self.seed_mask = torch.empty((batch, 16384 // 8), dtype=torch.int32, device=self.device)
            blob = np.r_[blob, np.array([self.seed_mask.data_ptr()], dtype="<u8").view("u1")]
        if blob.size != context_bytes:
            raise AssertionError(
                f"LIFetch Context ABI changed: packed={blob.size}, expected={context_bytes}"
            )
        self.context = torch.from_numpy(blob.copy()).to(self.device)
        self.arguments = tuple(arguments) + (self.context,)
        self.bindings = torch.tensor([x.data_ptr() for x in self.arguments], dtype=torch.uint64, device=self.device)
        self.functions = {}
        for name in ("init", "li"):
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
