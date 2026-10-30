"""Graph-safe V9 launches. Allocations/reset/readback stay outside timing."""

import ctypes
from numbers import Integral
from pathlib import Path

import numpy as np

if __package__:
    from .contract import (
        ABI_VERSION,
        GEOMETRY_FIELDS,
        POINTER_FIELDS,
        DeviceContract,
        Status,
        allocate_workspace,
        descriptor_bytes,
        validate_buffers,
    )
    from .geometry import final_aivs, li_geometry
    from .index_cache import allocate_cache, bind_caches
else:
    from contract import (
        ABI_VERSION,
        GEOMETRY_FIELDS,
        POINTER_FIELDS,
        DeviceContract,
        Status,
        allocate_workspace,
        descriptor_bytes,
        validate_buffers,
    )
    from geometry import final_aivs, li_geometry
    from index_cache import allocate_cache, bind_caches


def validate_transport_library(library, owner):
    marker = getattr(library, "v9_test_transport", None)
    if marker is not None:
        marker.argtypes, marker.restype = [], ctypes.c_uint32
    test_binary = bool(marker is not None and marker())
    test_owner = getattr(owner, "transport", None) == "npu-memory-test"
    if test_binary != test_owner:
        raise ValueError("test transport library and owner must match; production cannot use mock binaries")


class RotatingLaunch:
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
        prefetch_limit=2048,
        window_limit=480,
        li_tiling="auto",
        intermediate_fetches=32,
        aic_count=32,
        service_aics=0,
        final_workers=0,
        final_strategy="replicated",
    ):
        self.torch, self.owner = torch, owner
        self.library = ctypes.CDLL(str(Path(library).resolve(strict=True)))
        self.device = arguments[0].device
        if (
            len(arguments) != 14
            or self.device.type != "npu"
            or any(x.device != self.device or not x.is_contiguous() for x in arguments)
        ):
            raise ValueError("V9 requires 14 contiguous input buffers on the same NPU")
        for name, expected in (
            ("contract_mode", 0),
            ("final_replicated", int(final_strategy == "replicated")),
            ("final_workers", final_aivs(batch, li_tiling, aic_count, final_workers)),
            ("aic_count", aic_count),
            ("abi_version", 8),
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
            ("context_bytes", 136),
            ("row_bytes", 1024),
        ):
            fn = getattr(self.library, "v9_" + name)
            fn.argtypes, fn.restype = [], ctypes.c_uint32
            if fn() != expected:
                raise ValueError(f"V9 binary contract mismatch: {name}")
        if not 0 <= prefetch_limit <= 8192 or not 1 <= window_limit <= 480:
            raise ValueError("Invalid V9 admission budgets")
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

        def zeros(shape, dtype=torch.int32):
            return torch.zeros(shape, dtype=dtype, device=self.device)

        self.progress = zeros((batch, 4, 32))
        self.rows = zeros((batch, 256))
        self.mapping = zeros((batch, history))
        self.locations = zeros((batch, 4, 2048))
        self.events = zeros((max(1, self.rounds), self.service_lanes, 128), torch.uint8)
        self.final_workers = final_aivs(batch, li_tiling, aic_count, final_workers)
        self.owners = zeros((batch * 32 + batch * self.final_workers * (288 + 32),))
        self.final_posts = self.owners[batch * 32 + batch * self.final_workers * 288 :].reshape(
            batch, self.final_workers, 32
        )
        self.blocks, self.offsets, self.destinations = [zeros(batch * 8192) for _ in range(3)]
        self.count = zeros(1)
        self.row_status = zeros((batch, 32))  # Independent 128-byte cache line per AIV
        self.final_status = zeros((batch * 2, owner.abi_sizes["status"]), torch.uint8)
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
                np.zeros(8, dtype="u1"),
            )
        )
        self.context = torch.from_numpy(blob.copy()).to(self.device)
        self.arguments = tuple(arguments) + (self.context,)
        self.bindings = torch.tensor([x.data_ptr() for x in self.arguments], dtype=torch.uint64, device=self.device)
        self.functions = {}
        for name in ("init", "li", "wait", "resolve", "fetch", "probe"):
            fn = getattr(self.library, "v9_" + name)
            fn.argtypes, fn.restype = [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_uint32
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
        self.mapping.fill_(-1)
        self.locations.fill_(-1)

    def _call(self, name):
        stream = self.torch.npu.current_stream(self.device)
        result = self.functions[name](self.bindings.data_ptr(), stream.npu_stream)
        if result:
            raise RuntimeError(f"v9_{name}_launch_failed:{result}")

    def launch_compute(self, *, after_init=None):
        self._call("init")
        if after_init is not None:
            after_init()
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


def transient_buffer(torch, state, name, shape, dtype, device, fill=0):
    """Reuse launch storage without clearing the previous reservation dirty set."""
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    value = state.get(name)
    if value is None:
        value = torch.full(shape, fill, dtype=dtype, device=device)
        state[name] = value
    elif tuple(value.shape) != shape or value.dtype != dtype or value.device != torch.device(device):
        raise ValueError(f"V9 transient buffer geometry changed: {name}")
    return value


class ContractLaunch(RotatingLaunch):
    """Bind caller-owned tensors once, then enqueue exactly four native launches."""

    def __init__(
        self,
        torch,
        library,
        tensors,
        owner,
        geometry,
        *,
        workspace=None,
        launch_state=None,
        index_caches=None,
        prefetch_limit=2048,
        window_limit=480,
        li_tiling="auto",
        intermediate_fetches=32,
        aic_count=32,
        service_aics=0,
        final_workers=0,
        final_strategy="replicated",
    ):
        geometry.validate()
        if final_strategy != "replicated":
            raise ValueError("ABI 13 supports V9's replicated final strategy only")
        if not 0 <= prefetch_limit <= 2048 or not 1 <= window_limit <= 480:
            raise ValueError("invalid prefetch/window limit")
        self.torch, self.owner, self.geometry = torch, owner, geometry
        self.tensors = dict(tensors)
        self.device = tensors["query"].device
        self.workspace = workspace if workspace is not None else allocate_workspace(torch, geometry, self.device)
        self.index_caches = (
            index_caches
            if index_caches is not None
            else [allocate_cache(torch, geometry, self.device) for _ in range(geometry.batch)]
        )
        bind_caches(torch, self.tensors, geometry, self.index_caches)
        validate_buffers(geometry, self.tensors, self.workspace)
        self.library = ctypes.CDLL(str(Path(library).resolve(strict=True)))
        validate_transport_library(self.library, owner)
        b, q, c = geometry.batch, geometry.queries, geometry.candidate_capacity
        execution = li_geometry(b, li_tiling, intermediate_fetches, aic_count, service_aics, q)
        self.li_cores = execution["rotating_aic"]
        self.rounds = execution["intermediate_fetches"]
        self.service_lanes = execution["service_aivs_per_window"]
        self.final_workers = final_aivs(b, li_tiling, aic_count, final_workers, q)
        expected = dict(
            abi_version=ABI_VERSION,
            contract_mode=1,
            contract_bytes=ctypes.sizeof(DeviceContract),
            context_bytes=136,
            row_bytes=1024,
            batch=b,
            queries=q,
            history=c,
            resident=geometry.resident_capacity,
            li_cores=execution["rotating_aic"],
            aic_count=aic_count,
            simulation=0,
            rounds=self.rounds,
            service_lanes=self.service_lanes,
            final_workers=self.final_workers,
            final_replicated=1,
        )
        for name, value in expected.items():
            fn = getattr(self.library, "v9_" + name)
            fn.argtypes, fn.restype = [], ctypes.c_uint32
            actual = fn()
            if actual != value:
                raise ValueError(f"V9 binary {name}: {actual} != {value}")
        offset = self.library.v9_contract_offset
        offset.argtypes, offset.restype = [ctypes.c_uint32], ctypes.c_uint32
        for index, name in enumerate(POINTER_FIELDS + GEOMETRY_FIELDS):
            if offset(index) != getattr(DeviceContract, name).offset:
                raise ValueError(f"V9 descriptor offset mismatch: {name}")
        contexts = owner.build_device_args(library)
        if contexts.shape[0] < b * (self.final_workers + 1):
            raise ValueError("lease has too few exclusive queue views")
        if owner.output.data_ptr() != tensors["resident_output"].data_ptr():
            raise ValueError("registered output differs from caller-owned output")

        def zeros(shape, dtype=torch.int32):
            return torch.zeros(shape, dtype=dtype, device=self.device)

        state = {} if launch_state is None else launch_state
        # Rows, reservation lists and mapping must have the same lifetime. In init,
        # previous row.slots selects exactly the mappings that need to be cleared.
        workspace_identity = tuple((k, v.data_ptr()) for k, v in sorted(self.workspace.items()))
        if state and state.get("workspace_identity") != workspace_identity:
            raise ValueError("V9 launch state reused with a different workspace")
        state["workspace_identity"] = workspace_identity

        def buffer(name, shape, dtype=torch.int32, fill=0):
            return transient_buffer(torch, state, name, shape, dtype, self.device, fill)

        self.progress = buffer("progress", (b, 4, 32))
        self.rows = buffer("rows", (b, 256))
        self.mapping = buffer("mapping", (b, c), fill=-1)
        self.locations = buffer("locations", (b, 4, 2048))
        self.events = buffer("events", (max(1, self.rounds), self.service_lanes, 128), torch.uint8)
        self.owners = buffer("owners", b * 32 + b * self.final_workers * 320)
        self.final_posts = self.owners[b * 32 + b * self.final_workers * 288 :].reshape(b, self.final_workers, 32)
        self.blocks, self.offsets, self.destinations = [
            buffer(n, b * 8192) for n in ("blocks", "offsets", "destinations")
        ]
        self.count = buffer("count", 1)
        self.row_status = buffer("row_status", (b, 64))
        self.final_status = buffer("final_status", (b * 2, owner.abi_sizes["status"]), torch.uint8)
        self.prefetch_status = owner.core_status[: b * 2]
        self.score_scratch = buffer("score_scratch", (execution["rotating_aic"], 4, c), torch.int16)
        self.prefix = buffer("prefix", b * (4 * 2048 * 6 + 1024) + b * c, torch.uint8)
        self.descriptor = torch.from_numpy(
            np.frombuffer(descriptor_bytes(geometry, self.tensors, self.workspace), np.uint8).copy()
        ).to(self.device)
        pointers = [
            self.progress.data_ptr(),
            self.rows.data_ptr(),
            self.mapping.data_ptr(),
            self.locations.data_ptr(),
            self.events.data_ptr(),
            self.prefix.data_ptr() + b * (4 * 2048 * 6 + 1024),
            tensors["topk"].data_ptr(),
            self.owners.data_ptr(),
            contexts.data_ptr(),
            self.blocks.data_ptr(),
            self.offsets.data_ptr(),
            self.destinations.data_ptr(),
            self.count.data_ptr(),
            self.row_status.data_ptr(),
            self.final_status.data_ptr(),
        ]
        blob = (
            np.asarray(pointers, "<u8").tobytes()
            + np.asarray([prefetch_limit, window_limit], "<u4").tobytes()
            + np.asarray([self.descriptor.data_ptr()], "<u8").tobytes()
        )
        self.context = torch.from_numpy(np.frombuffer(blob, np.uint8).copy()).to(self.device)
        dummy = zeros(1, torch.uint8)
        self.arguments = tuple(
            tensors[n]
            for n in (
                "query",
                "key",
                "weights",
                "query_scale",
                "key_scale",
                "cumulative_query",
                "candidate_lens",
                "index_table",
            )
        ) + (dummy, tensors["topk"], dummy, dummy, self.score_scratch, self.prefix, self.context)
        self.bindings = torch.tensor([x.data_ptr() for x in self.arguments], dtype=torch.uint64, device=self.device)
        self.functions = {}
        for name in ("init", "li", "wait", "resolve", "fetch"):
            fn = getattr(self.library, "v9_" + name)
            fn.argtypes, fn.restype = [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_uint32
            self.functions[name] = fn
        self.outputs = {
            name: tensors[name]
            for name in (
                "topk",
                "destination",
                "miss_counts",
                "resident_lengths",
                "transfers",
                "status",
                "extra_prefetch_slots",
                "extra_prefetch_counts",
            )
        }
        owner.keepalive.append(self)

    def reset(self):
        raise RuntimeError(
            "init resets transient state on device; restore caller snapshots only after checked completion"
        )

    def set_mapping_epochs(self, epochs):
        """Before launch, on the launch stream, supply one nonnegative epoch per row.

        Zero (default) checks maps exactly. A positive epoch requires exclusive
        mapping ownership: increment before any external map write; restore the
        epoch with its paired cache snapshot; use zero on unknown writers or
        wraparound. Internal successful publication updates the matching H0.
        """
        if len(epochs) != self.geometry.batch or any(not isinstance(e, Integral) or not 0 <= e < 2**31 for e in epochs):
            raise ValueError("one int32 nonnegative mapping epoch per request is required")
        self.workspace["mapping_epoch"].copy_(self.torch.tensor(epochs, dtype=self.torch.int32, device=self.device))

    def completion(self):
        """Read back outside capture/timing; never infer DMA completion from sync alone."""
        self.torch.npu.current_stream(self.device).synchronize()
        status = self.tensors["status"].cpu().numpy().copy()
        rows = self.rows.cpu().numpy()
        posts = self.final_posts.cpu().numpy()
        pending = bool(np.any(rows[:, 211]) or np.any(posts[:, :, 1]) or np.any(posts[:, :, 3]))
        queue_errors = bool(np.any(rows[:, 11]) or np.any(posts[:, :, 2]) or np.any(status == Status.COMMUNICATION))
        if pending or queue_errors or np.any(status == Status.PENDING):
            self.owner.quarantine("v9_completion_unverified", f"status={status.tolist()}, pending={pending}")
        self.owner.mark_completed(completion_verified=True)
        return status
