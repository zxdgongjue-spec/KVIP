"""Real KVAF preparation and ownership for the V8 staged AIV1 benchmark.

Everything here is setup/verification, never part of an operator timing scope.
The proven V5 service session is reused ONLY for transport setup, not its
operator, plan, profiler or benchmark. No synthetic Fetch fallback exists.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

MAGIC = 0x514C495638535447
VERSION = 1
TOKENS = 131075
PADDED_TOKENS = 131200
TOPK = 2048
TOKEN_BYTES = 656
BLOCK_TOKENS = 128
CONTEXT_WORDS = 64
DEFAULT_SERVER = "127.0.0.1:28999"


def unsigned_address_view(addresses, torch):
    """Reinterpret KVAF's address bits for ACL_UINT64; never cast or copy."""
    if addresses.dtype not in (torch.int64, torch.uint64) or not addresses.is_contiguous():
        raise ValueError(f"kvaf_address_storage_invalid:{addresses.dtype}")
    result = addresses if addresses.dtype == torch.uint64 else addresses.view(torch.uint64)
    if result.data_ptr() != addresses.data_ptr() or result.shape != addresses.shape:
        raise ValueError("kvaf_address_view_must_alias_original_storage")
    return result


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def discover(
    *,
    server: str | None = None,
    kvaf_python_dir: Path | None = None,
    get_kernel: Path | None = None,
    put_kernel: Path | None = None,
    kvaf_root: Path | None = None,
) -> dict[str, Any]:
    """Read existing build artifacts/config; never build or contact a service."""
    candidates = []
    if kvaf_root is not None:
        candidates = [Path(kvaf_root) / "build-npu", Path(kvaf_root) / "build"]
    values = {
        "kvaf_python_dir": kvaf_python_dir,
        "get_kernel": get_kernel,
        "put_kernel": put_kernel,
    }
    suffixes = {
        "kvaf_python_dir": "python",
        "get_kernel": "bin/kvaf_execute_batch_get_kernels.o",
        "put_kernel": "bin/kvaf_execute_batch_put_kernels.o",
    }

    def ready(name: str, path: Path) -> bool:
        if name != "kvaf_python_dir":
            return path.is_file()
        package = path / "kvaf"
        return (package / "__init__.py").is_file() and any(package.glob("_kvaf*.so"))

    bundle = next(
        (
            base
            for base in candidates
            if all(values[name] is not None or ready(name, base / suffix) for name, suffix in suffixes.items())
        ),
        None,
    )
    for name, suffix in suffixes.items():
        if values[name] is None:
            values[name] = bundle / suffix if bundle is not None else None
        if values[name] is None or not ready(name, Path(values[name])):
            raise ValueError(
                f"v8_fetch_dependency_missing:{name}: expected a matching KVAF "
                "build-npu bundle; pass --kvaf-root to its source checkout"
            )
        values[name] = Path(values[name]).resolve()
    values["server"] = server or DEFAULT_SERVER
    return values


def context_rows(
    lease: dict[str, int],
    *,
    batch: int,
    resident_budget: int,
    output_address: int,
    valid_address: int,
    temporary_capacity: int = 0,
    shared_queue_address: int = 0,
) -> list[list[int]]:
    """Build the same ctx64 geometry validated by aiv1_stage_contract.h."""
    required = (
        "device_channel_entity",
        "device_queue_state",
        "device_core_status",
        "output_address",
        "output_bytes",
        "worker_count",
        "jettys_per_core",
        "token_bytes",
        "tokens_per_block",
    )
    if any(type(lease.get(name)) is not int or not 0 < lease[name] < 2**64 for name in required):
        raise ValueError("v8_fetch_lease_fields_invalid")
    if (
        not 1 <= batch <= 16
        or type(shared_queue_address) is not int
        or not 0 <= shared_queue_address < 2**64
        or not 0 <= temporary_capacity <= TOKENS
        or (temporary_capacity != 0 and resident_budget < TOPK)
        or not 0 <= resident_budget <= TOKENS
        or valid_address <= 0
        or lease["output_address"] != output_address
        or lease["output_bytes"]
        != batch * (resident_budget + temporary_capacity if temporary_capacity else TOPK) * TOKEN_BYTES
        or not (1 if shared_queue_address else batch) <= lease["worker_count"] <= 48
        or (
            shared_queue_address
            and (
                shared_queue_address % 128
                or lease["worker_count"] != 1
                or lease["jettys_per_core"] != 2
                or not temporary_capacity
            )
        )
        or not 1 <= lease["jettys_per_core"] <= 2
        or lease["token_bytes"] != TOKEN_BYTES
        or lease["tokens_per_block"] != BLOCK_TOKENS
    ):
        raise ValueError("v8_fetch_lease_geometry_mismatch")
    rows = []
    for row in range(batch):
        fields = [
            MAGIC,
            VERSION,
            resident_budget,
            TOKEN_BYTES,
            batch * (resident_budget + temporary_capacity if temporary_capacity else TOPK),
            1,
            0,
            PADDED_TOKENS // BLOCK_TOKENS,
            batch * (PADDED_TOKENS // BLOCK_TOKENS),
            TOKENS,
            row,
            lease["device_channel_entity"],
            lease["device_queue_state"],
            lease["device_core_status"],
            lease["worker_count"],
            lease["jettys_per_core"],
            valid_address,
            output_address,
            lease["output_bytes"],
            temporary_capacity,
            shared_queue_address,
        ]
        rows.append(fields + [0] * (CONTEXT_WORDS - len(fields)))
    return rows


def quarantine_all(resources: Any, reason: str, error: BaseException | str) -> None:
    """Do not unwind Python/NPU destructors after any uncertain stage3 launch."""
    # Keep the argument referenced until process teardown; os._exit is the
    # existing KVAF safety boundary, not an exception that finally may release.
    print(
        f"[QLI_V8_NATIVE] status=failed stage=fetch_quarantine reason={reason} "
        f"detail={str(error)[:240]!r} resources_retained={resources is not None}",
        flush=True,
    )
    os._exit(70)


def _session_class() -> Any:
    path = Path(__file__).resolve().parent / "native_batchget_support.py"
    spec = importlib.util.spec_from_file_location("qli_v8_verified_service_setup", path)
    if spec is None or spec.loader is None:
        raise ValueError("v8_verified_service_setup_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Session(module.NativeFetchSession):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # 128 blocks = 16384 KV tokens, ~10.25 MiB per PUT. All setup,
            # including completion fences, is outside benchmark timing.
            kwargs["put_chunk_blocks"] = 128
            super().__init__(*args, **kwargs)

        def quarantine(self, reason: str, error: BaseException) -> None:
            quarantine_all(self, reason, error)

    return Session


class Resources:
    def __init__(self, torch: Any, device: Any, resident_budget: int, batch: int = 2) -> None:
        self.torch = torch
        self.device = device
        self.resident_budget = resident_budget
        self.batch = batch
        self.session: Any = None
        self.lease: dict[str, int] | None = None
        self.lease_stream: Any = None
        self.context: Any = None
        self.in_flight = False
        self.closed = False
        self.graphs: list[Any] = []
        # The runner pins graph dictionaries, bridges and their ACL workspaces
        # here BEFORE first enqueue. Its stack can unwind before the outer
        # quarantine handler runs; local-only references are insufficient.
        self.keepalive: list[Any] = []

    def mark_launched(self) -> None:
        if self.closed or self.lease is None:
            raise ValueError("v8_fetch_resources_not_live")
        self.in_flight = True

    def acquire_fused(self) -> None:
        """Transfer idle native queues to fused ownership, outside timing."""
        if self.closed or self.in_flight or self.lease is not None:
            raise ValueError("v8_fetch_lease_transition_invalid")
        self.torch.npu.synchronize()
        self.lease_stream = self.torch.npu.current_stream()
        self.lease = self.session.runtime.acquire_fused_batch_get_context(self.temporary_kv, self.lease_stream)
        rows = context_rows(
            self.lease,
            batch=self.batch,
            resident_budget=self.resident_budget,
            output_address=self.temporary_kv.data_ptr(),
            valid_address=self.session.device_table.valid.data_ptr(),
            temporary_capacity=getattr(self, "temporary_capacity", 0),
        )
        self.context = self.torch.tensor(rows, dtype=self.torch.uint64, device=self.device)
        self.torch.npu.synchronize()
        print(
            f"[QLI_V8_NATIVE] stage=fetch_topology topology=per-row "
            f"fused_allocated_jettys={self.lease['worker_count'] * self.lease['jettys_per_core']} "
            f"fused_mapped_jettys={self.batch * self.lease['jettys_per_core']} "
            "native_topology=unchanged submitter=each_aiv1 timing=excluded",
            flush=True,
        )

    def reset(self) -> None:
        if self.in_flight:
            quarantine_all(self, "reset_before_verified_completion", "verify previous stage3 status first")
        self.session.reset()
        if hasattr(self, "initial_cache"):
            self.temporary_kv.copy_(self.initial_cache)

    def complete(self, status: Any) -> list[list[int]]:
        try:
            rows = status.cpu().reshape(self.batch, 32).tolist()
        except BaseException as error:
            quarantine_all(self, "uncertain_stage3_completion", error)
        invalid = [
            row
            for row in rows
            if row[0] != 1
            or row[1] != 64
            or row[2] != 131072
            or row[4] != 0
            or row[6] != 4
            or row[7] != TOKENS
            or row[8] != row[9]
        ]
        if invalid:
            if hasattr(self, "shared_queue_state"):
                words = self.shared_queue_state.view(self.torch.uint64).cpu().tolist()
                print(
                    f"[QLI_V8_NATIVE] stage=shared_queue_detail lock={words[0]} "
                    f"completed_pi={words[16:18]} validated_mask={words[18]} posted_cq={words[19:21]} "
                    f"error={words[21:23]} failed_pi={words[23:25]} failed_ci={words[25:27]} "
                    f"failed_cqe_entry_word0={words[27:29]}",
                    flush=True,
                )
            print(f"[QLI_V8_NATIVE] stage=fetch_status_detail status=failed rows={rows}", flush=True)
            quarantine_all(self, "stage3_status_not_complete", f"actual_rows={invalid}")
        self.in_flight = False
        return rows

    def verify(self, selected_ids: Any, selected_locations: Any, status: Any, *, emit: bool = False) -> dict[str, Any]:
        rows = self.complete(status)
        ids = selected_ids.cpu().reshape(self.batch, 4 * TOPK).long()
        locations = selected_locations.cpu().reshape(self.batch, 4 * TOPK).long()
        output = self.temporary_kv.cpu()
        examples = []
        mismatch_tokens = 0
        compared = 0
        for row in range(self.batch):
            for index in range(4 * TOPK):
                token, location = int(ids[row, index]), int(locations[row, index])
                capacity = getattr(self, "temporary_capacity", TOPK)
                if not 0 <= token < TOKENS or not 0 <= location < self.resident_budget + capacity:
                    reason = "index_or_location_out_of_range"
                elif hasattr(self, "initial_cache"):
                    compared += int(location >= self.resident_budget)
                    actual = output[row * (self.resident_budget + capacity) + location]
                    expected = self.session.cpu_payload[row, token]
                    reason = None if self.torch.equal(actual, expected) else "ddr_payload_mismatch"
                elif location < self.resident_budget:
                    resident_token = self.resident_ids[location] if hasattr(self, "resident_ids") else location
                    reason = None if token == resident_token else "resident_location_mismatch"
                else:
                    compared += 1
                    actual = output[row * TOPK + location - self.resident_budget]
                    expected = self.session.cpu_payload[row, token]
                    reason = None if self.torch.equal(actual, expected) else "ddr_payload_mismatch"
                if reason:
                    mismatch_tokens += 1
                    if len(examples) < 8:
                        example = {"row": row, "index": index, "token": token, "location": location, "reason": reason}
                        if reason == "ddr_payload_mismatch":
                            first = int((actual != expected).nonzero()[0, 0])
                            example.update(
                                first_bad_byte=first,
                                actual_bytes=actual[first : first + 8].tolist(),
                                expected_bytes=expected[first : first + 8].tolist(),
                            )
                        examples.append(example)
        result = {
            "payload_verified": mismatch_tokens == 0,
            "compared_temporary_tokens": compared,
            "mismatch_tokens": mismatch_tokens,
            "examples": examples,
            "submitted": [row[8] for row in rows],
            "completed": [row[9] for row in rows],
            "direct_wqe_groups": [row[11] & 65535 for row in rows],
            "kvaf_wqe_groups": [row[11] >> 16 for row in rows],
        }
        if emit or mismatch_tokens:
            print(
                f"[QLI_V8_NATIVE] stage=fetch_payload status={'pass' if mismatch_tokens == 0 else 'failed'} "
                f"{json.dumps(result)}",
                flush=True,
            )
        return result

    def finish_close(self) -> None:
        if self.closed:
            return
        if self.in_flight:
            quarantine_all(self, "close_before_verified_completion", "stage3 output may still have pending DMA")
        try:
            if self.session is not None:
                self.torch.npu.synchronize()
                if self.lease is not None:
                    self.session.runtime.release_fused_batch_get_context(self.lease, self.lease_stream)
                    self.lease = None
                for graph in self.graphs:
                    graph.reset()
                self.graphs.clear()
                self.session.close()
                self.session = None
        except BaseException as error:
            quarantine_all(self, "unsafe_fetch_resource_release", error)
        self.closed = True
        self.keepalive.clear()


def prepare(
    torch: Any,
    device: Any,
    *,
    server: str | None = None,
    kvaf_python_dir: Path | None = None,
    get_kernel: Path | None = None,
    put_kernel: Path | None = None,
    kvaf_root: Path | None = None,
    batch: int = 2,
    source_tokens: int = TOKENS,
    resident_budget: int = 12288,
    temporary_capacity: int = 0,
    fetch_topology: str = "per-row",
    defer_fused_lease: bool = False,
    resident_ids=None,
) -> Resources:
    if fetch_topology != "per-row":
        raise ValueError("invalid_fetch_topology")
    if not 1 <= batch <= 16 or source_tokens != TOKENS or not 0 <= resident_budget <= TOKENS:
        raise ValueError("v8_fetch_requires_b1_to_b16_128k_fixture")
    if resident_ids is not None and (
        not temporary_capacity
        or len(resident_ids) != batch
        or any(
            len(row) != resident_budget
            or len(set(row)) != resident_budget
            or any(token < 0 or token >= TOKENS for token in row)
            for row in resident_ids
        )
    ):
        raise ValueError("invalid_resident_ids")
    config = discover(
        server=server,
        kvaf_python_dir=kvaf_python_dir,
        get_kernel=get_kernel,
        put_kernel=put_kernel,
        kvaf_root=kvaf_root,
    )
    loaded = sys.modules.get("kvaf")
    if loaded is not None:
        origin = getattr(loaded, "__file__", None)
        if origin is None or not Path(origin).resolve().is_relative_to(config["kvaf_python_dir"]):
            raise ValueError("v8_kvaf_already_imported_from_different_runtime_bundle")
    sys.path.insert(0, str(config["kvaf_python_dir"]))
    resources = Resources(torch, device, resident_budget, batch)
    resources.server = config["server"]
    print(f"[QLI_V8_NATIVE] stage=fetch_prepare status=begin server={config['server']} timing=excluded", flush=True)
    try:
        placeholders = torch.zeros((batch, TOPK), dtype=torch.int32, device=device)
        counts = torch.zeros((batch,), dtype=torch.int32, device=device)
        entries = torch.arange(batch, dtype=torch.int32, device=device)
        budgets = torch.full((batch,), TOPK, dtype=torch.int32, device=device)
        lengths = torch.full((batch,), TOKENS, dtype=torch.int32, device=device)
        resources.session = _session_class().open(
            torch,
            device,
            batch,
            PADDED_TOKENS,
            TOPK,
            config["server"],
            config["get_kernel"],
            config["put_kernel"],
            placeholders,
            placeholders,
            counts,
            entries,
            budgets,
            lengths,
            temporary_only=True,
            npu_core_count=32,
            put_chunk_blocks=128,
        )
        session = resources.session
        resources.temporary_kv = session.output
        if temporary_capacity:
            if not TOPK <= resident_budget <= TOKENS or not 1 <= temporary_capacity <= TOKENS:
                raise ValueError("invalid_fixed_fetch_cache_capacity")
            resources.temporary_capacity = temporary_capacity
            resources.temporary_kv = torch.empty(
                (batch * (resident_budget + temporary_capacity), TOKEN_BYTES), dtype=torch.uint8, device=device
            )
            resources.initial_cache = torch.zeros_like(resources.temporary_kv)
            payload = (
                session.cpu_payload[:, :resident_budget]
                if resident_ids is None
                else torch.stack([session.cpu_payload[row, ids] for row, ids in enumerate(resident_ids)])
            ).to(device)
            resources.initial_cache.reshape(batch, resident_budget + temporary_capacity, TOKEN_BYTES)[
                :, :resident_budget
            ].copy_(payload)
            resources.temporary_kv.copy_(resources.initial_cache)
        resources.ddr_addresses = unsigned_address_view(session.device_table.addresses, torch)
        if not defer_fused_lease:
            resources.acquire_fused()
        resources.cache_map = torch.full((batch, TOKENS), -1, dtype=torch.int32, device=device)
        slots = torch.arange(resident_budget, dtype=torch.int32, device=device)
        if resident_ids is None:
            resources.cache_map[:, :resident_budget] = slots
        else:
            for row, ids in enumerate(resident_ids):
                resources.cache_map[row, torch.tensor(ids, dtype=torch.long, device=device)] = slots
        resources.cmp_to_raw = torch.arange(TOKENS, dtype=torch.int32, device=device).repeat(batch, 1)
        torch.npu.synchronize()
        print(
            f"[QLI_V8_NATIVE] stage=fetch_prepare status=pass timing=excluded batch={batch} "
            f"tokens={TOKENS} temporary_slots_per_row={temporary_capacity or TOPK} "
            f"final_destination={'hot' if temporary_capacity else 'temporary'}",
            flush=True,
        )
        return resources
    except BaseException:
        resources.finish_close()
        raise
