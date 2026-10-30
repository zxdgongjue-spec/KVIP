"""Real service KVAF BatchGet for a fixed production-C8 replay fixture.

The ORIGINAL registered prepare op executes once outside measurement. Its
mask is compacted to the service KVAF prefix ABI outside measurement too.
The caller must validate that every measured LI+Update output still describes
exactly this frozen plan. This is explicitly not dynamic end-to-end latency.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

TOKEN_BYTES = 656
BLOCK_SIZE = 128
TOPK = 2048
GUARD_BYTE = 0xA5


def compact_mask_plan(
    indices: list[int], offsets: list[int], mask: list[int], output_indices: list[int]
) -> dict[str, list[int] | int]:
    if not (len(indices) == len(offsets) == len(mask) == len(output_indices)):
        raise ValueError("native_prepare_descriptor_lengths_differ")
    if any(type(flag) is not int or flag not in (0, 1) for flag in mask):
        raise ValueError("native_prepare_mask_not_binary")
    active = [index for index, flag in enumerate(mask) if flag]
    result = {
        "indices": [indices[index] for index in active],
        "offsets": [offsets[index] for index in active],
        "output_indices": [output_indices[index] for index in active],
        "miss_count": len(active),
    }
    if len(set(result["output_indices"])) != len(active):
        raise ValueError("native_prepare_destination_alias")
    if any(
        type(value) is not int or value < 0
        for field in ("indices", "offsets", "output_indices")
        for value in result[field]
    ):
        raise ValueError("native_prepare_negative_active_descriptor")
    return result


def payload(torch: Any, batch: int, tokens: int) -> Any:
    byte = torch.arange(TOKEN_BYTES, dtype=torch.int32).reshape(1, -1)
    token = torch.arange(tokens, dtype=torch.int32).reshape(-1, 1)
    rows = []
    for pool in range(batch):
        values = (token * 17 + (token >> 8) * 29 + pool * 101 + byte * 7 + (byte >> 8)).to(torch.uint8)
        for part in range(4):
            values[:, part::16] = ((token >> (part * 8)) & 255).to(torch.uint8)
        values[:, 4::16] = pool
        values[:, 5::16] = 0  # one selected layer, layer_id=0
        rows.append(values)
    return torch.stack(rows)


class NativeFetchSession:
    @classmethod
    def open(cls, *args: Any, **kwargs: Any) -> NativeFetchSession:
        # Keep a Python reference during partial construction too. An error
        # after async enqueue must reach close()/quarantine before GC can
        # destroy registered caller memory or the remote-table handle.
        session = cls.__new__(cls)
        try:
            session.__init__(*args, **kwargs)
        except BaseException:
            session.close()
            raise
        return session

    def __init__(
        self,
        torch: Any,
        device: Any,
        batch: int,
        candidate_tokens: int,
        cache_tokens: int,
        server: str,
        get_kernel: Path,
        put_kernel: Path,
        source_ids: Any,
        destination_slots: Any,
        miss_counts: Any,
        request_entries: Any,
        cache_budgets: Any,
        candidate_lens: Any,
        temporary_only: bool = False,
        put_chunk_blocks: int = 0,
        npu_core_count: int = 32,
        payload_factory: Callable[[Any, int, int], Any] | None = None,
    ) -> None:
        self.torch = torch
        self.graphs: list[Any] = []
        self.runtime = None
        self.table = None
        self.batch_get_attempted = False
        self.batch = batch
        self.cache_tokens = cache_tokens
        self.sources = source_ids
        self.destinations = destination_slots
        self.misses = miss_counts
        if candidate_tokens % BLOCK_SIZE or cache_tokens % BLOCK_SIZE:
            raise ValueError("native_fetch_requires_complete_128_token_blocks")
        for kernel in (get_kernel, put_kernel):
            if not kernel.is_file():
                raise ValueError(f"native_kvaf_kernel_missing:{kernel}")
        kvaf = importlib.import_module("kvaf")
        doc = str(kvaf.NpuRuntime.batch_get.__doc__)
        if "miss_count" not in doc or "mask" in doc:
            raise ValueError("native_kvaf_requires_verified_service_prefix_binding_no_abi_guess")
        if any(
            not hasattr(kvaf.NpuRuntime, method)
            for method in ("acquire_fused_batch_get_context", "release_fused_batch_get_context")
        ):
            raise ValueError("matching_kvaf_queue_health_lease_api_missing")
        self.runtime = kvaf.NpuRuntime(
            str(get_kernel),
            batch_get_token_bytes=TOKEN_BYTES,
            batch_get_tokens_per_block=BLOCK_SIZE,
            device_id=int(device.index or 0),
            npu_core_count=npu_core_count,
            batch_put_kernel_object_path=str(put_kernel),
            jettys_per_core=2,
        )
        self.runtime.link_to(server)
        qcap = candidate_tokens // BLOCK_SIZE
        capacity = batch * qcap
        block_bytes = BLOCK_SIZE * TOKEN_BYTES
        self.runtime.initialize(capacity * block_bytes, block_bytes)
        # Select the fixture before the first PUT: PreparePut reserves new
        # remote slots and must not be used to overwrite an initialized pool.
        self.cpu_payload = (payload if payload_factory is None else payload_factory)(torch, batch, candidate_tokens)
        self.seed = self.cpu_payload.to(device).reshape(capacity, block_bytes)
        keys = list(range(1000000, 1000000 + capacity))
        try:
            chunk = put_chunk_blocks or capacity
            if chunk < 1:
                raise ValueError("put_chunk_blocks_must_be_nonnegative")
            for begin in range(0, capacity, chunk):
                end = min(begin + chunk, capacity)
                self.runtime.batch_put(
                    keys[begin:end],
                    self.seed,
                    [index * block_bytes for index in range(begin, end)],
                    [block_bytes] * (end - begin),
                )
                torch.npu.synchronize()
        except BaseException as error:
            # GET queue health cannot establish completion of a failed PUT.
            # Keep the registered seed alive and do not unwind its allocation.
            self.quarantine("uncertain_batch_put_completion", error)
        self.table = self.runtime.prepare_access_table(keys)
        self.device_table = self.runtime.access_table_view(self.table, device="npu", capacity=capacity)
        if temporary_only:
            if cache_tokens != TOPK:
                raise ValueError("v5_temporary_capacity_must_be_topk")
            self.block_table = torch.arange(batch * TOPK // BLOCK_SIZE, dtype=torch.int32, device=device).reshape(
                batch, -1
            )
            self.initial_cpu = torch.full((batch * TOPK, TOKEN_BYTES), GUARD_BYTE, dtype=torch.uint8)
            self.initial = self.initial_cpu.to(device)
            self.output = self.initial.clone()
            self.slot_rows = [torch.arange(row * TOPK, (row + 1) * TOPK) for row in range(batch)]
            # The fused kernel generates each group plan. Never freeze final H
            # descriptors or launch a standalone GET during this setup.
            return
        resident_blocks = batch * cache_tokens // BLOCK_SIZE
        block_table_cpu = torch.arange(resident_blocks, 0, -1, dtype=torch.int32).reshape(batch, -1)
        self.block_table = block_table_cpu.to(device)
        output_rows = (resident_blocks + 2) * BLOCK_SIZE
        self.initial_cpu = torch.full((output_rows, TOKEN_BYTES), GUARD_BYTE, dtype=torch.uint8)
        self.slot_rows = []
        for row in range(batch):
            slots = torch.arange(cache_tokens, dtype=torch.long)
            physical = block_table_cpu[row].long()[slots // BLOCK_SIZE] * BLOCK_SIZE + slots % BLOCK_SIZE
            self.slot_rows.append(physical)
            self.initial_cpu.index_copy_(0, physical, self.cpu_payload[row, :cache_tokens])
        self.initial = self.initial_cpu.to(device)
        self.output = self.initial.clone()
        self.fixed_sources = source_ids.cpu().reshape(batch, TOPK).tolist()
        self.fixed_destinations = destination_slots.cpu().reshape(batch, TOPK).tolist()
        self.fixed_misses = miss_counts.cpu().tolist()
        for row in range(batch):
            if not 0 <= self.fixed_misses[row] <= TOPK:
                raise ValueError("native_fetch_invalid_miss_count")
            if any(not 0 <= token < candidate_tokens for token in self.fixed_sources[row]):
                raise ValueError("native_fetch_source_out_of_range")
            if any(not 0 <= slot < cache_tokens for slot in self.fixed_destinations[row]):
                raise ValueError("native_fetch_destination_slot_out_of_range")
        descriptor_count = batch * TOPK
        prepared_indices = torch.full((descriptor_count,), -1, dtype=torch.int32, device=device)
        prepared_offsets = torch.full_like(prepared_indices, -1)
        prepared_mask = torch.full((descriptor_count,), 0xA5, dtype=torch.uint8, device=device)
        prepared_outputs = torch.full_like(prepared_indices, -1)
        resident_lengths = torch.empty((batch,), dtype=torch.int32, device=device)
        torch.ops.nanovllm_dsa.kvaf_batch_get_prepare_out(
            source_ids,
            destination_slots,
            miss_counts,
            request_entries,
            self.block_table,
            cache_budgets,
            candidate_lens,
            candidate_lens,
            prepared_indices,
            prepared_offsets,
            prepared_mask,
            prepared_outputs,
            resident_lengths,
            0,
            1,
            batch,
            qcap,
        )
        torch.npu.synchronize()
        self.plan = compact_mask_plan(
            prepared_indices.cpu().tolist(),
            prepared_offsets.cpu().tolist(),
            prepared_mask.cpu().tolist(),
            prepared_outputs.cpu().tolist(),
        )
        expected_descriptors = []
        self.expected_cpu = self.initial_cpu.clone()
        for row in range(batch):
            count = self.fixed_misses[row]
            if not 0 <= count <= TOPK:
                raise ValueError("native_fetch_invalid_miss_count")
            for index in range(count):
                token = self.fixed_sources[row][index]
                slot = self.fixed_destinations[row][index]
                output_row = int(self.slot_rows[row][slot])
                expected_descriptors.append(
                    (row * qcap + token // BLOCK_SIZE, token % BLOCK_SIZE * TOKEN_BYTES, output_row)
                )
                self.expected_cpu[output_row] = self.cpu_payload[row, token]
        actual_descriptors = list(zip(self.plan["indices"], self.plan["offsets"], self.plan["output_indices"]))
        if actual_descriptors != expected_descriptors or self.plan["miss_count"] != sum(self.fixed_misses):
            raise ValueError("original_prepare_output_does_not_match_native_miss_plan")
        # Keep nonempty caller capacity for an all-hit row/batch: count=0
        # still makes the real BatchGet a valid no-read launch.
        self.indices = torch.tensor(self.plan["indices"] or [0], dtype=torch.int32, device=device)
        self.offsets = torch.tensor(self.plan["offsets"] or [0], dtype=torch.int32, device=device)
        self.output_indices = torch.tensor(self.plan["output_indices"] or [0], dtype=torch.int32, device=device)
        self.count = torch.tensor([self.plan["miss_count"]], dtype=torch.int32, device=device)
        self.launch()
        torch.npu.synchronize()
        self.verify()

    def launch(self) -> None:
        self.batch_get_attempted = True
        self.runtime.batch_get(
            self.device_table.addresses,
            self.device_table.valid,
            self.indices,
            self.offsets,
            self.count,
            self.output_indices,
            self.output,
            self.torch.npu.current_stream(),
            False,
        )

    def reset(self) -> None:
        self.output.copy_(self.initial)

    def verify(self) -> dict[str, Any]:
        if (
            self.sources.cpu().reshape(self.batch, TOPK).tolist() != self.fixed_sources
            or self.destinations.cpu().reshape(self.batch, TOPK).tolist() != self.fixed_destinations
            or self.misses.cpu().tolist() != self.fixed_misses
        ):
            raise ValueError("native_replay_outputs_changed_frozen_fetch_plan_invalid")
        if not self.torch.equal(self.output.cpu(), self.expected_cpu):
            raise ValueError("native_batchget_exact_payload_or_guard_mismatch")
        return {
            "payload_and_guards_verified": True,
            "prepared_plan_still_matches_li_update": True,
            "effective_fetch_tokens": self.plan["miss_count"],
            "effective_fetch_bytes": self.plan["miss_count"] * TOKEN_BYTES,
            "fetch_multiset_hash": hashlib.sha256(json.dumps(self.plan, sort_keys=True).encode()).hexdigest(),
        }

    @staticmethod
    def quarantine(reason: str, error: BaseException) -> None:
        print(
            f"[QLI_V5_NATIVE] status=failed stage=fetch_quarantine reason={reason} detail={str(error)[:240]!r}",
            flush=True,
        )
        os._exit(70)

    def close(self) -> None:
        try:
            self.torch.npu.synchronize()
        except BaseException as error:
            # Identical ownership boundary to the C++ V5 smoke: completion
            # is unknown, therefore do not reset graphs, unregister HBM,
            # release the DDR table, shut down or run Python destructors.
            self.quarantine("uncertain_stream_completion", error)
        if self.runtime is not None and self.batch_get_attempted:
            try:
                # An enqueue-only BatchGet can return from its kernel after
                # CQ timeout while DMA completion remains unknown. The real
                # fused-lease acquire/release validates persistent GET queue
                # health, unlike an ACL stream fence alone. No launches occur
                # during this setup-only probe; keep all graphs/tables alive.
                stream = self.torch.npu.current_stream()
                probe = self.runtime.acquire_fused_batch_get_context(self.output, stream)
                self.runtime.release_fused_batch_get_context(probe, stream)
            except BaseException as error:
                self.quarantine("unhealthy_batch_get_queue", error)
        for graph in self.graphs:
            graph.reset()
        self.graphs.clear()
        if self.runtime is not None:
            if self.table is not None:
                self.runtime.release_access_table(self.table)
                self.table = None
            self.runtime.shutdown()
            self.runtime = None
