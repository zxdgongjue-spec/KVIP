"""Owned KVAF resources for the refactor's Q4 real-device benchmark.

Setup/PUT/address expansion are outside measurement. No simulator addresses or
synthetic WQE headers are reused. Queue leases and tensors outlive every launch.
This module does not import torch or contact a service until ``open`` is called.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from typing import Any

TOKEN_BYTES = 656
BLOCK_TOKENS = 128
QUERIES = 4
TOPK = 2048
PUT_CHUNK_BLOCKS = 128
GUARD_BYTE = 0xA5


def _support():
    import aiv1_fetch_support

    return aiv1_fetch_support


def output_geometry(batch: int, history: int, resident: int) -> dict[str, int]:
    if batch < 1 or history < TOPK or history % BLOCK_TOKENS:
        raise ValueError("real_fetch_requires_positive_batch_and_128_aligned_history")
    if not TOPK <= resident <= history or resident % BLOCK_TOKENS:
        raise ValueError("real_fetch_requires_aligned_resident_at_least_topk")
    row_tokens = resident + QUERIES * TOPK
    return {
        "batch": batch,
        "history": history,
        "resident": resident,
        "queries": QUERIES,
        "slots_per_query": TOPK,
        "row_tokens": row_tokens,
        "output_tokens": batch * row_tokens,
        "output_bytes": batch * row_tokens * TOKEN_BYTES,
        "blocks_per_row": history // BLOCK_TOKENS,
    }


class RealFetchSession:
    """Exclusive lease owner; never release memory on uncertain DMA completion."""

    @classmethod
    def open(
        cls,
        torch: Any,
        device: Any,
        *,
        batch: int,
        history: int,
        resident: int,
        server: str,
        get_kernel: Path | None = None,
        put_kernel: Path | None = None,
        kvaf_python_dir: Path | None = None,
        kvaf_root: Path | None = None,
        workers: int = 32,
    ):
        owner = cls(torch, device, batch, history, resident)
        try:
            owner._prepare(server, get_kernel, put_kernel, kvaf_python_dir, kvaf_root, workers)
            return owner
        except BaseException:
            owner.close()
            raise

    def __init__(self, torch, device, batch, history, resident):
        self.geometry = output_geometry(batch, history, resident)
        self.torch, self.device = torch, device
        self.batch, self.history, self.resident_budget = batch, history, resident
        self.support = _support()
        self.session = None
        self.lease = None
        self.lease_stream = None
        self.in_flight = False
        self.closed = False
        self.keepalive = []
        self.graphs = []

    def _prepare(self, server, get_kernel, put_kernel, python_dir, root, workers):
        config = self.support.discover(
            server=server, get_kernel=get_kernel, put_kernel=put_kernel, kvaf_python_dir=python_dir, kvaf_root=root
        )
        loaded = sys.modules.get("kvaf")
        if loaded is not None:
            origin = getattr(loaded, "__file__", None)
            if origin is None or not Path(origin).resolve().is_relative_to(config["kvaf_python_dir"]):
                raise ValueError("KVAF already imported from another runtime bundle")
        sys.path.insert(0, str(config["kvaf_python_dir"]))
        torch, device = self.torch, self.device
        placeholders = torch.zeros((self.batch, TOPK), dtype=torch.int32, device=device)
        counts = torch.zeros(self.batch, dtype=torch.int32, device=device)
        entries = torch.arange(self.batch, dtype=torch.int32, device=device)
        budgets = torch.full((self.batch,), TOPK, dtype=torch.int32, device=device)
        lengths = torch.full((self.batch,), self.history, dtype=torch.int32, device=device)
        # Existing class enforces 128-block PUT chunks and quarantines uncertain
        # PUT completion. Its Q1 temporary allocation is unused but retained.
        self.session = self.support._session_class().open(
            torch,
            device,
            self.batch,
            self.history,
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
            put_chunk_blocks=PUT_CHUNK_BLOCKS,
            npu_core_count=workers,
        )
        row_tokens = self.geometry["row_tokens"]
        initial = torch.full((self.batch, row_tokens, TOKEN_BYTES), GUARD_BYTE, dtype=torch.uint8)
        initial[:, : self.resident_budget].copy_(self.session.cpu_payload[:, : self.resident_budget])
        self.initial_output = initial.to(device)
        self.output = self.initial_output.clone()
        self.cache_map = torch.full((self.batch, self.history), -1, dtype=torch.int32, device=device)
        self.cache_map[:, : self.resident_budget] = torch.arange(self.resident_budget, dtype=torch.int32, device=device)
        # Prepared access-table addresses name DDR blocks, not individual KV.
        block_addresses = self.support.unsigned_address_view(self.session.device_table.addresses, torch)
        self.ddr_block_addresses = block_addresses
        self.ddr_valid = self.session.device_table.valid
        blocks = block_addresses.detach().cpu().reshape(-1).tolist()
        valid = self.ddr_valid.detach().cpu().reshape(-1).tolist()
        expected = self.batch * self.geometry["blocks_per_row"]
        if len(blocks) < expected or len(valid) < expected or not all(valid[:expected]):
            raise ValueError("remote access table incomplete")
        addresses = [
            [
                int(blocks[row * self.geometry["blocks_per_row"] + token // BLOCK_TOKENS])
                + (token % BLOCK_TOKENS) * TOKEN_BYTES
                for token in range(self.history)
            ]
            for row in range(self.batch)
        ]
        if any(not 0 < address < 2**64 for row in addresses for address in row):
            raise ValueError("invalid remote token address")
        self.remote_token_addresses = torch.tensor(addresses, dtype=torch.uint64, device=device)
        self.keepalive.extend(
            [
                placeholders,
                counts,
                entries,
                budgets,
                lengths,
                self.initial_output,
                self.output,
                self.cache_map,
                self.remote_token_addresses,
                block_addresses,
                self.ddr_valid,
            ]
        )
        torch.npu.synchronize()

    def acquire(self):
        if self.closed or self.lease is not None or self.in_flight:
            raise ValueError("invalid real-fetch lease transition")
        self.torch.npu.synchronize()
        self.lease_stream = self.torch.npu.current_stream()
        # KVAF validates bytes / shape[0] == token_bytes. Keep the convenient
        # [B, row_slots, bytes] owner layout, but register a zero-copy slot view.
        self.leased_output = self.output.reshape(self.geometry["output_tokens"], TOKEN_BYTES)
        if self.leased_output.data_ptr() != self.output.data_ptr():
            raise ValueError("real-fetch lease view must not copy output storage")
        self.lease = self.session.runtime.acquire_fused_batch_get_context(self.leased_output, self.lease_stream)
        for name in (
            "device_channel_entity",
            "device_queue_state",
            "device_core_status",
            "output_address",
            "output_bytes",
            "worker_count",
            "jettys_per_core",
        ):
            if type(self.lease.get(name)) is not int or self.lease[name] <= 0:
                raise ValueError(f"invalid lease field: {name}")
        if (
            self.lease["output_address"] != self.output.data_ptr()
            or self.lease["output_bytes"] != self.geometry["output_bytes"]
            or self.lease.get("token_bytes") != TOKEN_BYTES
            or self.lease.get("tokens_per_block") != BLOCK_TOKENS
        ):
            raise ValueError("leased buffer geometry mismatch")
        return dict(self.lease)

    def mark_launched(self, *tensors):
        if self.closed or self.lease is None:
            raise ValueError("launch requires a live queue lease")
        self.keepalive.extend(tensors)
        self.in_flight = True

    def build_device_args(self, abi_library: Path):
        """Serialize real header ABI, one independent single-jetty view/AIV.

        The lease has two jettys per AIC worker. Each physical AIV gets exactly
        one of them and private core status; no locks or shared SQ producer.
        AIV-to-view indexing must use physical AIV identity, not batch row.
        """
        if self.lease is None or self.in_flight:
            raise ValueError("build args requires an idle live lease")
        lease = self.lease
        if lease["jettys_per_core"] != 2:
            raise ValueError("refactor requires two independent jettys per AIC worker")
        lib = ctypes.CDLL(str(Path(abi_library).resolve()))
        sizes = {}
        for name in ("args", "status", "entity", "queue"):
            fn = getattr(lib, f"refactor_fetch_{name}_bytes")
            fn.argtypes, fn.restype = [], ctypes.c_uint32
            sizes[name] = int(fn())
            if sizes[name] <= 0:
                raise ValueError(f"invalid native ABI size: {name}")
        pack = lib.refactor_fetch_pack_args
        pack.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32]
        pack.restype = ctypes.c_uint32
        torch = self.torch
        owners = lease["worker_count"] * 2
        self.core_status = torch.zeros((owners, sizes["status"]), dtype=torch.uint8, device=self.device)
        # Not consumed by prepared-WQE submission, but valid storage keeps the
        # native args complete and available for final descriptor operations.
        self.descriptor_scratch = torch.zeros((4, QUERIES * TOPK), dtype=torch.int32, device=self.device)
        stride = QUERIES * TOPK * 4
        blobs = []
        for owner in range(owners):
            values = [
                self.ddr_block_addresses.data_ptr(),
                self.ddr_valid.data_ptr(),
                self.descriptor_scratch.data_ptr(),
                self.descriptor_scratch.data_ptr() + stride,
                self.descriptor_scratch.data_ptr() + 2 * stride,
                self.descriptor_scratch.data_ptr() + 3 * stride,
                self.output.data_ptr(),
                self.core_status.data_ptr() + owner * sizes["status"],
                lease["device_channel_entity"] + owner * sizes["entity"],
                lease["device_queue_state"] + owner * sizes["queue"],
                self.batch * self.geometry["blocks_per_row"],
                QUERIES * TOPK,
                self.geometry["output_tokens"],
                TOKEN_BYTES,
                BLOCK_TOKENS,
                1,
                1,
            ]
            fields = (ctypes.c_uint64 * len(values))(*values)
            output = ctypes.create_string_buffer(sizes["args"])
            result = pack(output, sizes["args"], fields, len(values))
            if result:
                raise ValueError(f"native ABI serialization failed: {result}")
            blobs.extend(output.raw)
        self.device_args = torch.tensor(blobs, dtype=torch.uint8, device=self.device).reshape(owners, sizes["args"])
        self.abi_library = lib
        self.abi_sizes = sizes
        self.keepalive.extend([self.core_status, self.descriptor_scratch, self.device_args, lib])
        self.torch.npu.synchronize()
        return self.device_args

    def mark_completed(self, *, completion_verified: bool):
        # Caller must inspect device CQ/drain status, not merely stream.sync.
        if not completion_verified:
            self.quarantine("real_fetch_completion_unverified", "device CQ status failed")
        self.in_flight = False

    def reset(self):
        if self.in_flight:
            self.quarantine("real_fetch_reset_inflight", "completion not verified")
        self.output.copy_(self.initial_output)

    def verify_locations(self, selected, locations):
        """Check final query indices resolve into hot/temporary, after CQ drain.

        ``locations`` are row-relative slots in the combined registered output.
        No restriction forbids two queries referring to the same fetched KV.
        """
        if self.in_flight:
            self.quarantine("verify_before_cq_drain", "pending communication")
        ids = selected.detach().cpu().reshape(self.batch, QUERIES, TOPK).long()
        slots = locations.detach().cpu().reshape_as(ids).long()
        row_tokens = self.geometry["row_tokens"]
        if bool(((ids < 0) | (ids >= self.history)).any()):
            raise ValueError("final token id outside history")
        if bool(((slots < 0) | (slots >= row_tokens)).any()):
            raise ValueError("final location outside registered row")
        values = self.output.detach().cpu().reshape(self.batch, row_tokens, TOKEN_BYTES)
        for row in range(self.batch):
            if not self.torch.equal(values[row, slots[row]], self.session.cpu_payload[row, ids[row]]):
                raise ValueError(f"final payload mismatch: row={row}")
        return {
            "payload_verified": True,
            "batch": self.batch,
            "queries": QUERIES,
            "query_slots_checked": self.batch * QUERIES * TOPK,
        }

    def release(self):
        if self.in_flight:
            self.quarantine("real_fetch_release_inflight", "completion not verified")
        if self.lease is not None:
            try:
                self.torch.npu.synchronize()
                self.session.runtime.release_fused_batch_get_context(self.lease, self.lease_stream)
                self.lease = None
            except BaseException as error:
                self.quarantine("real_fetch_release_failed", error)

    def quarantine(self, reason, error):
        self.support.quarantine_all(self, reason, error)

    def close(self):
        if self.closed:
            return
        if self.in_flight:
            self.quarantine("real_fetch_close_inflight", "pending communication")
        try:
            self.release()
            if self.session is not None:
                for graph in self.graphs:
                    graph.reset()
                self.graphs.clear()
                self.session.close()
                self.session = None
        except BaseException as error:
            self.quarantine("real_fetch_close_failed", error)
        self.closed = True
        self.keepalive.clear()
