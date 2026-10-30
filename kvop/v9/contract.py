"""ABI 13 for the standalone, equal-length V9 fused backend.

This module has no torch/NPU import. Binding/allocations happen before capture;
device-valued validation belongs to init, never to a Python .item() hot path.
"""

import ctypes
from dataclasses import dataclass
from enum import IntEnum

ABI_VERSION = 13
TOPK = 2048
MAX_QUERIES = 4
MAX_SELECTION = TOPK * MAX_QUERIES
TOKEN_BYTES = 656
BLOCK_TOKENS = 128
BITMAP_WORDS = 512
SUPPORTED_BUDGETS = (6144, 10240, 12288)


class Status(IntEnum):
    OK = 0
    SKIPPED = 1
    PENDING = 2
    INVALID_ARGUMENT = -1
    CAPACITY = -2
    DIRECTORY = -3
    COMMUNICATION = -4
    POISONED = -5


TENSOR_FIELDS = (
    "query",
    "key",
    "weights",
    "query_scale",
    "key_scale",
    "cumulative_query",
    "candidate_lens",
    "logical_lens",
    "index_table",
    "resident_table",
    "pool_rows",
    "budgets",
    "row_modes",
    "forward",
    "reverse",
    "cursors",
    "victims",
    "victim_bitmap",
    "address_table",
    "valid_table",
    "resident_output",
    "profile",
    "topk",
    "destination",
    "miss_counts",
    "resident_lengths",
    "transfers",
    "status",
    "extra_prefetch_slots",
    "extra_prefetch_counts",
    "index_cache",
    "index_dirty",
)
WORKSPACE_FIELDS = (
    "hot_bitmap",
    "hot_tokens",
    "hot_counts",
    "scores",
    "threshold_valid",
    "reserved_tokens",
    "reserved_slots",
    "planned_tokens",
    "planned_slots",
    "plan_count",
    "logical_misses",
    "proposed_cursor",
    "touched",
    "sticky",
    "error_detail",
    "eligible_victims",
    "eligible_counts",
    "hot_ready",
    "hot_thresholds",
    "cold_counts",
    "next_reverse",
    "mapping_epoch",
    "staged_destination",
    "changed_slots",
    "publish_meta",
    "kernel_timeline",
)
POINTER_FIELDS = TENSOR_FIELDS + WORKSPACE_FIELDS
GEOMETRY_FIELDS = (
    "batch",
    "queries",
    "candidate_capacity",
    "resident_capacity",
    "pool_capacity",
    "index_columns",
    "resident_columns",
    "index_blocks",
    "resident_blocks",
    "layer_id",
    "num_layers",
    "max_requests",
    "qcap",
    "table_capacity",
    "output_tokens",
    "score_stride",
)


class DeviceContract(ctypes.LittleEndianStructure):
    _fields_ = [(name, ctypes.c_uint64) for name in POINTER_FIELDS] + [
        (name, ctypes.c_uint32) for name in GEOMETRY_FIELDS
    ]


def align(value, multiple=256):
    return (value + multiple - 1) // multiple * multiple


@dataclass(frozen=True)
class Geometry:
    batch: int
    queries: int
    candidate_capacity: int
    resident_capacity: int
    pool_capacity: int
    index_columns: int
    resident_columns: int
    index_blocks: int
    resident_blocks: int
    layer_id: int
    num_layers: int
    max_requests: int
    qcap: int
    table_capacity: int
    output_tokens: int
    score_stride: int

    def validate(self):
        if not 1 <= self.batch <= 32 or self.queries != MAX_QUERIES:
            raise ValueError("V9 requires B1..32 and fixed MTP3/Q4")
        if not 256 <= self.candidate_capacity <= 262144 or self.candidate_capacity % 256:
            raise ValueError("candidate capacity must be 256-aligned in 256..262144")
        if not TOPK <= self.resident_capacity <= 16384 or self.resident_capacity % 256:
            raise ValueError("resident capacity exceeds the slot bitmap")
        if not 0 <= self.layer_id < self.num_layers:
            raise ValueError("invalid layer geometry")
        if not 0 < self.max_requests < self.pool_capacity:
            raise ValueError("pool must include a padding row")
        if self.max_requests < self.batch:
            raise ValueError("not enough distinct request rows")
        if self.index_columns * BLOCK_TOKENS < self.candidate_capacity:
            raise ValueError("index page table too narrow")
        if self.resident_columns * BLOCK_TOKENS < self.resident_capacity:
            raise ValueError("resident page table too narrow")
        if self.qcap * BLOCK_TOKENS < self.candidate_capacity:
            raise ValueError("directory qcap too narrow")
        if self.table_capacity < self.max_requests * self.num_layers * self.qcap:
            raise ValueError("directory too small for pool/layer geometry")
        if min(self.index_blocks, self.resident_blocks) <= 0:
            raise ValueError("physical pools must be nonempty")
        if self.output_tokens != self.resident_blocks * BLOCK_TOKENS:
            raise ValueError("registered output must cover the physical resident pool")
        if self.score_stride != self.candidate_capacity + BLOCK_TOKENS:
            raise ValueError("score stride must include one aligned hot/cold boundary tile")
        for name in GEOMETRY_FIELDS:
            if not 0 <= getattr(self, name) <= 0xFFFFFFFF:
                raise ValueError(f"{name} does not fit the ABI")
        return self

    def workspace_spec(self):
        b, c, s = self.batch, self.candidate_capacity, self.resident_capacity
        return {
            "hot_bitmap": ((b, (c + 31) // 32), "int32"),
            "hot_tokens": ((b, s), "int32"),
            "hot_counts": ((b,), "int32"),
            "scores": ((b, MAX_QUERIES, self.score_stride), "int16"),
            "threshold_valid": ((b, MAX_QUERIES), "int32"),
            "reserved_tokens": ((b, TOPK), "int32"),
            "reserved_slots": ((b, TOPK), "int32"),
            "planned_tokens": ((b, MAX_SELECTION), "int32"),
            "planned_slots": ((b, MAX_SELECTION), "int32"),
            "plan_count": ((b,), "int32"),
            "logical_misses": ((b,), "int32"),
            "proposed_cursor": ((b,), "int32"),
            "touched": ((b, BITMAP_WORDS), "int32"),
            "sticky": ((1,), "int32"),
            # error code, side effects, final-set-known, reserved for diagnostics
            "error_detail": ((b, 4), "int32"),
            "eligible_victims": ((b, TOPK), "int32"),
            "eligible_counts": ((b,), "int32"),
            "hot_ready": ((b, 32), "int32"),
            "hot_thresholds": ((b, 4), "int32"),
            "cold_counts": ((b,), "int32"),
            "next_reverse": ((b, s), "int32"),
            # Zero means unversioned: retain exact resident validation.
            "mapping_epoch": ((b,), "int32"),
            "staged_destination": ((b, 4, TOPK), "int32"),
            "changed_slots": ((b, s), "int32"),
            "publish_meta": ((b, 4), "int32"),
            # One isolated 128-byte record per kernel lane (start, end).
            "kernel_timeline": ((320, 16), "int64"),
        }

    def tensor_spec(self):
        b, q, c, s, p = (
            self.batch,
            self.queries,
            self.candidate_capacity,
            self.resident_capacity,
            self.pool_capacity,
        )
        i32 = "int32"
        specs = {
            "query": ((b * q, 64, 128), "float8_e4m3fn"),
            "key": ((self.index_blocks, 128, 1, 128), "float8_e4m3fn"),
            "weights": ((b * q, 64), "bfloat16"),
            "query_scale": ((b * q, 64), "float32"),
            "key_scale": ((self.index_blocks, 128, 1), "float32"),
            "index_table": ((b, self.index_columns), i32),
            "resident_table": ((b, self.resident_columns), i32),
            "forward": ((p, c), i32),
            "reverse": ((p, s), i32),
            "cursors": ((p,), i32),
            "victims": ((b, TOPK), i32),
            "victim_bitmap": ((b, BITMAP_WORDS), i32),
            "address_table": ((self.table_capacity,), "int64"),
            "valid_table": ((self.table_capacity,), "uint8"),
            "resident_output": ((self.resident_blocks, 128, TOKEN_BYTES), "uint8"),
            "profile": ((1,), i32),
            "topk": ((b * q, 1, TOPK), i32),
            "destination": ((b * q, 1, TOPK), i32),
            "miss_counts": ((b + 1,), i32),
            "resident_lengths": ((b,), i32),
            "transfers": ((b, 4), i32),
            "status": ((b,), i32),
            "extra_prefetch_slots": ((b, TOPK), i32),
            "extra_prefetch_counts": ((b,), i32),
            "index_cache": ((b,), "int64"),
            "index_dirty": ((b, 2), i32),
        }
        for name in (
            "cumulative_query",
            "candidate_lens",
            "logical_lens",
            "pool_rows",
            "budgets",
            "row_modes",
        ):
            specs[name] = ((b,), i32)
        return specs


def validate_buffers(geometry, tensors, workspace, *, device_type="npu"):
    """Validate metadata only; this is safe before capture and never reads values."""
    geometry.validate()
    expected = geometry.tensor_spec() | geometry.workspace_spec()
    supplied = dict(tensors) | dict(workspace)
    if set(tensors) != set(TENSOR_FIELDS) or set(workspace) != set(WORKSPACE_FIELDS):
        raise ValueError("contract tensor/workspace fields do not match ABI 13")
    device = tensors["query"].device
    if device.type != device_type:
        raise ValueError(f"contract requires {device_type} tensors")
    writable = set(WORKSPACE_FIELDS) | {
        "forward",
        "reverse",
        "cursors",
        "resident_output",
        "topk",
        "destination",
        "miss_counts",
        "resident_lengths",
        "transfers",
        "status",
    }
    ranges = []
    for name, (shape, dtype) in expected.items():
        tensor = supplied[name]
        if tuple(tensor.shape) != shape or str(tensor.dtype).split(".")[-1] != dtype:
            raise ValueError(f"{name}: expected {shape} {dtype}, got {tensor.shape} {tensor.dtype}")
        if tensor.device != device or not tensor.is_contiguous():
            raise ValueError(f"{name}: expected contiguous storage on {device}")
        begin = tensor.data_ptr()
        end = begin + tensor.numel() * tensor.element_size()
        for other, lo, hi in ranges:
            if begin < hi and lo < end and (name in writable or other in writable):
                raise ValueError(f"overlapping contract buffers: {other}, {name}")
        ranges.append((name, begin, end))


def descriptor_bytes(geometry, tensors, workspace):
    fields = {name: int(value.data_ptr()) for name, value in (dict(tensors) | dict(workspace)).items()}
    fields.update({name: getattr(geometry, name) for name in GEOMETRY_FIELDS})
    return bytes(DeviceContract(**fields))


def allocate_workspace(torch, geometry, device):
    return {
        name: torch.zeros(shape, dtype=getattr(torch, dtype), device=device)
        for name, (shape, dtype) in geometry.workspace_spec().items()
    }
