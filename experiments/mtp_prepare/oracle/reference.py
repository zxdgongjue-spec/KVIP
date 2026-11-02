#!/usr/bin/env python3
"""Accuracy, profiling, timing, and CPU contracts for fused MTP prepare."""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


SPARSE_COUNT = 2048
MIN_QUERIES_PER_REQUEST = 2
MAX_QUERIES_PER_REQUEST = 4
MTP_MISS_CAPACITY = SPARSE_COUNT * MAX_QUERIES_PER_REQUEST
MAX_CACHE_TOKENS = 16256
BITMAP_BLOCK_WORDS = 8
MISS_SLOT_BITMAP_WORDS = (
    ((MAX_CACHE_TOKENS + 31) // 32 + BITMAP_BLOCK_WORDS - 1)
    // BITMAP_BLOCK_WORDS
    * BITMAP_BLOCK_WORDS
)
BLOCK_SIZE = 128
PACKED_ROW_BYTES = 656
PROFILE_LANES = 8
PROFILE_FIELDS = (
    "total",
    "membership",
    "classify",
    "compact",
    "dedup",
    "clear",
    "install",
    "refill",
    "descriptor",
    "status",
    "unique_miss_count",
    "raw_miss_count",
    "query_begin",
    "query_count",
    "query0_miss_count",
    "query1_miss_count",
    "query2_miss_count",
    "query3_miss_count",
    "slot_select",
    *(f"reserved_{index}" for index in range(19, 32)),
)
OUTPUT_POISON = -777777
INDICES_POISON = -7001
OFFSETS_POISON = -7002
OUTPUT_INDICES_POISON = -7003
ERROR_UNIQUE_MISS_CAPACITY = -106
ERROR_INSUFFICIENT_EVICTION_SLOTS = -107


@dataclass
class CpuCase:
    topk: np.ndarray
    actual_seq_lengths_query: np.ndarray
    req_pool_entries: np.ndarray
    token_to_slot_pool: np.ndarray
    slot_to_token_pool: np.ndarray
    eviction_cursors: np.ndarray
    cache_tokens: np.ndarray
    candidate_lens: np.ndarray
    miss_slots: np.ndarray
    miss_slot_bitmap: np.ndarray
    resident_block_table: np.ndarray
    logical_seq_lengths: np.ndarray
    query_counts: list[int]
    requested_miss_counts: list[int]


@dataclass
class DeviceCase:
    topk: torch.Tensor
    actual_seq_lengths_query: torch.Tensor
    req_pool_entries: torch.Tensor
    token_to_slot_pool: torch.Tensor
    slot_to_token_pool: torch.Tensor
    eviction_cursors: torch.Tensor
    cache_tokens: torch.Tensor
    candidate_lens: torch.Tensor
    miss_slots: torch.Tensor
    miss_slot_bitmap: torch.Tensor
    resident_block_table: torch.Tensor
    logical_seq_lengths: torch.Tensor
    query_counts: list[int]
    requested_miss_counts: list[int]


@dataclass
class ReferenceResult:
    destination_slots: np.ndarray
    miss_counts: np.ndarray
    indices: np.ndarray
    offsets: np.ndarray
    output_indices: np.ndarray
    resident_seq_lengths: np.ndarray
    token_to_slot_pool: np.ndarray
    slot_to_token_pool: np.ndarray
    eviction_cursors: np.ndarray
    compacted_tokens: list[np.ndarray]
    query_miss_counts: list[list[int]]
    unique_sources: list[np.ndarray]
    resolved_miss_slots: list[np.ndarray]
    statuses: list[int]


def expand_per_request(
    values: Sequence[int], batch_size: int, option: str
) -> list[int]:
    if len(values) == 1:
        return [values[0]] * batch_size
    if len(values) != batch_size:
        raise ValueError(f"{option} needs one value or --batch-size values")
    return list(values)


def expand_query_counts(values: Sequence[int], batch_size: int) -> list[int]:
    if len(values) == 1 and values[0] == 0:
        # Match the MTP reference test: cover MTP1/MTP2/MTP3 by cycling
        # through 2/3/4 packed queries across requests.
        return [MIN_QUERIES_PER_REQUEST + index % 3 for index in range(batch_size)]
    return expand_per_request(values, batch_size, "--query-counts")


def _query_miss_sets(
    unique_misses: np.ndarray,
    query_count: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    if query_count <= 0:
        raise ValueError("query_count must be positive")
    miss_count = unique_misses.size
    if miss_count > query_count * SPARSE_COUNT:
        raise ValueError("unique misses exceed query_count * 2048")

    if miss_count > SPARSE_COUNT:
        # unique_misses is already randomized by build_case.
        chunks = np.array_split(unique_misses, query_count)
        return [chunk.copy() for chunk in chunks]

    # Assign each unique miss to one random query, then replicate some misses
    # to other queries to retain cross-query dedup coverage.
    shuffled = unique_misses[rng.permutation(miss_count)]
    query_tokens: list[list[int]] = [[] for _ in range(query_count)]
    owners = rng.integers(query_count, size=miss_count)
    if miss_count >= query_count:
        owners[:query_count] = np.arange(query_count)
        owners = owners[rng.permutation(miss_count)]
    for token, owner in zip(shuffled.tolist(), owners.tolist()):
        query_tokens[int(owner)].append(int(token))
        if query_count > 1 and rng.random() < 0.5:
            other = int(rng.integers(query_count - 1))
            if other >= int(owner):
                other += 1
            query_tokens[other].append(int(token))

    result: list[np.ndarray] = []
    for tokens in query_tokens:
        values = np.asarray(tokens, dtype=unique_misses.dtype)
        result.append(values[rng.permutation(values.size)])
    return result


def build_miss_slot_bitmap(miss_slots: np.ndarray) -> np.ndarray:
    bitmap = np.zeros(
        (miss_slots.shape[0], MISS_SLOT_BITMAP_WORDS), dtype=np.uint32
    )
    for row, slots in enumerate(miss_slots):
        for slot_value in slots.tolist():
            slot = int(slot_value)
            bitmap[row, slot >> 5] |= np.uint32(1) << np.uint32(slot & 31)
    return bitmap.view(np.int32)


def bitmap_contains(bitmap: np.ndarray, slots: np.ndarray) -> np.ndarray:
    slot_indices = slots.astype(np.int64, copy=False)
    words = bitmap[slot_indices >> 5].astype(np.uint32, copy=False)
    bits = (slot_indices & 31).astype(np.uint32, copy=False)
    return ((words >> bits) & np.uint32(1)).astype(np.bool_)


def _assert_reference_map_invariants(
    case: CpuCase, result: ReferenceResult
) -> None:
    """Check only mappings touched by successful requests."""

    ranges = _request_ranges(case.actual_seq_lengths_query)
    descriptor_ordinal = 0
    for batch_row, (query_begin, query_end) in enumerate(ranges):
        unique_sources = result.unique_sources[batch_row]
        unique_count = int(result.miss_counts[batch_row])
        pool_row = int(case.req_pool_entries[batch_row])
        if unique_count < 0:
            continue

        used_slots = result.resolved_miss_slots[batch_row]
        assert unique_count == unique_sources.size
        assert used_slots.size == unique_count
        assert np.unique(unique_sources).size == unique_count
        assert np.unique(used_slots).size == unique_count
        assert np.all(unique_sources >= 0)
        assert np.all(
            unique_sources < int(case.candidate_lens[batch_row])
        )
        assert np.all(used_slots >= 0)
        assert np.all(used_slots < int(case.cache_tokens[batch_row]))
        np.testing.assert_array_equal(
            result.token_to_slot_pool[pool_row, unique_sources], used_slots
        )
        np.testing.assert_array_equal(
            result.slot_to_token_pool[pool_row, used_slots], unique_sources
        )

        for local_query, query_row in enumerate(range(query_begin, query_end)):
            miss_count = result.query_miss_counts[batch_row][local_query]
            compacted = result.compacted_tokens[batch_row][local_query]
            assert 0 <= miss_count <= SPARSE_COUNT
            np.testing.assert_array_equal(
                result.destination_slots[query_row, 0, :miss_count],
                result.token_to_slot_pool[
                    pool_row, compacted[:miss_count]
                ],
            )

        descriptor_ordinal += unique_count

    assert int(result.miss_counts[-1]) == descriptor_ordinal
    np.testing.assert_array_equal(
        result.indices[descriptor_ordinal:], INDICES_POISON
    )
    np.testing.assert_array_equal(
        result.offsets[descriptor_ordinal:], OFFSETS_POISON
    )
    np.testing.assert_array_equal(
        result.output_indices[descriptor_ordinal:], OUTPUT_INDICES_POISON
    )


def build_case(
    batch_size: int,
    query_counts: Sequence[int],
    miss_counts: Sequence[int],
    source_capacity: int,
    cache_tokens: int,
    seed: int,
) -> CpuCase:
    """Build configurable packed MTP rows with exact unique miss counts."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(query_counts) != batch_size or len(miss_counts) != batch_size:
        raise ValueError("query_counts and miss_counts must contain B values")
    if any(
        count < MIN_QUERIES_PER_REQUEST or count > MAX_QUERIES_PER_REQUEST
        for count in query_counts
    ):
        raise ValueError("each query count must be in [2, 4]")
    if any(
        miss_count < 0 or miss_count > query_count * SPARSE_COUNT
        for miss_count, query_count in zip(miss_counts, query_counts)
    ):
        raise ValueError("each miss count must be in [0, query_count * 2048]")
    if source_capacity < cache_tokens or cache_tokens < 2 * SPARSE_COUNT:
        raise ValueError("need source_capacity >= cache_tokens >= 4096")
    if source_capacity % BLOCK_SIZE or cache_tokens % BLOCK_SIZE:
        raise ValueError("source_capacity and cache_tokens must be multiples of 128")

    pool_size = batch_size + 3
    request_rng = np.random.default_rng(seed)
    request_rows = request_rng.permutation(pool_size)[:batch_size].astype(np.int32)
    forward = np.full((pool_size, source_capacity), -1, dtype=np.int32)
    reverse = np.full((pool_size, cache_tokens), -1, dtype=np.int32)
    cursors = np.arange(pool_size, dtype=np.int32) * 97 + 11
    topk_rows: list[np.ndarray] = []
    miss_slot_rows: list[np.ndarray] = []
    cumulative_queries: list[int] = []
    packed_query_count = 0

    for batch_row, (query_count, requested_misses) in enumerate(
        zip(query_counts, miss_counts)
    ):
        rng = np.random.default_rng(seed + 1009 * (batch_row + 1))
        pool_row = int(request_rows[batch_row])
        forced_count = min(requested_misses // 5, SPARSE_COUNT)
        required_tokens = cache_tokens + requested_misses - forced_count
        if source_capacity < required_tokens:
            raise ValueError(
                "source_capacity is too small for cache_tokens plus unmapped misses"
            )

        token_order = rng.permutation(source_capacity).astype(np.int32)
        unique_misses = token_order[:requested_misses]
        query_misses = _query_miss_sets(unique_misses, query_count, rng)
        query_hit_capacity = max(
            SPARSE_COUNT - chunk.size for chunk in query_misses
        )
        hit_begin = requested_misses
        hit_end = hit_begin + query_hit_capacity
        hit_pool = token_order[hit_begin:hit_end]
        forced_tokens = unique_misses[:forced_count]
        filler_count = cache_tokens - forced_count - hit_pool.size
        filler_tokens = token_order[hit_end : hit_end + filler_count]
        resident_tokens = np.concatenate(
            (forced_tokens, hit_pool, filler_tokens)
        ).astype(np.int32)
        resident_slots = rng.permutation(cache_tokens).astype(np.int32)
        forward[pool_row, resident_tokens] = resident_slots
        reverse[pool_row, resident_slots] = resident_tokens

        forced_slots = forward[pool_row, forced_tokens].astype(np.int32)
        hit_slots = set(int(value) for value in forward[pool_row, hit_pool])
        forced_slot_set = set(int(value) for value in forced_slots)
        fill_candidates = np.asarray(
            [
                slot
                for slot in range(cache_tokens)
                if slot not in hit_slots and slot not in forced_slot_set
            ],
            dtype=np.int32,
        )
        fill_count = SPARSE_COUNT - forced_slots.size
        if fill_candidates.size < fill_count:
            raise ValueError("cache_tokens cannot provide 2048 non-hit miss slots")
        fill_slots = rng.choice(fill_candidates, size=fill_count, replace=False)
        row_miss_slots = np.concatenate((forced_slots, fill_slots)).astype(np.int32)
        row_miss_slots = row_miss_slots[rng.permutation(SPARSE_COUNT)]
        miss_slot_rows.append(row_miss_slots)

        for miss_tokens in query_misses:
            hit_count = SPARSE_COUNT - miss_tokens.size
            topk = np.concatenate((miss_tokens, hit_pool[:hit_count])).astype(np.int32)
            topk_rows.append(topk[rng.permutation(SPARSE_COUNT)][None, :])
        packed_query_count += query_count
        cumulative_queries.append(packed_query_count)

    resident_block_capacity = cache_tokens // BLOCK_SIZE
    resident_block_table = np.arange(
        batch_size * resident_block_capacity, dtype=np.int32
    ).reshape(batch_size, resident_block_capacity)
    resident_block_table = resident_block_table * 3 + 4096

    miss_slots = np.stack(miss_slot_rows)
    return CpuCase(
        topk=np.stack(topk_rows),
        actual_seq_lengths_query=np.asarray(cumulative_queries, dtype=np.int32),
        req_pool_entries=request_rows,
        token_to_slot_pool=forward,
        slot_to_token_pool=reverse,
        eviction_cursors=cursors,
        cache_tokens=np.full(batch_size, cache_tokens, dtype=np.int32),
        candidate_lens=np.full(batch_size, source_capacity, dtype=np.int32),
        miss_slots=miss_slots,
        miss_slot_bitmap=build_miss_slot_bitmap(miss_slots),
        resident_block_table=resident_block_table,
        logical_seq_lengths=(np.arange(batch_size, dtype=np.int32) + source_capacity),
        query_counts=list(query_counts),
        requested_miss_counts=list(miss_counts),
    )


def _request_ranges(actual_seq_lengths_query: np.ndarray) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    begin = 0
    for end_value in actual_seq_lengths_query.tolist():
        end = int(end_value)
        if end - begin not in (2, 3, 4):
            raise ValueError("each request must contain 2--4 packed queries")
        ranges.append((begin, end))
        begin = end
    return ranges


def reference_update(
    case: CpuCase,
    layer_id: int,
    num_layers: int,
    qcap: int,
) -> ReferenceResult:
    """Mirror membership/classify/compact/dedup/update/refill on CPU."""

    ranges = _request_ranges(case.actual_seq_lengths_query)
    if ranges[-1][1] != case.topk.shape[0]:
        raise ValueError("actual_seq_lengths_query must end at T")
    batch = len(ranges)
    forward = case.token_to_slot_pool.copy()
    reverse = case.slot_to_token_pool.copy()
    destination = np.full_like(case.topk, -1)
    counts = np.full(batch + 1, -1, dtype=np.int32)
    resident_lengths = np.zeros(batch, dtype=np.int32)
    descriptor_capacity = batch * MTP_MISS_CAPACITY
    indices = np.full(descriptor_capacity, INDICES_POISON, dtype=np.int32)
    offsets = np.full(descriptor_capacity, OFFSETS_POISON, dtype=np.int32)
    output_indices = np.full(descriptor_capacity, OUTPUT_INDICES_POISON, dtype=np.int32)
    compacted_by_request: list[np.ndarray] = []
    query_counts_by_request: list[list[int]] = []
    unique_by_request: list[np.ndarray] = []
    staged_destinations: list[list[np.ndarray]] = []
    resolved_slots_by_request: list[np.ndarray] = []
    request_succeeded: list[bool] = []
    proposed_cursors: list[int] = []
    request_statuses: list[int] = []

    # Match the kernel phases: all requests classify and compact against the
    # old maps before any request clears or installs a mapping.
    for batch_row, (query_begin, query_end) in enumerate(ranges):
        pool_row = int(case.req_pool_entries[batch_row])
        budget = int(case.cache_tokens[batch_row])
        query_compacted: list[np.ndarray] = []
        query_destinations: list[np.ndarray] = []
        query_miss_counts: list[int] = []
        unique: list[int] = []
        seen: set[int] = set()

        for query_row in range(query_begin, query_end):
            tokens = case.topk[query_row, 0]
            old_slots = forward[pool_row, tokens]
            mapped = (old_slots >= 0) & (old_slots < budget)
            forced = np.zeros(SPARSE_COUNT, dtype=np.bool_)
            forced[mapped] = bitmap_contains(
                case.miss_slot_bitmap[batch_row], old_slots[mapped]
            )
            is_miss = ~mapped | forced
            miss_tokens = tokens[is_miss]
            hit_tokens = tokens[~is_miss]
            hit_slots = old_slots[~is_miss]
            compacted = np.concatenate((miss_tokens, hit_tokens)).astype(np.int32)
            query_compacted.append(compacted)
            query_miss_counts.append(int(miss_tokens.size))
            query_destinations.append(
                np.concatenate(
                    (
                        np.full(miss_tokens.size, -1, dtype=np.int32),
                        hit_slots.astype(np.int32),
                    )
                )
            )
            for token_value in miss_tokens.tolist():
                token = int(token_value)
                if token not in seen:
                    seen.add(token)
                    unique.append(token)

        compacted_by_request.append(np.stack(query_compacted))
        query_counts_by_request.append(query_miss_counts)
        unique_by_request.append(np.asarray(unique, dtype=np.int32))
        staged_destinations.append(query_destinations)

    # Resolve the complete unique-source list before any request mutates maps.
    # The <=2048 path continues to use the input slots directly. Extended
    # requests protect input miss slots and every query's compacted hit suffix,
    # then select zero bits in circular cursor order.
    for batch_row, unique_sources in enumerate(unique_by_request):
        unique_count = int(unique_sources.size)
        if unique_count > MTP_MISS_CAPACITY:
            resolved_slots_by_request.append(np.empty(0, dtype=np.int32))
            request_succeeded.append(False)
            proposed_cursors.append(-1)
            request_statuses.append(ERROR_UNIQUE_MISS_CAPACITY)
            continue
        if unique_count <= SPARSE_COUNT:
            resolved_slots_by_request.append(
                case.miss_slots[batch_row, :unique_count].copy()
            )
            request_succeeded.append(True)
            proposed_cursors.append(-1)
            request_statuses.append(0)
            continue

        pool_row = int(case.req_pool_entries[batch_row])
        budget = int(case.cache_tokens[batch_row])
        protected = bitmap_contains(
            case.miss_slot_bitmap[batch_row],
            np.arange(budget, dtype=np.int32),
        )
        for local_query, row_destination in enumerate(
            staged_destinations[batch_row]
        ):
            miss_count = query_counts_by_request[batch_row][local_query]
            protected[row_destination[miss_count:]] = True

        cursor_value = int(case.eviction_cursors[pool_row])
        scan_begin = cursor_value % budget if cursor_value >= 0 else 0
        extra_count = unique_count - SPARSE_COUNT
        selected: list[int] = []
        for step in range(budget):
            slot = (scan_begin + step) % budget
            if not protected[slot]:
                selected.append(slot)
                if len(selected) == extra_count:
                    break
        if len(selected) != extra_count:
            resolved_slots_by_request.append(np.empty(0, dtype=np.int32))
            request_succeeded.append(False)
            proposed_cursors.append(-1)
            request_statuses.append(ERROR_INSUFFICIENT_EVICTION_SLOTS)
            continue

        extra_slots = np.asarray(selected, dtype=np.int32)
        resolved_slots_by_request.append(
            np.concatenate((case.miss_slots[batch_row], extra_slots))
        )
        request_succeeded.append(True)
        proposed_cursors.append((selected[-1] + 1) % budget)
        request_statuses.append(0)

    # Global clear phase.
    for batch_row, unique_sources in enumerate(unique_by_request):
        if not request_succeeded[batch_row]:
            continue
        pool_row = int(case.req_pool_entries[batch_row])
        candidate_len = int(case.candidate_lens[batch_row])
        used_slots = resolved_slots_by_request[batch_row]
        victims = reverse[pool_row, used_slots].copy()
        for victim_value, slot_value in zip(victims.tolist(), used_slots.tolist()):
            victim = int(victim_value)
            slot = int(slot_value)
            if 0 <= victim < candidate_len and forward[pool_row, victim] == slot:
                forward[pool_row, victim] = -1

    # Global install and per-query refill phase.
    for batch_row, ((query_begin, query_end), unique_sources) in enumerate(
        zip(ranges, unique_by_request)
    ):
        if not request_succeeded[batch_row]:
            destination[query_begin:query_end] = -1
            continue
        pool_row = int(case.req_pool_entries[batch_row])
        used_slots = resolved_slots_by_request[batch_row]
        if unique_sources.size:
            forward[pool_row, unique_sources] = used_slots
            reverse[pool_row, used_slots] = unique_sources
        for local_query, query_row in enumerate(range(query_begin, query_end)):
            row_destination = staged_destinations[batch_row][local_query]
            miss_count = query_counts_by_request[batch_row][local_query]
            if miss_count:
                miss_tokens = compacted_by_request[batch_row][local_query, :miss_count]
                row_destination[:miss_count] = forward[pool_row, miss_tokens]
            destination[query_row, 0] = row_destination
        counts[batch_row] = unique_sources.size
        cache_tokens = int(case.cache_tokens[batch_row])
        candidate_len = int(case.candidate_lens[batch_row])
        logical_len = int(case.logical_seq_lengths[batch_row])
        if (
            cache_tokens >= SPARSE_COUNT
            and candidate_len >= SPARSE_COUNT
            and logical_len >= candidate_len
        ):
            resident_lengths[batch_row] = cache_tokens + logical_len - candidate_len
        elif cache_tokens == 0 and logical_len >= 0:
            resident_lengths[batch_row] = logical_len

    cursors = case.eviction_cursors.copy()
    for batch_row, proposed_cursor in enumerate(proposed_cursors):
        if request_succeeded[batch_row] and proposed_cursor >= 0:
            pool_row = int(case.req_pool_entries[batch_row])
            cursors[pool_row] = proposed_cursor

    descriptor_ordinal = 0
    for batch_row, unique_sources in enumerate(unique_by_request):
        if not request_succeeded[batch_row]:
            continue
        pool_row = int(case.req_pool_entries[batch_row])
        used_slots = resolved_slots_by_request[batch_row]
        for source, slot in zip(unique_sources.tolist(), used_slots.tolist()):
            indices[descriptor_ordinal] = (
                pool_row * num_layers + layer_id
            ) * qcap + source // BLOCK_SIZE
            offsets[descriptor_ordinal] = (source % BLOCK_SIZE) * PACKED_ROW_BYTES
            physical_block = case.resident_block_table[batch_row, slot // BLOCK_SIZE]
            output_indices[descriptor_ordinal] = (
                int(physical_block) * BLOCK_SIZE + slot % BLOCK_SIZE
            )
            descriptor_ordinal += 1
    counts[batch] = descriptor_ordinal

    return ReferenceResult(
        destination_slots=destination,
        miss_counts=counts,
        indices=indices,
        offsets=offsets,
        output_indices=output_indices,
        resident_seq_lengths=resident_lengths,
        token_to_slot_pool=forward,
        slot_to_token_pool=reverse,
        eviction_cursors=cursors,
        compacted_tokens=compacted_by_request,
        query_miss_counts=query_counts_by_request,
        unique_sources=unique_by_request,
        resolved_miss_slots=resolved_slots_by_request,
        statuses=request_statuses,
    )


def to_device_case(case: CpuCase, device: torch.device) -> DeviceCase:
    def move(value: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(value).to(device)

    return DeviceCase(
        topk=move(case.topk),
        actual_seq_lengths_query=move(case.actual_seq_lengths_query),
        req_pool_entries=move(case.req_pool_entries),
        token_to_slot_pool=move(case.token_to_slot_pool),
        slot_to_token_pool=move(case.slot_to_token_pool),
        eviction_cursors=move(case.eviction_cursors),
        cache_tokens=move(case.cache_tokens),
        candidate_lens=move(case.candidate_lens),
        miss_slots=move(case.miss_slots),
        miss_slot_bitmap=move(case.miss_slot_bitmap),
        resident_block_table=move(case.resident_block_table),
        logical_seq_lengths=move(case.logical_seq_lengths),
        query_counts=case.query_counts,
        requested_miss_counts=case.requested_miss_counts,
    )


def load_npu_extension():
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Build A5LiduFusedBatchGetPrepareMtp and run this mode in an "
            "Ascend PyTorch environment"
        ) from exc

    try:
        from vllm_ascend.a5_dsa_registration import require_a5_dsa_registration

        require_a5_dsa_registration(
            (
                "lidu_fused_batch_get_prepare_mtp",
                "lidu_fused_batch_get_prepare_mtp_out",
            )
        )
        return torch.ops.nanovllm_dsa
    except (ImportError, RuntimeError) as root_extension_error:
        try:
            import nanovllm_dsa_lidu_fused_batch_get_prepare_mtp as extension
        except (ImportError, RuntimeError):
            raise RuntimeError(
                "Install vLLM-Ascend from the repository root or build the "
                "standalone A5LiduFusedBatchGetPrepareMtp extension"
            ) from root_extension_error
        return extension


def run_operator(
    case: DeviceCase,
    extension,
    profile_enabled: bool,
    layer_id: int,
    num_layers: int,
    max_num_reqs: int,
    qcap: int,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    batch = case.actual_seq_lengths_query.numel()
    descriptor_capacity = batch * MTP_MISS_CAPACITY
    profile_flag = torch.tensor(
        [int(profile_enabled)], dtype=torch.int32, device=case.topk.device
    )
    destination_slots = torch.full_like(case.topk, OUTPUT_POISON)
    miss_counts = torch.full(
        (batch + 1,), OUTPUT_POISON, dtype=torch.int32, device=case.topk.device
    )
    indices = torch.full(
        (descriptor_capacity,),
        INDICES_POISON,
        dtype=torch.int32,
        device=case.topk.device,
    )
    offsets = torch.full(
        (descriptor_capacity,),
        OFFSETS_POISON,
        dtype=torch.int32,
        device=case.topk.device,
    )
    output_indices = torch.full(
        (descriptor_capacity,),
        OUTPUT_INDICES_POISON,
        dtype=torch.int32,
        device=case.topk.device,
    )
    resident_seq_lengths = torch.full(
        (batch,), OUTPUT_POISON, dtype=torch.int32, device=case.topk.device
    )
    profiling = torch.zeros(
        (batch, PROFILE_LANES, len(PROFILE_FIELDS)),
        dtype=torch.int64,
        device=case.topk.device,
    )
    outputs = extension.lidu_fused_batch_get_prepare_mtp_out(
        case.topk,
        case.actual_seq_lengths_query,
        case.req_pool_entries,
        case.token_to_slot_pool,
        case.slot_to_token_pool,
        case.eviction_cursors,
        case.cache_tokens,
        case.candidate_lens,
        case.miss_slots,
        case.miss_slot_bitmap,
        case.resident_block_table,
        case.logical_seq_lengths,
        profile_flag,
        destination_slots,
        miss_counts,
        indices,
        offsets,
        output_indices,
        resident_seq_lengths,
        profiling,
        layer_id,
        num_layers,
        max_num_reqs,
        qcap,
    )
    torch.npu.synchronize()
    return outputs, profiling


def _assert_equal(name: str, actual: torch.Tensor, expected: np.ndarray) -> None:
    actual_cpu = actual.cpu()
    expected_cpu = torch.from_numpy(expected)
    if torch.equal(actual_cpu, expected_cpu):
        return
    mismatch = torch.nonzero(actual_cpu != expected_cpu, as_tuple=False)
    first = tuple(int(value) for value in mismatch[0])
    raise AssertionError(
        f"{name} differs at {first}: actual={int(actual_cpu[first])}, "
        f"expected={int(expected_cpu[first])}, mismatches={mismatch.size(0)}"
    )


def _assert_unordered_outputs(
    case: CpuCase,
    actual_case: DeviceCase,
    outputs: tuple[torch.Tensor, ...],
    reference: ReferenceResult,
    layer_id: int,
    num_layers: int,
    qcap: int,
) -> None:
    actual_destination = outputs[0].cpu().numpy()
    actual_indices = outputs[2].cpu().numpy()
    actual_offsets = outputs[3].cpu().numpy()
    actual_output_indices = outputs[4].cpu().numpy()
    actual_forward = actual_case.token_to_slot_pool.cpu().numpy()
    descriptor_ordinal = 0

    ranges = _request_ranges(case.actual_seq_lengths_query)
    for batch_row, (query_begin, query_end) in enumerate(ranges):
        unique_sources = reference.unique_sources[batch_row]
        unique_count = int(reference.miss_counts[batch_row])
        if unique_count < 0:
            if not np.all(actual_destination[query_begin:query_end] == -1):
                raise AssertionError(
                    f"row {batch_row}: invalid request destination is not poison"
                )
            continue

        pool_row = int(case.req_pool_entries[batch_row])
        used_slots = reference.resolved_miss_slots[batch_row]
        actual_slots = actual_forward[pool_row, unique_sources]
        if sorted(actual_slots.tolist()) != sorted(used_slots.tolist()):
            raise AssertionError(
                f"row {batch_row}: unique sources are not mapped to miss slots"
            )
        if set(
            actual_forward[pool_row, unique_sources].tolist()
        ) != set(used_slots.tolist()):
            raise AssertionError(
                f"row {batch_row}: duplicate final slot assignment"
            )

        for local_query, query_row in enumerate(range(query_begin, query_end)):
            miss_count = reference.query_miss_counts[batch_row][local_query]
            compacted = reference.compacted_tokens[batch_row][local_query]
            actual_miss_slots = actual_forward[
                pool_row, compacted[:miss_count]
            ]
            np.testing.assert_array_equal(
                actual_destination[query_row, 0, :miss_count],
                actual_miss_slots,
            )
            np.testing.assert_array_equal(
                actual_destination[query_row, 0, miss_count:],
                reference.destination_slots[query_row, 0, miss_count:],
            )

        descriptor_keys: set[tuple[int, int, int]] = set()
        expected_keys: set[tuple[int, int, int]] = set()
        layer_base = (pool_row * num_layers + layer_id) * qcap
        for source in unique_sources.tolist():
            slot = int(actual_forward[pool_row, source])
            physical_block = case.resident_block_table[
                batch_row, slot // BLOCK_SIZE
            ]
            expected_keys.add(
                (
                    layer_base + source // BLOCK_SIZE,
                    (source % BLOCK_SIZE) * PACKED_ROW_BYTES,
                    int(physical_block) * BLOCK_SIZE + slot % BLOCK_SIZE,
                )
            )
        for index, offset, output_index in zip(
            actual_indices[descriptor_ordinal : descriptor_ordinal + unique_count],
            actual_offsets[descriptor_ordinal : descriptor_ordinal + unique_count],
            actual_output_indices[
                descriptor_ordinal : descriptor_ordinal + unique_count
            ],
        ):
            if int(offset) % PACKED_ROW_BYTES != 0:
                raise AssertionError(
                    f"row {batch_row}: descriptor offset is not packed"
                )
            source = (
                (int(index) - layer_base) * BLOCK_SIZE
                + int(offset) // PACKED_ROW_BYTES
            )
            descriptor_keys.add((int(index), int(offset), int(output_index)))
            if source not in set(unique_sources.tolist()):
                raise AssertionError(
                    f"row {batch_row}: descriptor source is not a unique miss"
                )
        if descriptor_keys != expected_keys:
            raise AssertionError(
                f"row {batch_row}: unordered descriptor set mismatch"
            )
        descriptor_ordinal += unique_count

    np.testing.assert_array_equal(
        actual_indices[descriptor_ordinal:], INDICES_POISON
    )
    np.testing.assert_array_equal(
        actual_offsets[descriptor_ordinal:], OFFSETS_POISON
    )
    np.testing.assert_array_equal(
        actual_output_indices[descriptor_ordinal:], OUTPUT_INDICES_POISON
    )


def _assert_unordered_maps(
    case: CpuCase, actual_case: DeviceCase, reference: ReferenceResult
) -> None:
    actual_forward = actual_case.token_to_slot_pool.cpu().numpy()
    actual_reverse = actual_case.slot_to_token_pool.cpu().numpy()
    ranges = _request_ranges(case.actual_seq_lengths_query)
    for batch_row, _ in enumerate(ranges):
        unique_sources = reference.unique_sources[batch_row]
        unique_count = int(reference.miss_counts[batch_row])
        pool_row = int(case.req_pool_entries[batch_row])
        if unique_count < 0:
            np.testing.assert_array_equal(
                actual_forward[pool_row], reference.token_to_slot_pool[pool_row]
            )
            np.testing.assert_array_equal(
                actual_reverse[pool_row], reference.slot_to_token_pool[pool_row]
            )
            continue

        source_mask = np.ones(actual_forward.shape[1], dtype=np.bool_)
        source_mask[unique_sources] = False
        np.testing.assert_array_equal(
            actual_forward[pool_row, source_mask],
            reference.token_to_slot_pool[pool_row, source_mask],
        )

        used_slots = reference.resolved_miss_slots[batch_row]
        slot_mask = np.ones(actual_reverse.shape[1], dtype=np.bool_)
        slot_mask[used_slots] = False
        np.testing.assert_array_equal(
            actual_reverse[pool_row, slot_mask],
            reference.slot_to_token_pool[pool_row, slot_mask],
        )
        if set(actual_reverse[pool_row, used_slots].tolist()) != set(
            unique_sources.tolist()
        ):
            raise AssertionError(
                f"row {batch_row}: reverse map does not contain unique sources"
            )
        if set(actual_forward[pool_row, unique_sources].tolist()) != set(
            used_slots.tolist()
        ):
            raise AssertionError(
                f"row {batch_row}: forward map does not contain miss slots"
            )

        for token_value in unique_sources.tolist():
            token = int(token_value)
            slot = int(actual_forward[pool_row, token])
            if slot < 0 or slot >= actual_reverse.shape[1]:
                raise AssertionError(
                    f"row {batch_row}: forward map contains invalid slot"
                )
            if int(actual_reverse[pool_row, slot]) != token:
                raise AssertionError(
                    f"row {batch_row}: forward/reverse map mismatch "
                    f"for token={token}, slot={slot}"
                )

        for slot_value in used_slots.tolist():
            slot = int(slot_value)
            token = int(actual_reverse[pool_row, slot])
            if token < 0 or token >= actual_forward.shape[1]:
                raise AssertionError(
                    f"row {batch_row}: reverse map contains invalid token"
                )
            if int(actual_forward[pool_row, token]) != slot:
                raise AssertionError(
                    f"row {batch_row}: reverse/forward map mismatch "
                    f"for slot={slot}, token={token}"
                )


def decode_profiling(profiling: torch.Tensor, reference: ReferenceResult) -> None:
    profile = profiling.cpu()
    index = {name: field for field, name in enumerate(PROFILE_FIELDS)}
    print(
        "LIDU_FUSED_BATCH_GET_PREPARE_MTP_PROFILE_SCHEMA "
        "unit=device_cycles timing_aggregation=active_lane_max "
        "control_aggregation=leader_lane"
    )
    for row in range(profile.size(0)):
        active = profile[row][profile[row, :, index["total"]] > 0]
        if active.size(0) == 0:
            raise AssertionError(f"row {row}: no active profiling lanes")
        leader = active[0]
        expected_count = int(reference.miss_counts[row])
        expected_status = reference.statuses[row]
        status = int(leader[index["status"]])
        if status != expected_status:
            raise AssertionError(
                f"row {row}: profile status={status}, expected={expected_status}"
            )
        if int(leader[index["unique_miss_count"]]) != int(
            reference.unique_sources[row].size
        ):
            raise AssertionError(f"row {row}: profile unique count mismatch")

        def lane_max(field: str) -> int:
            return int(active[:, index[field]].max())

        print(
            "LIDU_FUSED_BATCH_GET_PREPARE_MTP_PROFILE "
            f"row={row} active_lanes={active.size(0)} status={status} "
            f"query_count={int(leader[index['query_count']])} "
            f"unique_misses={int(leader[index['unique_miss_count']])} "
            f"raw_misses={int(leader[index['raw_miss_count']])} "
            f"total={lane_max('total')} "
            f"membership={lane_max('membership')} "
            f"classify={lane_max('classify')} "
            f"compact={lane_max('compact')} "
            f"dedup={lane_max('dedup')} "
            f"slot_select={lane_max('slot_select')} "
            f"clear={lane_max('clear')} "
            f"install={lane_max('install')} "
            f"refill={lane_max('refill')} "
            f"descriptor={lane_max('descriptor')}"
        )


def check_case(
    case: CpuCase,
    device: torch.device,
    extension,
    profile_detail: bool,
    layer_id: int,
    num_layers: int,
    max_num_reqs: int,
    qcap: int,
) -> None:
    reference = reference_update(case, layer_id, num_layers, qcap)
    device_case = to_device_case(case, device)
    outputs, profiling = run_operator(
        device_case,
        extension,
        profile_detail,
        layer_id,
        num_layers,
        max_num_reqs,
        qcap,
    )
    if len(outputs) != 6:
        raise AssertionError(f"expected six public outputs, got {len(outputs)}")
    _assert_equal("miss_counts", outputs[1], reference.miss_counts)
    _assert_equal(
        "resident_seq_lengths", outputs[5], reference.resident_seq_lengths
    )
    _assert_unordered_outputs(
        case, device_case, outputs, reference, layer_id, num_layers, qcap
    )
    _assert_unordered_maps(case, device_case, reference)
    _assert_equal(
        "eviction_cursors",
        device_case.eviction_cursors,
        reference.eviction_cursors,
    )
    if profile_detail:
        decode_profiling(profiling, reference)
    print(
        "LIDU_FUSED_BATCH_GET_PREPARE_MTP_ACCURACY status=PASS "
        f"batch={case.actual_seq_lengths_query.size} "
        f"packed_queries={case.topk.shape[0]} "
        f"query_counts={case.query_counts} "
        f"unique_miss_counts={reference.miss_counts[:-1].tolist()} "
        f"total_miss_count={int(reference.miss_counts[-1])}"
    )


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def benchmark(
    case: CpuCase,
    device: torch.device,
    extension,
    warmup: int,
    iters: int,
    layer_id: int,
    num_layers: int,
    max_num_reqs: int,
    qcap: int,
) -> None:
    device_case = to_device_case(case, device)
    initial_forward = device_case.token_to_slot_pool.clone()
    initial_reverse = device_case.slot_to_token_pool.clone()
    initial_cursors = device_case.eviction_cursors.clone()
    batch = case.actual_seq_lengths_query.size
    descriptor_capacity = batch * MTP_MISS_CAPACITY
    profile_flag = torch.zeros(1, dtype=torch.int32, device=device)
    destination_slots = torch.empty_like(device_case.topk)
    miss_counts = torch.empty(batch + 1, dtype=torch.int32, device=device)
    indices = torch.empty(descriptor_capacity, dtype=torch.int32, device=device)
    offsets = torch.empty(descriptor_capacity, dtype=torch.int32, device=device)
    output_indices = torch.empty(descriptor_capacity, dtype=torch.int32, device=device)
    resident_seq_lengths = torch.empty(batch, dtype=torch.int32, device=device)
    profiling = torch.zeros(
        (batch, PROFILE_LANES, len(PROFILE_FIELDS)),
        dtype=torch.int64,
        device=device,
    )

    def reset_state() -> None:
        device_case.token_to_slot_pool.copy_(initial_forward)
        device_case.slot_to_token_pool.copy_(initial_reverse)
        device_case.eviction_cursors.copy_(initial_cursors)

    def launch() -> None:
        extension.lidu_fused_batch_get_prepare_mtp_out(
            device_case.topk,
            device_case.actual_seq_lengths_query,
            device_case.req_pool_entries,
            device_case.token_to_slot_pool,
            device_case.slot_to_token_pool,
            device_case.eviction_cursors,
            device_case.cache_tokens,
            device_case.candidate_lens,
            device_case.miss_slots,
            device_case.miss_slot_bitmap,
            device_case.resident_block_table,
            device_case.logical_seq_lengths,
            profile_flag,
            destination_slots,
            miss_counts,
            indices,
            offsets,
            output_indices,
            resident_seq_lengths,
            profiling,
            layer_id,
            num_layers,
            max_num_reqs,
            qcap,
        )

    for _ in range(warmup):
        reset_state()
        launch()
    torch.npu.synchronize()

    starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        reset_state()
        start.record()
        launch()
        end.record()
    ends[-1].synchronize()
    samples_us = [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]
    print(
        "LIDU_FUSED_BATCH_GET_PREPARE_MTP_PERF "
        f"samples={iters} warmup_excluded={warmup} "
        f"batch={batch} packed_queries={case.topk.shape[0]} "
        f"query_counts={case.query_counts} "
        f"source_capacity={case.token_to_slot_pool.shape[1]} "
        f"cache_tokens={int(case.cache_tokens[0])} "
        f"unique_miss_counts={case.requested_miss_counts} "
        f"mean_us={statistics.mean(samples_us):.3f} "
        f"p50_us={percentile(samples_us, 0.50):.3f} "
        f"p90_us={percentile(samples_us, 0.90):.3f} "
        f"p99_us={percentile(samples_us, 0.99):.3f}",
        flush=True,
    )


def parse_args(values: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("check", "bench", "all"), default="all")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--query-counts",
        "--queries-per-request",
        dest="query_counts",
        type=int,
        nargs="+",
        default=[0],
        help="2/3/4, B values, or 0 to cycle through MTP1/MTP2/MTP3",
    )
    parser.add_argument(
        "--miss-counts",
        type=int,
        nargs="+",
        default=[1000],
        help="request-level unique misses; one value or B values",
    )
    parser.add_argument("--source-capacity", type=int, default=131072)
    parser.add_argument("--cache-tokens", type=int, default=12288)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--layer-id", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--profile-detail", action="store_true")
    return parser.parse_args(values)


def main() -> None:
    args = parse_args()
    query_counts = expand_query_counts(args.query_counts, args.batch_size)
    miss_counts = expand_per_request(args.miss_counts, args.batch_size, "--miss-counts")
    if args.num_layers <= 0 or not 0 <= args.layer_id < args.num_layers:
        raise ValueError("need num_layers > 0 and layer_id in [0, num_layers)")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("need warmup >= 0 and iters > 0")
    extension = load_npu_extension()
    device = torch.device(args.device)
    case = build_case(
        batch_size=args.batch_size,
        query_counts=query_counts,
        miss_counts=miss_counts,
        source_capacity=args.source_capacity,
        cache_tokens=args.cache_tokens,
        seed=args.seed,
    )
    max_num_reqs = case.token_to_slot_pool.shape[0]
    qcap = args.source_capacity // BLOCK_SIZE
    print(
        "LIDU_FUSED_BATCH_GET_PREPARE_MTP_CONFIG "
        f"device={device} batch={args.batch_size} "
        f"packed_queries={case.topk.shape[0]} query_counts={query_counts} "
        f"source_capacity={args.source_capacity} "
        f"cache_tokens={args.cache_tokens} "
        f"unique_miss_counts={miss_counts} "
        f"layer_id={args.layer_id} num_layers={args.num_layers} "
        f"max_num_reqs={max_num_reqs} qcap={qcap} "
        f"warmup={args.warmup} iters={args.iters}"
    )
    if args.mode in ("check", "all"):
        check_case(
            case,
            device,
            extension,
            args.profile_detail,
            args.layer_id,
            args.num_layers,
            max_num_reqs,
            qcap,
        )
    if args.mode in ("bench", "all"):
        benchmark(
            case,
            device,
            extension,
            args.warmup,
            args.iters,
            args.layer_id,
            args.num_layers,
            max_num_reqs,
            qcap,
        )


def test_compress_then_dedup_update_and_refill() -> None:
    case = build_case(
        batch_size=2,
        query_counts=[2, 3],
        miss_counts=[768, 1100],
        source_capacity=8192,
        cache_tokens=4096,
        seed=20260818,
    )
    original_forward = case.token_to_slot_pool.copy()
    result = reference_update(case, layer_id=1, num_layers=4, qcap=64)
    np.testing.assert_array_equal(
        result.miss_counts, np.asarray([768, 1100, 1868], dtype=np.int32)
    )
    ranges = _request_ranges(case.actual_seq_lengths_query)
    for batch_row, (query_begin, query_end) in enumerate(ranges):
        pool_row = int(case.req_pool_entries[batch_row])
        unique = result.unique_sources[batch_row]
        used_slots = result.resolved_miss_slots[batch_row]
        np.testing.assert_array_equal(
            result.token_to_slot_pool[pool_row, unique], used_slots
        )
        for local_query, query_row in enumerate(range(query_begin, query_end)):
            miss_count = result.query_miss_counts[batch_row][local_query]
            compacted = result.compacted_tokens[batch_row][local_query]
            np.testing.assert_array_equal(
                result.destination_slots[query_row, 0, :miss_count],
                result.token_to_slot_pool[pool_row, compacted[:miss_count]],
            )
            old_slots = original_forward[pool_row, case.topk[query_row, 0]]
            mapped = old_slots >= 0
            forced = np.zeros(SPARSE_COUNT, dtype=np.bool_)
            forced[mapped] = bitmap_contains(
                case.miss_slot_bitmap[batch_row], old_slots[mapped]
            )
            expected_hits = old_slots[mapped & ~forced]
            np.testing.assert_array_equal(
                result.destination_slots[query_row, 0, miss_count:],
                expected_hits,
            )
    total_unique = int(result.miss_counts[-1])
    assert np.all(result.indices[:total_unique] >= 0)
    assert np.all(result.indices[total_unique:] == INDICES_POISON)
    assert np.all(result.offsets[total_unique:] == OFFSETS_POISON)
    assert np.all(result.output_indices[total_unique:] == OUTPUT_INDICES_POISON)


def test_unique_miss_boundaries_zero_one_and_2047_preserve_maps() -> None:
    layer_id = 1
    num_layers = 4
    qcap = 64
    case = build_case(
        batch_size=3,
        query_counts=[2, 3, 4],
        miss_counts=[0, 1, SPARSE_COUNT - 1],
        source_capacity=8192,
        cache_tokens=4096,
        seed=20260831,
    )
    zero_pool_row = int(case.req_pool_entries[0])
    original_zero_forward = case.token_to_slot_pool[zero_pool_row].copy()
    original_zero_reverse = case.slot_to_token_pool[zero_pool_row].copy()
    original_zero_cursor = int(case.eviction_cursors[zero_pool_row])
    result = reference_update(case, layer_id, num_layers, qcap)
    assert result.statuses == [0, 0, 0]
    assert result.miss_counts.tolist() == [
        0,
        1,
        SPARSE_COUNT - 1,
        SPARSE_COUNT,
    ]
    _assert_reference_map_invariants(case, result)
    np.testing.assert_array_equal(
        result.token_to_slot_pool[zero_pool_row], original_zero_forward
    )
    np.testing.assert_array_equal(
        result.slot_to_token_pool[zero_pool_row], original_zero_reverse
    )
    assert int(result.eviction_cursors[zero_pool_row]) == original_zero_cursor

    descriptor_ordinal = 0
    for batch_row, unique_sources in enumerate(result.unique_sources):
        pool_row = int(case.req_pool_entries[batch_row])
        used_slots = result.resolved_miss_slots[batch_row]
        layer_base = (pool_row * num_layers + layer_id) * qcap
        for source, slot in zip(
            unique_sources.tolist(), used_slots.tolist()
        ):
            physical_block = case.resident_block_table[
                batch_row, slot // BLOCK_SIZE
            ]
            assert result.indices[descriptor_ordinal] == (
                layer_base + source // BLOCK_SIZE
            )
            assert result.offsets[descriptor_ordinal] == (
                source % BLOCK_SIZE
            ) * PACKED_ROW_BYTES
            assert result.output_indices[descriptor_ordinal] == (
                int(physical_block) * BLOCK_SIZE + slot % BLOCK_SIZE
            )
            descriptor_ordinal += 1


def test_mixed_query_shapes_cover_empty_query_and_cross_query_dedup() -> None:
    case = build_case(
        batch_size=3,
        query_counts=[2, 3, 4],
        miss_counts=[0, 1, SPARSE_COUNT - 1],
        source_capacity=8192,
        cache_tokens=4096,
        seed=20260832,
    )
    result = reference_update(case, layer_id=1, num_layers=4, qcap=64)
    assert [len(counts) for counts in result.query_miss_counts] == [2, 3, 4]
    assert result.query_miss_counts[0] == [0, 0]

    controlled = build_case(
        batch_size=1,
        query_counts=[4],
        miss_counts=[1],
        source_capacity=8192,
        cache_tokens=4096,
        seed=20260833,
    )
    baseline = reference_update(
        controlled, layer_id=1, num_layers=4, qcap=64
    )
    non_empty_queries = [
        local_query
        for local_query, count in enumerate(
            baseline.query_miss_counts[0]
        )
        if count > 0
    ]
    assert non_empty_queries
    if len(non_empty_queries) == 1:
        owner = non_empty_queries[0]
        empty_query = next(
            local_query
            for local_query in range(4)
            if local_query != owner
        )
        miss_token = int(
            baseline.compacted_tokens[0][owner][0]
        )
        controlled.topk[empty_query, 0, 0] = miss_token

    result = reference_update(
        controlled, layer_id=1, num_layers=4, qcap=64
    )
    query_miss_counts = result.query_miss_counts[0]
    assert len(query_miss_counts) == 4
    assert any(count == 0 for count in query_miss_counts)
    assert sum(query_miss_counts) > int(result.miss_counts[0])
    assert int(result.miss_counts[0]) == 1
    _assert_reference_map_invariants(controlled, result)


def test_more_than_2048_unique_misses_selects_unprotected_slots() -> None:
    case = build_case(
        batch_size=1,
        query_counts=[2],
        miss_counts=[SPARSE_COUNT + 1],
        source_capacity=8192,
        cache_tokens=4096,
        seed=20260819,
    )
    original_forward = case.token_to_slot_pool.copy()
    result = reference_update(case, layer_id=1, num_layers=4, qcap=64)
    assert result.unique_sources[0].size == SPARSE_COUNT + 1
    assert result.miss_counts.tolist() == [SPARSE_COUNT + 1, SPARSE_COUNT + 1]
    resolved_slots = result.resolved_miss_slots[0]
    np.testing.assert_array_equal(resolved_slots[:SPARSE_COUNT], case.miss_slots[0])
    extra_slot = int(resolved_slots[-1])
    assert not bool(bitmap_contains(
        case.miss_slot_bitmap[0], np.asarray([extra_slot], dtype=np.int32)
    )[0])
    all_hit_slots: set[int] = set()
    for local_query, query_row in enumerate(range(2)):
        miss_count = result.query_miss_counts[0][local_query]
        all_hit_slots.update(
            int(slot)
            for slot in result.destination_slots[
                query_row, 0, miss_count:
            ].tolist()
        )
    assert extra_slot not in all_hit_slots
    pool_row = int(case.req_pool_entries[0])
    unique_sources = result.unique_sources[0]
    np.testing.assert_array_equal(
        result.token_to_slot_pool[pool_row, unique_sources], resolved_slots
    )
    for token, slot in zip(unique_sources.tolist(), resolved_slots.tolist()):
        assert result.slot_to_token_pool[pool_row, slot] == token
    assert result.eviction_cursors[pool_row] == (extra_slot + 1) % 4096
    assert np.any(result.token_to_slot_pool != original_forward)


def test_2048_boundary_and_mixed_extended_requests() -> None:
    case = build_case(
        batch_size=3,
        query_counts=[2, 2, 3],
        miss_counts=[SPARSE_COUNT, SPARSE_COUNT + 1, 3000],
        source_capacity=16384,
        cache_tokens=8192,
        seed=20260829,
    )
    original_cursors = case.eviction_cursors.copy()
    result = reference_update(case, layer_id=1, num_layers=4, qcap=128)
    assert result.statuses == [0, 0, 0]
    assert result.miss_counts.tolist() == [
        SPARSE_COUNT,
        SPARSE_COUNT + 1,
        3000,
        7097,
    ]

    boundary_pool_row = int(case.req_pool_entries[0])
    assert result.eviction_cursors[boundary_pool_row] == original_cursors[
        boundary_pool_row
    ]
    np.testing.assert_array_equal(
        result.resolved_miss_slots[0], case.miss_slots[0]
    )

    for batch_row in (1, 2):
        resolved_slots = result.resolved_miss_slots[batch_row]
        unique_count = int(result.unique_sources[batch_row].size)
        assert resolved_slots.size == unique_count
        np.testing.assert_array_equal(
            resolved_slots[:SPARSE_COUNT], case.miss_slots[batch_row]
        )
        assert np.unique(resolved_slots).size == unique_count


def test_extended_slot_cursor_wraps_at_cache_boundary() -> None:
    case = build_case(
        batch_size=1,
        query_counts=[2],
        miss_counts=[SPARSE_COUNT + 1],
        source_capacity=8192,
        cache_tokens=4096,
        seed=20260830,
    )
    pool_row = int(case.req_pool_entries[0])
    ranges = _request_ranges(case.actual_seq_lengths_query)
    protected = bitmap_contains(
        case.miss_slot_bitmap[0], np.arange(4096, dtype=np.int32)
    )
    for query_row in range(*ranges[0]):
        tokens = case.topk[query_row, 0]
        old_slots = case.token_to_slot_pool[pool_row, tokens]
        mapped = (old_slots >= 0) & (old_slots < 4096)
        forced = np.zeros(SPARSE_COUNT, dtype=np.bool_)
        forced[mapped] = bitmap_contains(
            case.miss_slot_bitmap[0], old_slots[mapped]
        )
        protected[old_slots[mapped & ~forced]] = True
    eligible = np.flatnonzero(~protected)
    assert eligible.size > 0
    last_eligible = int(eligible[-1])
    assert last_eligible == 4095
    case.eviction_cursors[pool_row] = last_eligible

    result = reference_update(case, layer_id=1, num_layers=4, qcap=64)
    assert int(result.resolved_miss_slots[0][-1]) == last_eligible
    assert result.eviction_cursors[pool_row] == 0


def test_extended_miss_capacity_8192_succeeds() -> None:
    case = build_case(
        batch_size=1,
        query_counts=[4],
        miss_counts=[MTP_MISS_CAPACITY],
        source_capacity=32768,
        cache_tokens=12288,
        seed=20260826,
    )
    result = reference_update(case, layer_id=1, num_layers=4, qcap=256)
    assert result.miss_counts.tolist() == [MTP_MISS_CAPACITY, MTP_MISS_CAPACITY]
    resolved_slots = result.resolved_miss_slots[0]
    assert resolved_slots.size == MTP_MISS_CAPACITY
    assert np.unique(resolved_slots).size == MTP_MISS_CAPACITY
    assert np.all(resolved_slots >= 0)
    assert np.all(resolved_slots < 12288)
    assert np.all(result.indices[:MTP_MISS_CAPACITY] >= 0)


def test_extended_misses_fail_closed_when_slots_are_insufficient() -> None:
    case = build_case(
        batch_size=1,
        query_counts=[4],
        miss_counts=[MTP_MISS_CAPACITY],
        source_capacity=12288,
        cache_tokens=4096,
        seed=20260827,
    )
    original_forward = case.token_to_slot_pool.copy()
    original_reverse = case.slot_to_token_pool.copy()
    original_cursors = case.eviction_cursors.copy()
    result = reference_update(case, layer_id=1, num_layers=4, qcap=96)
    assert result.statuses == [ERROR_INSUFFICIENT_EVICTION_SLOTS]
    assert result.miss_counts.tolist() == [-1, 0]
    assert np.all(result.destination_slots == -1)
    np.testing.assert_array_equal(result.token_to_slot_pool, original_forward)
    np.testing.assert_array_equal(result.slot_to_token_pool, original_reverse)
    np.testing.assert_array_equal(result.eviction_cursors, original_cursors)
    assert np.all(result.indices == INDICES_POISON)
    assert np.all(result.offsets == OFFSETS_POISON)
    assert np.all(result.output_indices == OUTPUT_INDICES_POISON)


def test_default_parameter_configuration_matches_non_mtp() -> None:
    defaults = parse_args([])
    assert defaults.mode == "all"
    assert defaults.device == "npu:0"
    assert defaults.batch_size == 8
    assert defaults.query_counts == [0]
    assert expand_query_counts(defaults.query_counts, defaults.batch_size) == [
        2,
        3,
        4,
        2,
        3,
        4,
        2,
        3,
    ]
    assert defaults.miss_counts == [1000]
    assert defaults.source_capacity == 131072
    assert defaults.cache_tokens == 12288
    assert defaults.seed == 20260818
    assert defaults.layer_id == 2
    assert defaults.num_layers == 4
    assert defaults.warmup == 10
    assert defaults.iters == 100
    assert not defaults.profile_detail


def test_workspace_slices_are_distinct_and_8192_wide() -> None:
    operator_root = Path(__file__).resolve().parents[1] / (
        "a5_lidu_fused_batch_get_prepare_mtp"
    )
    kernel = (
        operator_root / "op_kernel" / "a5_lidu_fused_batch_get_prepare_mtp.cpp"
    ).read_text()
    host = (
        operator_root / "op_host" / "a5_lidu_fused_batch_get_prepare_mtp.cpp"
    ).read_text()
    for source in (kernel, host):
        assert "MTP_MISS_CAPACITY" in source
        assert "CLASSIFICATION_OFFSET = 0" in source
        assert (
            "COMPACTED_TOKEN_OFFSET =\n" "    CLASSIFICATION_OFFSET + MTP_MISS_CAPACITY"
        ) in source
        assert (
            "UNIQUE_SOURCE_OFFSET =\n" "    COMPACTED_TOKEN_OFFSET + MTP_MISS_CAPACITY"
        ) in source
        assert (
            "RESOLVED_SLOT_OFFSET =\n"
            "    UNIQUE_SOURCE_OFFSET + MTP_MISS_CAPACITY"
        ) in source
        assert (
            "ROW_CONTROL_OFFSET =\n"
            "    RESOLVED_SLOT_OFFSET + MTP_MISS_CAPACITY"
        ) in source
        assert "PRIVATE_MEMBERSHIP_OFFSET" not in source
        assert "MEMBERSHIP_STATUS_OFFSET" not in source


def test_extended_slot_selection_is_transactional_and_bitmap_driven() -> None:
    kernel_path = (
        Path(__file__).resolve().parents[1]
        / "a5_lidu_fused_batch_get_prepare_mtp"
        / "op_kernel"
        / "a5_lidu_fused_batch_get_prepare_mtp.cpp"
    )
    kernel = kernel_path.read_text()
    resolve_begin = kernel.index(
        "    __aicore__ inline int32_t ResolveExtendedMissSlots("
    )
    resolve_end = kernel.index(
        "\n    __aicore__ inline void StoreRowControl", resolve_begin
    )
    resolve = kernel[resolve_begin:resolve_end]
    assert "uniqueCount <= SPARSE_COUNT" in resolve
    assert "FindFirstSetBit(" in kernel
    assert "protectedSlots.GetValue(wordIndex)" in resolve
    assert "MergeHitSlotsSimt" in resolve
    assert "asc_ldcg(destinationSlots + queryBase + queryIndex)" in kernel
    assert "CountAvailableBitmapWordsSimt" in resolve
    assert "WriteAvailableBitmapSlotsSimt" in resolve
    assert "SLOT_SELECT_SCALAR_EXTRA_LIMIT" in resolve
    assert "SLOT_SELECT_SCAN_PREFIX_OFFSET" in resolve
    assert "orderedWordCount =\n            firstWordCount + secondWordCount" in resolve
    assert "LoadDestinationSlots" not in resolve
    assert "ERROR_INSUFFICIENT_EVICTION_SLOTS" in resolve
    assert "StoreWorkspaceVector(" in resolve

    prepare_begin = kernel.index(
        "    __aicore__ inline void PrepareRequestAfterClassification("
    )
    prepare_end = kernel.index(
        "\n    __aicore__ inline void ClearPreparedRequest", prepare_begin
    )
    assert "ResolveExtendedMissSlots(" in kernel[prepare_begin:prepare_end]
    process_begin = kernel.index("    __aicore__ inline void Process()")
    process_end = kernel.index("\nprivate:", process_begin)
    process = kernel[process_begin:process_end]
    assert process.index("PrepareRequestAfterClassification(") < process.index(
        "ClearPreparedRequest(row, lane, timing)"
    )
    assert "CONTROL_USES_RESOLVED_SLOTS" in kernel
    assert "RESOLVED_SLOT_OFFSET + column" in kernel
    assert "evictionCursorsGm_.SetValue(poolRow, proposedCursor)" in kernel


def test_simt_slot_write_clears_the_original_bitmap_bit() -> None:
    kernel_path = (
        Path(__file__).resolve().parents[1]
        / "a5_lidu_fused_batch_get_prepare_mtp"
        / "op_kernel"
        / "a5_lidu_fused_batch_get_prepare_mtp.cpp"
    )
    kernel = kernel_path.read_text()
    write_begin = kernel.index("WriteAvailableBitmapSlotsSimt(")
    write_end = kernel.index("\n}\n\n__simt_vf__", write_begin)
    write = kernel[write_begin:write_end]

    assert "uint32_t probe = available;" in write
    assert "probe >>= 16;" in write
    assert "available &= ~(1U << bit);" in write

    # The old implementation shifted `available` itself before clearing the
    # bit. For a high bit, that left the original bit set and emitted the same
    # slot repeatedly.
    available = 1 << 16
    probe = available
    bit = 0
    if (probe & 0x0000FFFF) == 0:
        probe >>= 16
        bit += 16
    assert bit == 16
    available &= ~(1 << bit)
    assert available == 0


def test_miss_slot_bitmap_is_explicit_and_drives_membership() -> None:
    case = build_case(
        batch_size=1,
        query_counts=[4],
        miss_counts=[1000],
        source_capacity=8192,
        cache_tokens=4096,
        seed=20260825,
    )
    assert case.miss_slot_bitmap.shape == (1, MISS_SLOT_BITMAP_WORDS)
    assert case.miss_slot_bitmap.dtype == np.int32
    np.testing.assert_array_equal(
        bitmap_contains(case.miss_slot_bitmap[0], case.miss_slots[0]),
        np.ones(SPARSE_COUNT, dtype=np.bool_),
    )

    without_forced_misses = build_case(
        batch_size=1,
        query_counts=[4],
        miss_counts=[1000],
        source_capacity=8192,
        cache_tokens=4096,
        seed=20260825,
    )
    without_forced_misses.miss_slot_bitmap.fill(0)
    result = reference_update(
        without_forced_misses, layer_id=1, num_layers=4, qcap=64
    )
    assert result.miss_counts.tolist() == [800, 800]

    operator_root = Path(__file__).resolve().parents[1] / (
        "a5_lidu_fused_batch_get_prepare_mtp"
    )
    kernel = (
        operator_root / "op_kernel" / "a5_lidu_fused_batch_get_prepare_mtp.cpp"
    ).read_text()
    host = (
        operator_root / "op_host" / "a5_lidu_fused_batch_get_prepare_mtp.cpp"
    ).read_text()
    extension = (
        operator_root
        / "torch_extension"
        / "csrc"
        / "lidu_fused_batch_get_prepare_mtp_ops.cpp"
    ).read_text()
    ops_json = (operator_root / "ops.json").read_text()
    assert "LoadMissSlotMembership(row)" in kernel
    assert "BuildPrivateMissSlotMembership" not in kernel
    assert 'this->Input("miss_slot_bitmap")' in host
    assert "constexpr size_t INPUT_COUNT = 13;" in host
    assert "missSlotBitmap.GetDim(1) != MISS_BITMAP_STORAGE_WORDS" in host
    assert "Tensor miss_slots, Tensor miss_slot_bitmap" in extension
    assert '"name": "miss_slot_bitmap"' in ops_json


def test_requests_use_power_of_two_multicore_lane_groups() -> None:
    operator_root = Path(__file__).resolve().parents[1] / (
        "a5_lidu_fused_batch_get_prepare_mtp"
    )
    host = (
        operator_root / "op_host" / "a5_lidu_fused_batch_get_prepare_mtp.cpp"
    ).read_text()
    kernel = (
        operator_root / "op_kernel" / "a5_lidu_fused_batch_get_prepare_mtp.cpp"
    ).read_text()

    assert "uint32_t SelectCoresPerRequest(uint32_t availableCores)" in host
    for lane_count in (8, 4, 2, 1):
        assert f"return {lane_count};" in host
    assert "coresPerRequest * static_cast<uint32_t>(batch)" in host
    assert "tiling->coresPerRequest = coresPerRequest" in host
    assert "const uint32_t row = coreIdx_ / tiling_->coresPerRequest" in kernel
    assert "const uint32_t lane = coreIdx_ % tiling_->coresPerRequest" in kernel


def test_batch_larger_than_aiv_count_uses_sequential_waves() -> None:
    operator_root = Path(__file__).resolve().parents[1] / (
        "a5_lidu_fused_batch_get_prepare_mtp"
    )
    host = (
        operator_root / "op_host" / "a5_lidu_fused_batch_get_prepare_mtp.cpp"
    ).read_text()
    kernel = (
        operator_root / "op_kernel" / "a5_lidu_fused_batch_get_prepare_mtp.cpp"
    ).read_text()

    assert "if (aivCount == 0)" in host
    assert "const bool useSequentialBatchAssignment" in host
    assert "? aivCount" in host
    assert "tiling_->batchSize > tiling_->usedCoreNum" in kernel
    assert "ProcessSequentialBatch()" in kernel
    assert "waveBegin += tiling_->usedCoreNum" in kernel
    assert "PrepareSequentialRow(row, timing)" in kernel

    process_begin = kernel.index(
        "    __aicore__ inline void ProcessSequentialBatch()"
    )
    process_end = kernel.index("\nprivate:", process_begin)
    process = kernel[process_begin:process_end]
    assert process.index("PrepareSequentialRow") < process.index(
        "ClearPreparedRequest"
    )
    assert process.index("ClearPreparedRequest") < process.index(
        "InstallPreparedRequest"
    )
    assert process.index("InstallPreparedRequest") < process.index(
        "RefillPreparedRequest"
    )


def test_multicore_stage_ownership_matches_non_mtp_flow() -> None:
    kernel_path = (
        Path(__file__).resolve().parents[1]
        / "a5_lidu_fused_batch_get_prepare_mtp"
        / "op_kernel"
        / "a5_lidu_fused_batch_get_prepare_mtp.cpp"
    )
    kernel = kernel_path.read_text()
    process_begin = kernel.index("    __aicore__ inline void Process()")
    process_end = kernel.index("\nprivate:", process_begin)
    process = kernel[process_begin:process_end]

    assert "LoadMissSlotMembership(row)" in process
    assert "BuildPrivateMissSlotMembership(" not in process
    assert "MergeMissSlotMembership(row, lane)" not in process
    assert process.count("localQuery = lane") == 1
    assert process.count("localQuery += tiling_->coresPerRequest") == 1
    assert "if (lane == 0)" in process
    assert "PrepareRequestAfterClassification(" in process

    prepare_begin = kernel.index(
        "    __aicore__ inline void PrepareRequestAfterClassification("
    )
    prepare_end = kernel.index(
        "\n    __aicore__ inline void ClearPreparedRequest", prepare_begin
    )
    prepare = kernel[prepare_begin:prepare_end]
    assert "localQuery = 0" in prepare
    assert "CompactQuery(" in prepare
    assert "DeduplicateCompactedQuery(" in prepare

    clear = process.index("ClearPreparedRequest(row, lane, timing)")
    install = process.index("InstallPreparedRequest(row, lane, timing)")
    refill = process.index("RefillPreparedRequest(row, lane, timing)")
    assert clear < process.index("GlobalSync();", clear) < install
    assert install < process.index("GlobalSync();", install) < refill
    descriptor = process.index("GenerateCompactDescriptorTask()")
    assert "GlobalSync();" not in process[refill:descriptor]
    assert "const uint64_t totalStart = ProfileStart();" in process

    assert "Simt::VF_CALL<ClearVictimsSimt>" in kernel
    assert "Simt::VF_CALL<InstallMapsSimt>" in kernel
    assert "lane, tiling_->coresPerRequest" in kernel
    assert "totalLines) * coreIdx_ /" in kernel
    assert "tiling_->usedCoreNum" in kernel


def test_dedup_uses_query_ordered_compacted_ub_tokens() -> None:
    kernel_path = (
        Path(__file__).resolve().parents[1]
        / "a5_lidu_fused_batch_get_prepare_mtp"
        / "op_kernel"
        / "a5_lidu_fused_batch_get_prepare_mtp.cpp"
    )
    kernel = kernel_path.read_text()
    init_begin = kernel.index("        // Host tiling publishes")
    init_end = kernel.index("\n    }\n\n    __aicore__ inline void Process", init_begin)
    init_buffers = kernel[init_begin:init_end]
    scalar_buffer_position = init_buffers.index("sourceMembershipBuf_")
    for vector_buffer in (
        "missMembershipBuf_",
        "topkBuf_",
        "classificationBuf_",
        "outputTokensBuf_",
        "outputSlotsBuf_",
        "controlBuf_",
    ):
        assert init_buffers.index(vector_buffer) < scalar_buffer_position
    assert "uniqueSourcesBuf_" in init_buffers[scalar_buffer_position:]
    assert (
        "Duplicate(membership, static_cast<uint32_t>(0), SOURCE_BITMAP_WORDS)"
        not in kernel
    )
    dedup_begin = kernel.index("    __aicore__ inline void InitializeDedup(")
    dedup_end = kernel.index(
        "\n    __aicore__ inline uint32_t FindFirstSetBit", dedup_begin
    )
    dedup = kernel[dedup_begin:dedup_end]
    assert "Simt::VF_CALL<DeduplicateOneQuerySimt>" in dedup
    assert "compactedTokens.GetPhyAddr()" in dedup
    assert "LoadWorkspaceScalar(" not in dedup
    assert "StoreWorkspaceVector(" in dedup
    simt_begin = kernel.index("DeduplicateOneQuerySimt(")
    simt_end = kernel.index("\n}\n\n__simt_vf__", simt_begin)
    simt = kernel[simt_begin:simt_end]
    membership_read = simt.index(
        "const uint32_t current = sourceMembership[wordIndex];"
    )
    membership_skip = simt.index("if ((current & mask) != 0U)")
    atomic_update = simt.index("const int32_t previous = asc_atomic_or(")
    assert membership_read < membership_skip < atomic_update
    assert "continue;" in simt[membership_skip:atomic_update]
    assert "ordinal < static_cast<int32_t>(MTP_MISS_CAPACITY)" in kernel
    assert (
        "const uint32_t storedCount = uniqueCount > MTP_MISS_CAPACITY"
        in dedup
    )
    assert "return uniqueCount > MTP_MISS_CAPACITY" in dedup
    assert "asc_atomic_or" in kernel
    assert "asc_atomic_add" in kernel
    assert "ClearSourceMembershipSimt" not in kernel
    assert "compactedTokens.GetValue(index)" not in dedup
    assert "CompactedTokenBase(" not in dedup


def test_destination_validation_is_simt_parallel() -> None:
    kernel_path = (
        Path(__file__).resolve().parents[1]
        / "a5_lidu_fused_batch_get_prepare_mtp"
        / "op_kernel"
        / "a5_lidu_fused_batch_get_prepare_mtp.cpp"
    )
    kernel = kernel_path.read_text()
    begin = kernel.index(
        "    __aicore__ inline int32_t ValidatePreparedDestinations("
    )
    end = kernel.index("\n    __aicore__ inline void StoreRowControl(", begin)
    validation = kernel[begin:end]
    assert "residentBlockTableGm_.GetValue" not in validation
    assert "Simt::VF_CALL<ValidatePreparedDestinationsSimt>" in validation
    assert "for (uint32_t index = 0; index < uniqueCount" not in validation


def test_miss_counts_includes_request_and_total_counts() -> None:
    operator_root = Path(__file__).resolve().parents[1] / (
        "a5_lidu_fused_batch_get_prepare_mtp"
    )
    host = (
        operator_root / "op_host" / "a5_lidu_fused_batch_get_prepare_mtp.cpp"
    ).read_text()
    kernel = (
        operator_root / "op_kernel" / "a5_lidu_fused_batch_get_prepare_mtp.cpp"
    ).read_text()
    extension = (
        operator_root
        / "torch_extension"
        / "csrc"
        / "lidu_fused_batch_get_prepare_mtp_ops.cpp"
    ).read_text()

    assert "context->GetOutputShape(1)->SetDim(0, batch + 1)" in host
    assert "missCounts, tiling_->batchSize + 1" in kernel
    assert "missCountsGm_[tiling_->batchSize]" in kernel
    assert "uint32_t totalMissCount = 0" in kernel
    assert "counts.SetValue(0, static_cast<int32_t>(totalMissCount))" in kernel
    assert "missCounts.size(0) == batch + 1" in extension
    assert "at::empty({batch + 1}, intOptions)" in extension
    assert "at::empty({batch + 1}, options)" in extension


def test_refill_is_query_sharded_and_simt_parallel() -> None:
    kernel_path = (
        Path(__file__).resolve().parents[1]
        / "a5_lidu_fused_batch_get_prepare_mtp"
        / "op_kernel"
        / "a5_lidu_fused_batch_get_prepare_mtp.cpp"
    )
    kernel = kernel_path.read_text()
    refill_begin = kernel.index("    __aicore__ inline void RefillMissPrefixes(")
    refill_end = kernel.index(
        "\n    __aicore__ inline void ClearPreparedRequest", refill_begin
    )
    refill = kernel[refill_begin:refill_end]
    assert "Simt::VF_CALL<RefillMissPrefixesSimt>" in refill
    assert "queryCount, lane, tiling_->coresPerRequest" in refill
    assert "tokenToSlotPoolGm_[tokenBase].GetPhyAddr()" in refill
    assert "FetchEventID(HardEvent::V_S)" in refill
    assert "workspaceGm_.GetValue(compactedBase + index)" not in refill
    assert "localQuery = lane" in kernel
    assert "asc_ldcg(tokenToSlot + token)" in kernel


def test_clear_and_install_own_forward_map_domains() -> None:
    kernel_path = (
        Path(__file__).resolve().parents[1]
        / "a5_lidu_fused_batch_get_prepare_mtp"
        / "op_kernel"
        / "a5_lidu_fused_batch_get_prepare_mtp.cpp"
    )
    kernel = kernel_path.read_text()
    clear_simt_begin = kernel.index("ClearVictimsSimt(")
    clear_simt_end = kernel.index("\n}\n\n__simt_vf__", clear_simt_begin)
    clear_simt = kernel[clear_simt_begin:clear_simt_end]
    assert "index = Simt::GetThreadIdx()" in clear_simt
    assert "lane * threadCount" not in clear_simt
    assert "victim >= static_cast<int32_t>(forwardBegin)" in clear_simt
    assert "victim < static_cast<int32_t>(forwardEnd)" in clear_simt

    clear_begin = kernel.index("    __aicore__ inline void ClearVictims(")
    clear_end = kernel.index("\n    __aicore__ inline void InstallMaps(", clear_begin)
    clear = kernel[clear_begin:clear_end]
    install_begin = clear_end
    install_end = kernel.index(
        "\n    __aicore__ inline void RefillMissPrefixes(", install_begin
    )
    install = kernel[install_begin:install_end]
    for domain_expression in (
        "lane * forwardShardCount / tiling_->coresPerRequest",
        "(lane + 1) * forwardShardCount / tiling_->coresPerRequest",
        "forwardShardBegin * MAP_SHARD_WORDS",
        "forwardShardEnd * MAP_SHARD_WORDS",
    ):
        assert domain_expression in clear
        assert domain_expression in install


def test_refill_reads_updated_forward_map_after_simt_fence() -> None:
    kernel_path = (
        Path(__file__).resolve().parents[1]
        / "a5_lidu_fused_batch_get_prepare_mtp"
        / "op_kernel"
        / "a5_lidu_fused_batch_get_prepare_mtp.cpp"
    )
    kernel = kernel_path.read_text()
    simt_begin = kernel.index("RefillMissPrefixesSimt(")
    simt_end = kernel.index(
        "\n// Descriptor tasks are partitioned", simt_begin
    )
    refill_simt = kernel[simt_begin:simt_end]
    assert "destinationSlots[queryBase + index]" in refill_simt
    assert "uniqueSources" not in refill_simt
    assert "uniqueSlots" not in refill_simt
    assert "asc_ldcg(tokenToSlot + token)" in refill_simt
    assert "CleanAndInvalidateTokenMapDCache" not in kernel
    assert "CacheLine::ENTIRE_DATA_CACHE" not in kernel
    assert '#include "simt_api/device_functions.h"' in kernel

    clear_begin = kernel.index("ClearVictimsSimt(")
    clear_end = kernel.index("\n}\n\n__simt_vf__", clear_begin)
    clear_simt = kernel[clear_begin:clear_end]
    install_begin = kernel.index("InstallMapsSimt(")
    install_end = kernel.index("\n}\n\n__simt_vf__", install_begin)
    install_simt = kernel[install_begin:install_end]
    assert "asc_threadfence();" in clear_simt
    assert "asc_threadfence();" in install_simt
    assert "asc_stcg(tokenToSlot + victim" in clear_simt
    assert "asc_stcg(tokenToSlot + token, slot)" in install_simt
    assert "asc_stcg(slotToToken + slot, token)" in install_simt

    install_position = kernel.index("InstallPreparedRequest(row, lane, timing)")
    sync_position = kernel.index("GlobalSync()", install_position)
    refill_position = kernel.index("RefillPreparedRequest", sync_position)
    assert install_position < sync_position < refill_position


def test_simt_classification_uses_configured_dynamic_ub() -> None:
    operator_root = Path(__file__).resolve().parents[1] / (
        "a5_lidu_fused_batch_get_prepare_mtp"
    )
    kernel_path = (
        operator_root / "op_kernel" / ("a5_lidu_fused_batch_get_prepare_mtp.cpp")
    )
    host_path = operator_root / "op_host" / ("a5_lidu_fused_batch_get_prepare_mtp.cpp")
    kernel = kernel_path.read_text()
    host = host_path.read_text()
    for source in (kernel, host):
        assert "constexpr uint32_t DYNAMIC_UB_BYTES" in source
        assert "DYNAMIC_UB_BYTES % UB_BLOCK_BYTES == 0" in source
    assert "context->SetDynUBufSize(DYNAMIC_UB_BYTES)" in host
    assert "!= ge::GRAPH_SUCCESS" in host
    assert "ClassifyTopkSimt(" in kernel
    assert "LoadTopk(queryRow, topk)" in kernel
    assert "classification[index] = slot" in kernel
    assert "TOPK_TILE" not in kernel
    assert "topkTileBuf_" not in kernel


def test_store_prepared_query_waits_for_mte3_completion() -> None:
    kernel_path = (
        Path(__file__).resolve().parents[1]
        / "a5_lidu_fused_batch_get_prepare_mtp"
        / "op_kernel"
        / "a5_lidu_fused_batch_get_prepare_mtp.cpp"
    )
    kernel = kernel_path.read_text()

    store_begin = kernel.index("    __aicore__ inline void StorePreparedQuery(")
    store_end = kernel.index(
        "\n    __aicore__ inline void StoreInvalidQueryRows", store_begin
    )
    store_query = kernel[store_begin:store_end]
    output_copy_position = store_query.index("destinationSlotsGm_[QueryBase(queryRow)]")
    completion_position = store_query.index(
        "SetFlag<HardEvent::MTE3_S>(EVENT_ID3)", output_copy_position
    )
    wait_position = store_query.index(
        "WaitFlag<HardEvent::MTE3_S>(EVENT_ID3)", completion_position
    )
    assert output_copy_position < completion_position < wait_position
    assert "HardEvent::MTE3_V" not in store_query
    assert "FlushWorkspaceDCache" not in kernel


def test_mtp_graph_padding_sentinel_is_strict_and_map_noop() -> None:
    kernel_path = (
        Path(__file__).resolve().parents[1]
        / "a5_lidu_fused_batch_get_prepare_mtp"
        / "op_kernel"
        / "a5_lidu_fused_batch_get_prepare_mtp.cpp"
    )
    kernel = kernel_path.read_text()

    sentinel_begin = kernel.index(
        "    __aicore__ inline bool IsMtpGraphPaddingRow("
    )
    sentinel_end = kernel.index(
        "\n    __aicore__ inline void LoadMissSlotMembership", sentinel_begin
    )
    sentinel = kernel[sentinel_begin:sentinel_end]
    for condition in (
        "budget == 0",
        "candidateLen == 0",
        "logicalSeqLengthsGm_.GetValue(row) == 1",
        "poolRow == static_cast<int32_t>(tiling_->maxNumReqs)",
    ):
        assert condition in sentinel

    layout_begin = kernel.index("    __aicore__ inline void PublishQueryLayoutStatus()")
    layout_end = kernel.index("\n    __aicore__ inline void LoadTopk", layout_begin)
    layout = kernel[layout_begin:layout_end]
    assert "const bool paddingRow = IsMtpGraphPaddingRow(row);" in layout
    assert "!(paddingRow && IsMtpGraphPaddingRow(previousRow))" in layout

    process_begin = kernel.index("    __aicore__ inline void Process()")
    process_end = kernel.index("\nprivate:", process_begin)
    process = kernel[process_begin:process_end]
    assert process.count("status == STATUS_OK && !paddingRow") == 2
    assert process.count("PrepareMtpGraphPaddingRow(") == 1
    sequential_begin = kernel.index(
        "    __aicore__ inline void PrepareSequentialRow("
    )
    sequential_end = kernel.index(
        "\n    __aicore__ inline uint64_t RequestBase", sequential_begin
    )
    sequential = kernel[sequential_begin:sequential_end]
    assert "PrepareMtpGraphPaddingRow(" in sequential

    padding_begin = kernel.index(
        "    __aicore__ inline void PrepareMtpGraphPaddingRow("
    )
    padding_end = kernel.index(
        "\n    __aicore__ inline void ClearVictims", padding_begin
    )
    padding = kernel[padding_begin:padding_end]
    assert "StoreInvalidQueryRows(queryBegin, queryCount);" in padding
    assert "row, STATUS_OK, 0, queryBegin, queryCount," in padding
    assert "queryMissCounts, false, -1" in padding

    clear_begin = kernel.index("    __aicore__ inline void ClearPreparedRequest(")
    clear_end = kernel.index(
        "\n    __aicore__ inline void InstallPreparedRequest", clear_begin
    )
    clear = kernel[clear_begin:clear_end]
    assert "uniqueCount <= 0" in clear
    install_begin = clear_end
    install_end = kernel.index(
        "\n    __aicore__ inline void RefillPreparedRequest", install_begin
    )
    install = kernel[install_begin:install_end]
    assert "uniqueCountValue <= 0" in install

    resident_begin = kernel.index(
        "    __aicore__ inline int32_t CalculateResidentLength("
    )
    resident_end = kernel.index(
        "\n    __aicore__ inline void LoadRowControl", resident_begin
    )
    resident = kernel[resident_begin:resident_end]
    assert "cacheTokens == 0 && logicalLen >= 0" in resident


if __name__ == "__main__":
    main()
