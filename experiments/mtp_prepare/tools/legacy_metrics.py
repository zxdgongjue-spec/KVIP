"""Vendored CPU-oracle checks and simulator profile export, without Git dependencies."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORACLE_PATH = ROOT / "oracle" / "reference.py"


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


TIMED_PHASES = (
    "membership",
    "classify",
    "compact",
    "dedup",
    "slot_select",
    "clear",
    "install",
    "refill",
    "descriptor",
)


def load_oracle():
    spec = importlib.util.spec_from_file_location("mtp_prepare_oracle", ORACLE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load CPU oracle from {ORACLE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_int32(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    result = np.fromfile(path, dtype="<i4")
    expected = int(np.prod(shape))
    if result.size != expected:
        raise AssertionError(f"{path}: got {result.size} int32, expected {expected}")
    return result.reshape(shape)


def assert_unordered_result(
    oracle,
    case,
    reference,
    run_dir: Path,
    *,
    check_eviction_cursors: bool = True,
) -> None:
    batch = len(case.query_counts)
    packed_queries = int(case.topk.shape[0])
    destination = read_int32(run_dir / "actual_destination_slots.bin", (packed_queries, 1, 2048))
    miss_counts = read_int32(run_dir / "actual_miss_counts.bin", (batch + 1,))
    indices = read_int32(run_dir / "actual_indices.bin", (batch * 8192,))
    offsets = read_int32(run_dir / "actual_offsets.bin", (batch * 8192,))
    output_indices = read_int32(run_dir / "actual_output_indices.bin", (batch * 8192,))
    resident_lengths = read_int32(run_dir / "actual_resident_seq_lengths.bin", (batch,))
    forward = read_int32(run_dir / "actual_token_to_slot_pool.bin", case.token_to_slot_pool.shape)
    reverse = read_int32(run_dir / "actual_slot_to_token_pool.bin", case.slot_to_token_pool.shape)
    cursors = read_int32(run_dir / "actual_eviction_cursors.bin", case.eviction_cursors.shape)

    np.testing.assert_array_equal(miss_counts, reference.miss_counts)
    np.testing.assert_array_equal(resident_lengths, reference.resident_seq_lengths)
    if check_eviction_cursors:
        np.testing.assert_array_equal(cursors, reference.eviction_cursors)

    descriptor_ordinal = 0
    ranges = oracle._request_ranges(case.actual_seq_lengths_query)
    for row, (query_begin, query_end) in enumerate(ranges):
        unique_sources = reference.unique_sources[row]
        unique_count = int(reference.miss_counts[row])
        pool_row = int(case.req_pool_entries[row])
        if unique_count < 0:
            if not np.all(destination[query_begin:query_end] == -1):
                raise AssertionError(f"row {row}: failed request destination not -1")
            continue
        used_slots = reference.resolved_miss_slots[row]
        actual_slots = forward[pool_row, unique_sources]
        if sorted(actual_slots.tolist()) != sorted(used_slots.tolist()):
            raise AssertionError(f"row {row}: unique-source slot set mismatch")
        if len(set(actual_slots.tolist())) != unique_count:
            raise AssertionError(f"row {row}: duplicate final slot assignment")

        for local_query, query_row in enumerate(range(query_begin, query_end)):
            miss_count = reference.query_miss_counts[row][local_query]
            compacted = reference.compacted_tokens[row][local_query]
            expected_miss_slots = forward[pool_row, compacted[:miss_count]]
            np.testing.assert_array_equal(destination[query_row, 0, :miss_count], expected_miss_slots)
            np.testing.assert_array_equal(
                destination[query_row, 0, miss_count:],
                reference.destination_slots[query_row, 0, miss_count:],
            )

        source_mask = np.ones(forward.shape[1], dtype=np.bool_)
        source_mask[unique_sources] = False
        np.testing.assert_array_equal(
            forward[pool_row, source_mask],
            reference.token_to_slot_pool[pool_row, source_mask],
        )
        slot_mask = np.ones(reverse.shape[1], dtype=np.bool_)
        slot_mask[used_slots] = False
        np.testing.assert_array_equal(
            reverse[pool_row, slot_mask],
            reference.slot_to_token_pool[pool_row, slot_mask],
        )
        if set(reverse[pool_row, used_slots].tolist()) != set(unique_sources.tolist()):
            raise AssertionError(f"row {row}: reverse map unique-source set mismatch")
        for token_value in unique_sources.tolist():
            token = int(token_value)
            slot = int(forward[pool_row, token])
            if slot < 0 or slot >= reverse.shape[1] or reverse[pool_row, slot] != token:
                raise AssertionError(f"row {row}: map inverse mismatch token={token}")

        layer_base = (pool_row * 4 + 2) * (case.token_to_slot_pool.shape[1] // 128)
        expected_descriptors = set()
        for source_value in unique_sources.tolist():
            source = int(source_value)
            slot = int(forward[pool_row, source])
            physical_block = int(case.resident_block_table[row, slot // 128])
            expected_descriptors.add(
                (
                    layer_base + source // 128,
                    (source % 128) * 656,
                    physical_block * 128 + slot % 128,
                )
            )
        actual_descriptors = set(
            zip(
                indices[descriptor_ordinal : descriptor_ordinal + unique_count].tolist(),
                offsets[descriptor_ordinal : descriptor_ordinal + unique_count].tolist(),
                output_indices[descriptor_ordinal : descriptor_ordinal + unique_count].tolist(),
            )
        )
        if actual_descriptors != expected_descriptors:
            raise AssertionError(f"row {row}: descriptor set mismatch")
        descriptor_ordinal += unique_count

    np.testing.assert_array_equal(indices[descriptor_ordinal:], -7001)
    np.testing.assert_array_equal(offsets[descriptor_ordinal:], -7002)
    np.testing.assert_array_equal(output_indices[descriptor_ordinal:], -7003)


def summarize_profile(run_dir: Path, batch: int, active: int) -> dict[str, Any]:
    profile = np.fromfile(run_dir / "profiling.bin", dtype="<i8")
    profile = profile.reshape(active, batch, 8, 32)
    totals = profile[..., 0].max(axis=(1, 2))
    result: dict[str, Any] = {
        "active_total_cycles": totals.tolist(),
        "median_total_cycles": float(np.median(totals)),
        "min_total_cycles": int(totals.min()),
        "max_total_cycles": int(totals.max()),
    }
    for phase in TIMED_PHASES:
        field = PROFILE_FIELDS.index(phase)
        values = profile[..., field].max(axis=(1, 2))
        result[f"median_{phase}_cycles"] = float(np.median(values))
    for counter in ("unique_miss_count", "raw_miss_count"):
        field = PROFILE_FIELDS.index(counter)
        values = profile[..., field].max(axis=(1, 2))
        result[f"median_{counter}"] = float(np.median(values))
    host_us = np.fromfile(run_dir / "host_microseconds.bin", dtype="<f8")
    result["host_microseconds_diagnostic"] = host_us.tolist()
    result["profile_shape"] = list(profile.shape)
    return result


def write_perfetto_trace(
    path: Path,
    variant: str,
    case_name: str,
    run_dir: Path,
    batch: int,
    active: int,
) -> None:
    profile = np.fromfile(run_dir / "profiling.bin", dtype="<i8").reshape(active, batch, 8, 32)
    totals = profile[..., 0].max(axis=(1, 2))
    sample = int(np.argmin(np.abs(totals - np.median(totals))))
    events: list[dict[str, Any]] = [
        {
            "name": "process_name",
            "ph": "M",
            "pid": 1,
            "tid": 0,
            "args": {"name": f"{variant}/{case_name}"},
        }
    ]
    for row in range(batch):
        for lane in range(8):
            lane_values = profile[sample, row, lane]
            if int(lane_values[0]) <= 0:
                continue
            tid = row * 8 + lane + 1
            events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": 1,
                    "tid": tid,
                    "args": {"name": f"request_{row}/lane_{lane}"},
                }
            )
            cursor = 0.0
            for phase in TIMED_PHASES:
                duration = int(lane_values[PROFILE_FIELDS.index(phase)])
                if duration <= 0:
                    continue
                events.append(
                    {
                        "name": phase,
                        "cat": "mtp_prepare_profile",
                        "ph": "X",
                        "pid": 1,
                        "tid": tid,
                        "ts": cursor,
                        "dur": float(duration),
                        "args": {
                            "unit": "simulator_cycle",
                            "reconstructed_order": True,
                            "sample": sample,
                        },
                    }
                )
                cursor += duration
            events.append(
                {
                    "name": "profile_total",
                    "cat": "mtp_prepare_total",
                    "ph": "X",
                    "pid": 1,
                    "tid": tid,
                    "ts": 0.0,
                    "dur": float(int(lane_values[0])),
                    "args": {
                        "unit": "simulator_cycle",
                        "phase_positions_are_reconstructed": True,
                    },
                }
            )
    path.write_text(
        json.dumps(
            {
                "traceEvents": events,
                "displayTimeUnit": "ns",
                "metadata": {
                    "time_unit": "simulator_cycle",
                    "warning": (
                        "Phase durations are device profile fields; phase start "
                        "positions are reconstructed and are not instruction timestamps."
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
