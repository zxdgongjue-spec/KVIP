#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare checked, identical-demand fusion benchmark JSONs without importing NPU libraries."""

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

from support.benchmark_reads import fixed_read_workload


CONTRACT_INTS = (
    "batch",
    "queries_per_request",
    "mtp_speculative_tokens",
    "history_capacity",
    "valid_sequence_length",
    "resident",
    "token_bytes",
    "index_heads",
    "index_head_dim",
    "topk",
    "output_stride",
    "requested_windows",
    "prefetch_limit",
    "window_limit",
)
CONTRACT_TEXT = ("lru_order", "fixture_sha256", "scope", "visibility", "snapshot")
GEOMETRY_FIELDS = (
    "service_aics_per_window",
    "service_aivs_per_window",
    "service_groups",
    "li_tiling",
    "available_aic",
    "logical_aic_tasks",
    "cores_per_batch",
    "queries_per_aic",
    "queries_per_aiv",
    "rotating_aic",
    "rotating_aiv",
    "intermediate_fetches_requested",
    "intermediate_fetches",
    "service_aic",
)
READ_FIELDS = ("prefetch_hot", "prefetch_cold", "unused_hot", "unused_cold", "final", "total", "unused")
MATCH_FIELDS = (
    "contract",
    "geometry",
    "final_workers",
    "device",
    "soc",
    "host",
    "device_visibility",
    "kvip_commit",
    "final_cold_per_query",
    "distinct_final_cold_per_request",
    "ddr_slot_bytes",
    "warmup",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def finite_positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def validate(result, transport):
    """Check the producer's evidence before using its latency or work counters."""
    require(isinstance(result, dict), "result must be a JSON object")
    require(type(result.get("schema_version")) is int and result["schema_version"] == 1, "unsupported schema_version")
    require(result.get("status") == "pass", "benchmark did not pass its correctness checks")
    require(result.get("transport") == transport, f"expected transport={transport}")
    require(integer(result.get("device")), "device must be a nonnegative integer")
    for name in ("soc", "host", "kvip_commit"):
        require(isinstance(result.get(name), str) and bool(result[name]), f"missing {name}")
    require(isinstance(result.get("device_visibility"), str), "missing device_visibility")
    for name in ("final_workers", "ddr_slot_bytes"):
        require(integer(result.get(name), 1), f"invalid {name}")
    require(integer(result.get("warmup")), "invalid warmup")

    contract = result.get("contract")
    require(isinstance(contract, dict), "missing contract")
    for name in CONTRACT_INTS:
        require(integer(contract.get(name), 1), f"invalid contract.{name}")
    for name in CONTRACT_TEXT:
        require(isinstance(contract.get(name), str) and bool(contract[name]), f"missing contract.{name}")
    miss = contract.get("requested_miss_percent")
    require(type(miss) in (int, float) and math.isfinite(miss) and 0 <= miss <= 100, "invalid requested_miss_percent")
    require(contract["queries_per_request"] == 4 and contract["mtp_speculative_tokens"] == 3, "expected Q4/MTP3")
    require(contract["token_bytes"] in (656, 2624), "unsupported token_bytes")
    require(contract["scope"] == "LI+TopK+Fetch+IndexUpdate+TND_outputs", "unsupported timing scope")
    require(contract["snapshot"] == "restore_before_every_replay", "timing requires a restored snapshot")
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    require(result.get("comparison_key") == fingerprint, "comparison_key does not match contract")
    geometry = result.get("geometry")
    require(isinstance(geometry, dict), "missing geometry")
    for name in GEOMETRY_FIELDS:
        require(name in geometry, f"missing geometry.{name}")
    require(integer(geometry["available_aic"], 1), "invalid available_aic")

    cold = result.get("final_cold_per_query")
    require(isinstance(cold, list) and len(cold) == 4 and all(integer(v) for v in cold), "invalid final_cold_per_query")
    require(integer(result.get("distinct_final_cold_per_request")), "invalid distinct_final_cold_per_request")
    timing = result.get("timing")
    require(isinstance(timing, dict), "missing timing")
    samples = timing.get("samples_us")
    require(
        isinstance(samples, list) and bool(samples) and all(finite_positive(v) for v in samples), "invalid samples_us"
    )
    require(integer(timing.get("count"), 1) and timing["count"] == len(samples), "timing.count differs from samples_us")
    require(finite_positive(timing.get("p50_us")), "invalid p50_us")
    require(
        math.isclose(timing["p50_us"], statistics.median(samples), rel_tol=1e-9, abs_tol=1e-9),
        "p50_us differs from sample median",
    )

    reads = result.get("reads_per_replay")
    require(isinstance(reads, dict), "missing reads_per_replay")
    workload = fixed_read_workload(reads)
    require(reads["total"] > 0, "benchmark recorded no Fetch reads")
    read_samples = result.get("reads_samples")
    require(
        isinstance(read_samples, list) and len(read_samples) == len(samples),
        "reads_samples differs from timing sample count",
    )
    for index, sample in enumerate(read_samples):
        require(
            fixed_read_workload(sample) == workload,
            f"Fetch workload changed at timed replay {index}; only prefetch_cold/final redistribution is allowed",
        )
    return result


def load_results(paths, transport):
    results = {}
    for path in paths:
        try:
            result = validate(json.loads(Path(path).read_text()), transport)
            key = (result["device"], result["contract"]["token_bytes"])
            require(key not in results, f"duplicate device/width pair {key}; compare one shape per pair")
            results[key] = result
        except (OSError, ValueError) as error:
            raise ValueError(f"{path}: {error}") from error
    require(bool(results), f"no {transport} results")
    return results


def differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        return [
            difference
            for name in sorted(left.keys() | right.keys())
            for difference in differences(left.get(name), right.get(name), f"{prefix}.{name}" if prefix else name)
        ]
    return [] if left == right else [prefix]


def read_ranges(samples):
    return {
        name: dict(min=min(sample[name] for sample in samples), max=max(sample[name] for sample in samples))
        for name in READ_FIELDS
    }


def compare(ubmem, urma):
    require(ubmem.keys() == urma.keys(), f"device/width pairs differ: ubmem={sorted(ubmem)}, urma={sorted(urma)}")
    pairs = []
    for key in sorted(ubmem):
        local, remote = ubmem[key], urma[key]
        mismatch = differences(
            {name: local[name] for name in MATCH_FIELDS}, {name: remote[name] for name in MATCH_FIELDS}
        )
        if local["timing"]["count"] != remote["timing"]["count"]:
            mismatch.append("timing.count")
        require(not mismatch, f"device/width {key}: incomparable results; mismatched {', '.join(mismatch)}")
        local_us, remote_us = [statistics.median(row["timing"]["samples_us"]) for row in (local, remote)]
        local_reads, remote_reads = local["reads_per_replay"], remote["reads_per_replay"]
        local_samples, remote_samples = local["reads_samples"], remote["reads_samples"]
        equal_reads = all(sample == local_samples[0] for sample in (*local_samples, *remote_samples))
        equal_bytes = local_reads["total"] == remote_reads["total"]
        contract = local["contract"]
        pairs.append(
            dict(
                device=key[0],
                token_bytes=key[1],
                host=local["host"],
                device_visibility=local["device_visibility"],
                batch=contract["batch"],
                queries_per_request=contract["queries_per_request"],
                history_capacity=contract["history_capacity"],
                valid_sequence_length=contract["valid_sequence_length"],
                resident=contract["resident"],
                index_heads=contract["index_heads"],
                ubmem_p50_us=local_us,
                urma_p50_us=remote_us,
                urma_over_ubmem=remote_us / local_us,
                reduction_percent=100 * (remote_us - local_us) / remote_us,
                samples_per_transport=local["timing"]["count"],
                equal_actual_read_counts=equal_reads,
                equal_actual_read_bytes=equal_bytes,
                ubmem_reads_per_replay=local_reads,
                urma_reads_per_replay=remote_reads,
                ubmem_read_count_ranges=read_ranges(local_samples),
                urma_read_count_ranges=read_ranges(remote_samples),
                ubmem_read_bytes_per_replay=local_reads["total"] * key[1],
                urma_read_bytes_per_replay=remote_reads["total"] * key[1],
                interpretation=(
                    "Same-demand full fusion pipeline latency with equal recorded Fetch counts."
                    if equal_reads
                    else "Same-demand full fusion pipeline latency with equal total Fetch bytes; recorded Fetch counter breakdowns differ."
                    if equal_bytes
                    else "Same-demand full fusion pipeline latency includes different speculative Fetch traffic; not an equal-byte bandwidth comparison."
                ),
            )
        )
    return dict(schema_version=1, metric="fixed_snapshot_fusion_latency_us", comparisons=pairs, note="Not model TPOT.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ubmem", nargs="+", type=Path, required=True, help="Successful UBmem benchmark JSON files")
    parser.add_argument(
        "--urma", nargs="+", type=Path, required=True, help="Matching successful URMA benchmark JSON files"
    )
    args = parser.parse_args(argv)
    try:
        report = compare(load_results(args.ubmem, "ubmem"), load_results(args.urma, "urma"))
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
