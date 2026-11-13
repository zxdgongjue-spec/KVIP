#!/usr/bin/env python3
"""Structural workload accounting, NOT a latency simulator or a device run.

The 42 input cases come from the actual customer runner. Hardware, head count
and payload axes expand them to 336 analytical rows. Unknown runtime quantities
stay null. No cost coefficients, speedups or winning policy are invented.
"""

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = (
    "scripts/run_split_core_cases.py",
    "csrc/operators/lightning_indexer_with_fetch/include/runtime_shape.h",
    "csrc/operators/common/lifetch/sequence_schedule.h",
    "csrc/operators/common/lifetch/independent_schedule.h",
    "csrc/operators/lightning_indexer_with_fetch/op_kernel/streaming.cpp",
    "csrc/operators/lightning_indexer_with_fetch/op_kernel/resolver_staged_dma2.h",
    "csrc/operators/common/lifetch/split_topk.cpp",
    "csrc/operators/lightning_indexer_with_fetch/op_kernel/lifecycle.cpp",
    "csrc/operators/lightning_indexer_with_fetch/op_kernel/prefetch.h",
    "csrc/operators/lightning_indexer_with_fetch/op_kernel/pipeline_types.h",
    "csrc/operators/lightning_indexer_with_fetch/op_kernel/output_worker.h",
    "csrc/operators/lightning_indexer_with_fetch/op_kernel/shadow_exchange_parallel.h",
    "csrc/operators/lightning_indexer_with_fetch/op_kernel/staged_ubmem_fetch.h",
    "csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer/indexer.h",
    "csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer/indexer_vector.h",
    "csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer/indexer_score.h",
    "csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer/indexer_cube.h",
    "python/kvip/operators/lifetch/binding.py",
    "python/kvip/operators/lifetch_ubmem/binding.py",
    "tests/operators/lifetch/support/real_fetch_inputs.py",
)
H, R, Q, K, D = 131072, 16384, 4, 2048, 128


def customer_cases():
    spec = importlib.util.spec_from_file_location("customer_runner", ROOT / SOURCE_PATHS[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.cases())


def ceildiv(a, b):
    return (a + b - 1) // b


def account(case, aic=32, aiv=64, heads=32, token_bytes=656, fetch_width=2):
    """Account only for the fixed full-length, full-hot-valid customer fixture.

    Tail layouts with fewer roles below are *proposed budget counts only*.
    A fitting count does not establish UB fit, progress, ABI support or speed.
    """
    if (aic, aiv) not in ((32, 64), (28, 56)) or heads not in (32, 64):
        raise ValueError("unsupported analytical axis")
    if token_bytes not in (656, 2624) or fetch_width not in (1, 2, 4):
        raise ValueError("unsupported payload or Fetch width")
    if (case["history"], case["resident"], case["queries_per_request"]) != (H, R, Q):
        raise ValueError("accounting assumptions require full 128K/Q4 geometry")
    b, m = case["batch"], case["miss_per_query"]
    unique = 2 * m  # Controlled overlap fixture ONLY, not arbitrary Q4 input.
    split_q = 2 * b <= aic
    li = b * (1 + split_q)
    shape_ok = b <= aic
    rounds = min(8, li) if shape_ok else None
    p2 = bool(shape_ok and split_q and heads == 32)
    roles = 3 * b + 4 * fetch_width
    native = shape_ok and b <= 16 and 2 * li <= aiv and roles <= aiv
    reasons = []
    if b > aic:
        reasons.append("PrepareShape: B>C")
    if b > 16:
        reasons.append("public streaming ABI: B>16")
    if 2 * li > aiv:
        reasons.append("current mixed score: 2*LI>V")
    if roles > aiv:
        reasons.append("current tail: 3*B+4*W>V")
    row = dict(
        case=case["case"],
        batch=b,
        nominal_miss_percent=case["miss_percent"],
        miss_per_query=m,
        actual_miss_percent=100 * m / K,
        fixture_cold_union_per_request=unique,
        history=H,
        resident=R,
        queries=Q,
        heads=heads,
        head_dim=D,
        token_bytes=token_bytes,
        aic_budget=aic,
        aiv_budget=aiv,
        fetch_width=fetch_width,
        native_full_pipeline_supported=native,
        unsupported_reasons="; ".join(reasons),
        shape_li_aic=li if shape_ok else None,
        shape_li_aiv=2 * li if shape_ok else None,
        shape_history_shards=2 if p2 else (1 if shape_ok else None),
        shape_actual_rounds=rounds,
        # Hypothetical geometry above is not an executed native plan when invalid.
        cold_prefetch_admission_upper_conditional_per_request=(min(512, 64 * (rounds - 1), unique) if native else None),
        history_tiles_per_request=H // 128,
        hot_tiles_per_request=R // 128,
        cold_tiles_per_request=(H - R) // 128,
        qk_math_mac_batch=b * Q * H * heads * D,
        key_tensor_bytes_batch=b * H * D,
        score_capacity_bytes_batch=b * Q * H * 2,
        score_write_plus_final_read_bytes_batch=2 * b * Q * H * 2,
        resident_threshold_score_read_bytes_batch=b * Q * R * 2,
        classification_hot_metadata_logical_read_bytes_batch=b * Q * R * 8,
        conditional_no_prefetch_cold_payload_bytes_batch=b * unique * token_bytes,
        conditional_po_avoidable_reread_bytes_batch=b * (65536 + 8 * unique),
        tail_three_role_aiv_count=roles,
        tail_po_plus_consumer_proposed_aiv_count=2 * b + 4 * fetch_width,
        tail_one_owner_proposed_aiv_count=b + 4 * fetch_width,
        tail_three_role_budget_fits=roles <= aiv,
        tail_po_plus_consumer_budget_only_fits=2 * b + 4 * fetch_width <= aiv,
        tail_one_owner_budget_only_fits=b + 4 * fetch_width <= aiv,
        nonempty_static_fetch_workers_upper=4 * min(b, fetch_width),
        # These cannot be determined from nominal miss or static dimensions.
        actual_prefetch_submitted=None,
        actual_prefetch_useful=None,
        actual_repair_descriptors=None,
        actual_index_events=None,
        actual_safe_conflict_split=None,
        physical_dram_bytes=None,
        consumer_ready_ticks=None,
        predicted_speedup=None,
        winning_policy=None,
        evidence_status="ANALYTICAL_COUNTS_ONLY_NOT_EXECUTED",
    )
    for t in (1, 4, 8):
        tasks = b * Q * t
        row.update(
            {
                f"t{t}_local_tasks": tasks,
                f"t{t}_local_task_count_waves": ceildiv(tasks, aiv),
                f"t{t}_merge_tasks": b * Q if t > 1 else 0,
                f"t{t}_candidate_capacity_bytes_batch": b * Q * t * K * 6,
            }
        )
    # A proposed producer-local exact-K interface, distinct from the current
    # research entry which still reads a materialized full score tensor.
    for p in (2, 8, 16, 32):
        row[f"proposed_p{p}_candidate_write_read_bytes_batch"] = 2 * b * Q * p * K * 6
    return row


def build():
    cases = customer_cases()
    rows = [
        account(c, aic, aiv, g, size)
        for c, (aic, aiv), g, size in itertools.product(cases, ((32, 64), (28, 56)), (32, 64), (656, 2624))
    ]
    return dict(
        schema_version=1,
        scope="structural counts, not an execution or calibrated cost model",
        source_base="155fa9707d67fa95a028df485592240f0dd23138",
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCE_PATHS},
        assumptions=[
            "42 standard cases: full H131072, R16384, Q4, K2048, requested windows8, W2",
            "fixture Q0=Q2, Q1=Q3 cold sets disjoint; batch duplicates the same request seed",
            "MACs are mathematical; tensor/logical bytes are not measured DRAM transactions",
            "cold-only prefetch upper assumes no cold decoys/tail/extra misses and adequate room; hot-victim admission can occur and shares budgets; actual cold admission can be smaller",
            "conditional cold payload and PO rereads assume no prefetch and no extra invalidation; U is not runtime repair count",
            "proposed role counts and producer-local candidate bytes establish no correctness, UB fit or performance",
            "local task-count waves ignore per-task time, scheduling overhead and resource interference",
            "all unobserved runtime/cost quantities are null; no automatic winning policy",
        ],
        base_cases=cases,
        rows=rows,
    )


def write(output):
    output.mkdir(parents=True, exist_ok=True)
    result = build()
    (output / "adaptive-workloads.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (output / "adaptive-workloads.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["rows"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result["rows"])
    base = [r for r in result["rows"] if (r["aic_budget"], r["heads"], r["token_bytes"]) == (32, 32, 656)]
    with (output / "adaptive-standard42.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(base[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(base)
    print(
        json.dumps(
            dict(base_cases=len(base), analytical_variants=len(result["rows"]), device_runs=0, calibrated_latency=False)
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    write(parser.parse_args().output)
