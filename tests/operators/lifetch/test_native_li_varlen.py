#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A5 device test for per-request actual Q/K lengths consumed by native QLI.

This is the P1-A native baseline only. It does not exercise the custom v9_li
launcher, LIFetch resolver, KVAF BatchGet, or Index-cache update.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import acl
import torch
import torch_npu  # noqa: F401

from kvip.adapters.vllm.inputs import QueryInputs


BATCH = 4
QUERY_CAPACITY = 4
TOPK = 2048
HISTORY = 8192
BLOCK_TOKENS = 128


SCENARIOS = (
    {
        "name": "mixed_qk_round1",
        "query_counts": (4, 2, 1, 0),
        "query_starts": (0, 4, 6, 7),
        "positions": (8188, 8189, 8190, 8191, 6142, 6143, 2047),
        "expected_query_lengths": (4, 2, 1, 0),
        "expected_key_lengths": (8192, 6144, 2048, 0),
    },
    {
        "name": "mixed_qk_round2",
        "query_counts": (1, 0, 3, 2),
        "query_starts": (0, 1, 1, 4),
        "positions": (4095, 8189, 8190, 8191, 6142, 6143),
        "expected_query_lengths": (1, 0, 3, 2),
        "expected_key_lengths": (4096, 0, 8192, 6144),
    },
)


def make_native_inputs(device: torch.device, index_heads: int):
    """Build a deterministic BSND/PA_BSND fixture with an unambiguous TopK."""
    pages = HISTORY // BLOCK_TOKENS

    # Every Q/K component is positive and constant. Key scales are constant
    # within each 2048-token band and increase between bands. For any K length
    # in {2048, 4096, 6144, 8192}, the highest-scoring valid band contains
    # exactly TOPK tokens, so the expected TopK set has no boundary tie.
    query = torch.full(
        (BATCH, QUERY_CAPACITY, index_heads, 128),
        0.5,
        dtype=torch.float32,
        device=device,
    ).to(torch.float8_e4m3fn)
    key = torch.full(
        (pages, BLOCK_TOKENS, 1, 128),
        0.5,
        dtype=torch.float32,
        device=device,
    ).to(torch.float8_e4m3fn)
    weights = torch.ones(
        (BATCH, QUERY_CAPACITY, index_heads),
        dtype=torch.bfloat16,
        device=device,
    )
    query_scale = torch.ones(
        (BATCH, QUERY_CAPACITY, index_heads),
        dtype=torch.float32,
        device=device,
    )
    key_scale = torch.cat(
        [
            torch.full((TOPK,), float(band + 1), dtype=torch.float32, device=device)
            for band in range(HISTORY // TOPK)
        ]
    ).view(pages, BLOCK_TOKENS, 1)
    block_table = (
        torch.arange(pages, dtype=torch.int32, device=device)
        .view(1, pages)
        .expand(BATCH, pages)
        .contiguous()
    )

    return query, key, weights, query_scale, key_scale, block_table


def call_native_li(torch_npu, query, key, weights, query_scale, key_scale, q_lengths, k_lengths, block_table):
    result = torch_npu.npu_quant_lightning_indexer(
        query=query,
        key=key,
        weights=weights,
        query_dequant_scale=query_scale,
        key_dequant_scale=key_scale,
        actual_seq_lengths_query=q_lengths,
        actual_seq_lengths_key=k_lengths,
        block_table=block_table,
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="BSND",
        layout_key="PA_BSND",
        sparse_count=TOPK,
        sparse_mode=0,
    )
    selected = result[0] if isinstance(result, (tuple, list)) else result
    if not isinstance(selected, torch.Tensor):
        raise TypeError(f"native QLI returned unsupported output type: {type(selected)!r}")
    expected_numel = query.shape[0] * QUERY_CAPACITY * TOPK
    if selected.numel() != expected_numel:
        raise AssertionError(
            f"native QLI output size mismatch: got {tuple(selected.shape)}, "
            f"expected {expected_numel} elements"
        )
    return selected.reshape(query.shape[0], QUERY_CAPACITY, TOPK).to(torch.int32)


def update_actual_lengths(inputs, scenario, device):
    inputs.prepare(
        scenario["query_counts"],
        query_starts=scenario["query_starts"],
        query_owners=(0, 1, 2, 3),
    )
    positions = torch.tensor(scenario["positions"], dtype=torch.int64, device=device)
    inputs.bind_positions(positions)
    torch.npu.synchronize()

    q_lengths = tuple(int(x) for x in inputs.query_lengths.cpu().tolist())
    k_lengths = tuple(int(x) for x in inputs.key_lengths.cpu().tolist())
    if q_lengths != scenario["expected_query_lengths"]:
        raise AssertionError(
            f"{scenario['name']}: query lengths {q_lengths} != "
            f"{scenario['expected_query_lengths']}"
        )
    if k_lengths != scenario["expected_key_lengths"]:
        raise AssertionError(
            f"{scenario['name']}: key lengths {k_lengths} != "
            f"{scenario['expected_key_lengths']}"
        )
    return q_lengths, k_lengths


def verify_round(torch_npu, inputs, fixture, scenario, device):
    query, key, weights, query_scale, key_scale, block_table = fixture
    q_lengths, k_lengths = update_actual_lengths(inputs, scenario, device)

    batch_output = call_native_li(
        torch_npu,
        query,
        key,
        weights,
        query_scale,
        key_scale,
        inputs.query_lengths,
        inputs.key_lengths,
        block_table,
    )
    torch.npu.synchronize()

    # Verify each batched row against an isolated native invocation with the
    # exact same Q/K lengths. Compare sorted IDs to ignore TopK output ordering.
    rowwise_checks = 0
    for row in range(BATCH):
        reference = call_native_li(
            torch_npu,
            query[row : row + 1],
            key,
            weights[row : row + 1],
            query_scale[row : row + 1],
            key_scale,
            inputs.query_lengths[row : row + 1],
            inputs.key_lengths[row : row + 1],
            block_table[row : row + 1],
        )
        torch.npu.synchronize()
        for query_index in range(QUERY_CAPACITY):
            batched_ids = torch.sort(batch_output[row, query_index]).values
            isolated_ids = torch.sort(reference[0, query_index]).values
            if not torch.equal(batched_ids, isolated_ids):
                raise AssertionError(
                    f"{scenario['name']}: batched/isolated native QLI mismatch "
                    f"at row={row}, query={query_index}"
                )
        rowwise_checks += 1

    active_queries = 0
    padded_queries = 0
    for row, (q_len, k_len) in enumerate(zip(q_lengths, k_lengths)):
        if q_len == 0:
            if k_len != 0:
                raise AssertionError(
                    f"{scenario['name']}: inactive row {row} unexpectedly has K={k_len}"
                )
        else:
            if k_len < TOPK or k_len > HISTORY or k_len % TOPK:
                raise AssertionError(
                    f"{scenario['name']}: invalid active K length at row {row}: {k_len}"
                )
            expected = torch.arange(
                k_len - TOPK, k_len, dtype=torch.int32, device=device
            )
            for query_index in range(q_len):
                actual = torch.sort(batch_output[row, query_index]).values
                if not torch.equal(actual, expected):
                    raise AssertionError(
                        f"{scenario['name']}: TopK does not match actual K range "
                        f"at row={row}, query={query_index}, K={k_len}"
                    )
                active_queries += 1

        padding = batch_output[row, q_len:]
        if padding.numel():
            if not bool(torch.all(padding == -1).item()):
                raise AssertionError(
                    f"{scenario['name']}: padded Q output was not cleared to -1 at row {row}"
                )
            padded_queries += QUERY_CAPACITY - q_len

    return {
        "name": scenario["name"],
        "query_lengths": list(q_lengths),
        "key_lengths": list(k_lengths),
        "batch_vs_isolated_rows": rowwise_checks,
        "active_queries_checked": active_queries,
        "padded_queries_checked": padded_queries,
        "topk_actual_k_range": "PASS",
        "padding_clear_to_minus_one": "PASS",
        "result": "PASS",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--index-heads", type=int, choices=(32, 64), default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/kvip-native-li-varlen"),
    )
    args = parser.parse_args()

    soc = acl.get_soc_name()
    if "950" not in soc:
        raise RuntimeError(
            f"This test is scoped to Ascend A5/950; detected SoC={soc!r}"
        )

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    inputs = QueryInputs(
        torch,
        BATCH,
        device,
        pin_memory=True,
        history=HISTORY,
        index_heads=args.index_heads,
    )
    query_lengths = torch.zeros((BATCH,), dtype=torch.int32, device=device)
    key_lengths = torch.zeros((BATCH,), dtype=torch.int32, device=device)
    inputs.bind_actual_lengths(query_lengths, key_lengths)
    query_ptr = query_lengths.data_ptr()
    key_ptr = key_lengths.data_ptr()

    fixture = make_native_inputs(device, args.index_heads)
    summaries = []
    for scenario in SCENARIOS:
        summaries.append(verify_round(torch_npu, inputs, fixture, scenario, device))

    if query_lengths.data_ptr() != query_ptr or key_lengths.data_ptr() != key_ptr:
        raise AssertionError("actual Q/K length buffer addresses changed between forwards")

    torch.npu.synchronize()
    report = {
        "test": "native_li_variable_actual_qk_lengths",
        "device": str(device),
        "soc": soc,
        "batch": BATCH,
        "history": HISTORY,
        "index_heads": args.index_heads,
        "topk": TOPK,
        "layout_query": "BSND",
        "layout_key": "PA_BSND",
        "length_buffers_stable": True,
        "query_lengths_ptr": query_ptr,
        "key_lengths_ptr": key_ptr,
        "rounds": summaries,
        "scope_limit": (
            "Direct torch_npu.npu_quant_lightning_indexer native baseline only; "
            "does not validate KVIP custom v9_li launch, LIFetch resolver, KVAF BatchGet, "
            "or Index-cache update."
        ),
        "result": "PASS",
    }

    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "native_li_varlen_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    for item in summaries:
        print(
            "NATIVE_LI_VARLEN_PASS "
            f"round={item['name']} "
            f"Q={item['query_lengths']} K={item['key_lengths']} "
            f"active_queries={item['active_queries_checked']} "
            f"padded_queries={item['padded_queries_checked']} "
            f"rowwise={item['batch_vs_isolated_rows']}"
        )
    print(f"BUFFER_ADDRESSES_STABLE: PASS q={query_ptr} k={key_ptr}")
    print(f"NATIVE_LI_VARLEN: PASS soc={soc} report={report_path}")


if __name__ == "__main__":
    main()
