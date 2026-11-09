# SPDX-License-Identifier: Apache-2.0
"""Reject mismatched real-snapshot comparisons and invalid timing evidence."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from kvip.diagnostics.compare import CASES, command, summarize
from kvip.diagnostics.replay_measure import fused_reads, statistics


def results():
    values = {}
    for transport, mode in CASES:
        latency = 200.0 if mode == "serial" else 100.0
        values[f"{transport}-{mode}"] = dict(
            transport=transport,
            mode=mode,
            status="pass",
            snapshot_sha256="capture_hash",
            geometry=dict(batch=16, token_bytes=2624),
            scope="one_group",
            soc="soc",
            device=0,
            kvip_commit="one_revision",
            runtime_workers=2,
            timing=dict(mean_us=latency, p50_us=latency),
            reads_samples=[dict(bytes=26240)],
        )
    return values


def test_comparison_uses_fixed_archive_and_reports_fusion_reduction():
    value = summarize(results())
    assert value["fusion_vs_serial"]["ubmem"] == dict(serial_over_fused=2.0, latency_reduction_percent=50.0)
    assert value["table"]["urma-fused"]["mean_read_bytes"] == 26240


@pytest.mark.parametrize(
    "key,value",
    [
        ("snapshot_sha256", "other"),
        ("geometry", dict(batch=1, token_bytes=656)),
        ("scope", "fetch_only"),
        ("device", 1),
        ("status", "failed"),
        ("kvip_commit", "other"),
    ],
)
def test_comparison_rejects_mixed_evidence(key, value):
    entries = results()
    entries["urma-fused"][key] = value
    with pytest.raises(ValueError):
        summarize(entries)


def test_command_never_loads_two_transports_in_one_process(tmp_path):
    args = SimpleNamespace(
        snapshot=Path("/snapshot"),
        ubmem_build_dir=Path("/ubmem/build"),
        urma_build_dir=Path("/urma/build"),
        kvaf_server="127.0.0.1:28999",
        device=0,
        soc="soc",
        iterations=10,
        warmup=2,
        stage_iterations=3,
        result_dir=tmp_path,
        ubmem_serial_workers=32,
        urma_serial_workers=2,
    )
    for transport, mode in CASES:
        argv, output = command(args, transport, mode)
        assert argv[argv.index("--build-dir") + 1] == f"/{transport}/build"
        assert ("--kvaf-server" in argv) == (transport == "urma")
        assert ("--serial-workers" in argv) == (mode == "serial")
        assert output.name == f"{transport}-{mode}.json"


@pytest.mark.parametrize("samples", [[], [0], [-1], [float("nan")], [float("inf")], [[1]]])
def test_invalid_device_timing_is_never_a_performance_result(samples):
    with pytest.raises(ValueError):
        statistics(samples)


def test_urma_completion_and_speculative_read_accounting():
    import torch

    rows = torch.zeros((1, 256), dtype=torch.int32)
    rows[0, 9], rows[0, 247] = 2, 1
    completion = torch.zeros((2, 1216), dtype=torch.uint8)
    completion.view(torch.int64)[:, 0] = 0x4B5641464247444F
    mapping = torch.tensor([[4, -1, 5, 4 + 8192]], dtype=torch.int32)
    topk = torch.tensor([[[0, 3]]], dtype=torch.int32)
    rotating = SimpleNamespace(
        rows=rows,
        row_status=torch.zeros((1, 32), dtype=torch.int32),
        final_posts=torch.zeros((1, 4, 32), dtype=torch.int32),
        prefetch_status=completion,
        final_status=completion.clone(),
        mapping=mapping,
        arguments=[None] * 9 + [topk],
        hot_valid=torch.ones((1, 4), dtype=torch.int32),
    )
    update = SimpleNamespace(status=torch.zeros(1, dtype=torch.int32))
    read = fused_reads(torch, rotating, update, "urma", 2624)
    assert read == dict(prefetch=2, final=1, bytes=7872, unused_prefetch=1, unused_prefetch_bytes=2624)
    bad = copy.copy(rotating)
    bad.final_status = completion.clone()
    bad.final_status.view(torch.int64)[0, 2] = 1
    with pytest.raises(AssertionError, match="URMA completion"):
        fused_reads(torch, bad, update, "urma", 2624)
