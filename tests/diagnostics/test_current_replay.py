# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import pytest

from kvip.diagnostics.compare import summarize
from kvip.diagnostics.replay_runtime import create_runtime, runtime_worker_config
from kvip.diagnostics.selection import verify_fused_scores
from kvip.diagnostics.threshold_check import reference_thresholds
from kvip.diagnostics import threshold_check


def test_recent_threshold_excludes_future_deduplicates_prefix_and_covers_retired_page():
    scores = np.full((1, 4, 8192), 0x8000, np.uint16)
    scores[:, :, :2047] = 0xBF80
    scores[:, :, 5000:5100] = 0xC000
    scores[:, :, 6000:] = 0xFF00
    ends = np.array([[5097, 5098, 5099, 5100]], np.int32)
    hot = reference_thresholds(scores, ends, 4096)
    recent = reference_thresholds(scores, ends, 4096, np.array([[4960, 6200]], np.int32))
    assert np.all(hot == 0x8000) and np.all(recent == 0xBF80)
    overlap = reference_thresholds(scores, ends, 4096, np.array([[2000, 5100]], np.int32))
    np.testing.assert_array_equal(recent, overlap)


def test_rank_threshold_is_distinct_from_safe_bound_and_respects_causal_seed(monkeypatch):
    scores = np.tile(np.arange(8192, dtype=np.uint16) + 0x8000, (1, 4, 1))
    ends = np.array([[5097, 5098, 5099, 5100]], np.int32)
    bounds = np.array([[4960, 6200]], np.int32)
    safe = reference_thresholds(scores, ends, 4096, bounds)
    gate = reference_thresholds(scores, ends, 4096, bounds, rank=1792)
    np.testing.assert_array_equal(gate.astype(np.int32) - safe, 256)
    # Future scores are much larger but must not enter either threshold.
    original = gate.copy()
    scores[:, :, 6000:] = 0xFFFF
    np.testing.assert_array_equal(reference_thresholds(scores, ends, 4096, bounds, rank=1792), original)
    row_bytes = 4 * 2048 * 6 + 1024
    raw = np.zeros((1, row_bytes), dtype=np.uint8)
    banks = raw[:, 4 * 2048 * 6:].view(np.uint16).reshape(1, 4, 128)
    banks[:, :, 0], banks[:, :, 1] = safe, gate
    arguments = [None] * 14
    arguments[13] = raw
    bundle = SimpleNamespace(
        rotating=SimpleNamespace(arguments=arguments),
        tuning=dict(recent_threshold=True, prefetch_rank=1792),
    )
    arrays = {"post.score_keys": scores, "pre.query_ends": ends, "pre.threshold_bounds": bounds}
    archive = SimpleNamespace(manifest={"geometry": {"batch": 1, "resident": 4096}}, array=arrays.__getitem__)
    monkeypatch.setattr(threshold_check, "cpu_array", lambda torch, tensor: tensor)
    report = threshold_check.verify_initial_threshold(None, bundle, archive)
    assert report["status"] == "bit_exact"
    assert report["prefetch_gate"]["rank"] == 1792
    assert report["prefetch_gate"]["mean_raise_key_steps"] == 256
    banks[0, 0, 1] += 1
    with pytest.raises(AssertionError, match="speculative gate differs"):
        threshold_check.verify_initial_threshold(None, bundle, archive)
    banks[:, :, 1] = gate
    banks[0, 0, 0] += 1
    with pytest.raises(AssertionError, match="Early threshold differs"):
        threshold_check.verify_initial_threshold(None, bundle, archive)


@pytest.mark.parametrize("rank", [True, 0, 2049, 1.5])
def test_reference_rejects_invalid_speculative_ranks(rank):
    with pytest.raises(ValueError, match="rank"):
        reference_thresholds(np.zeros((1, 4, 2048), np.uint16), np.full((1, 4), 2048), 2048, rank=rank)


def _archive():
    b, h = 1, 4096
    arrays = {
        "post.score_keys": np.full((b, 4, h), 0xBF80, np.uint16),
        "pre.query_ends": np.full((b, 4), 3000, np.int32),
        "pre.metadata": np.zeros((5, b * 4), np.int64),
        "pre.logical_ids": np.tile(np.arange(h, dtype=np.int64), (2, 1)),
        "post.logical_output": np.tile(np.arange(2048, dtype=np.int32), (4, 1)),
    }
    arrays["pre.metadata"][2] = np.arange(4)
    arrays["pre.metadata"][4].view(np.int32)[0] = 1
    return SimpleNamespace(manifest={"geometry": {"batch": b, "history": h}}, array=arrays.__getitem__), arrays


def test_fused_selection_accepts_only_proven_full_score_boundary_ties():
    archive, arrays = _archive()
    choice = np.tile(np.arange(952, 3000, dtype=np.int32), (1, 4, 1))
    logical, report = verify_fused_scores(archive, choice)
    assert report["status"] == "verified_bf16_boundary_ties"
    assert logical.shape == (4, 2048)
    arrays["post.score_keys"][0, 0, 0] = 0xC000  # Strictly better omitted score.
    with pytest.raises(AssertionError, match="exact captured-score"):
        verify_fused_scores(archive, choice)
    arrays["post.score_keys"][0, 0, 0] = 0xBF80
    choice[0, 0, 0] = 3000
    with pytest.raises(AssertionError, match="causality"):
        verify_fused_scores(archive, choice)


@pytest.mark.parametrize("transport", ["urma", "ubmem"])
def test_latest_runtime_uses_current_objects_and_constructor(tmp_path, transport):
    (tmp_path / "bin").mkdir()
    for kind in ("get", "put"):
        (tmp_path / "bin" / f"kvaf_execute_batch_{kind}_kernels.o").touch()
    args = SimpleNamespace(transport=transport, mode="serial", serial_workers=32, device=0, build_dir=tmp_path)
    archive = SimpleNamespace(
        manifest={"geometry": {"token_bytes": 656}, "ddr": {"token_bytes": 656, "tokens_per_block": 128}}
    )
    calls = []
    kvaf = SimpleNamespace(NpuRuntime=lambda *pos, **kw: calls.append((pos, kw)))
    create_runtime(args, archive, kvaf, None, None, None, None, 32)
    pos, kw = calls[0]
    assert pos[0].endswith("kvaf_execute_batch_get_kernels.o")
    assert kw["npu_core_count"] == 32
    assert not {"ubmem_page_mode", "ubmem_numa_node"} & set(kw)
    if transport == "urma":
        assert kw["batch_get_max_inflight_total"] == 2048
        assert runtime_worker_config(args, archive, 32)["batch_get_post_size"] == 128
    else:
        assert kw["transport"] == "ubmem"


def test_summary_rejects_different_input_or_changed_fused_selection():
    def case():
        return dict(
            status="pass",
            snapshot_sha256="same",
            geometry={},
            scope="full",
            soc="A5",
            device=0,
            reads_samples=[
                dict(
                    cold_miss_rate=0.1,
                    cold_demand=2,
                    selected_unique=20,
                    prefetch=2,
                    final=1,
                    unused_prefetch=1,
                    unused_prefetch_bytes=656,
                    bytes=1968,
                    useful_transfer_bytes=1312,
                )
            ],
            timing={"mean_us": 10, "p50_us": 10},
            runtime_workers=32,
            selection_comparison={"selected_set_sha256": "ids", "ordered_output_sha256": "ordered"},
            timing_boundary="complete graph",
        )

    reports = {
        f"{t}-{mode}": case() for t in ("urma", "ubmem") for mode in ("serial", "fused-resident", "fused-recent")
    }
    assert summarize(reports)["effects"]["urma"]["recent_vs_resident_reduction_pct"] == 0
    reports["ubmem-fused-recent"]["selection_comparison"] = {
        "selected_set_sha256": "different",
        "ordered_output_sha256": "ordered",
    }
    with pytest.raises(ValueError, match="selection"):
        summarize(reports)
    reports["ubmem-fused-recent"]["selection_comparison"] = {
        "selected_set_sha256": "ids",
        "ordered_output_sha256": "reordered",
    }
    with pytest.raises(ValueError, match="selection"):
        summarize(reports)
