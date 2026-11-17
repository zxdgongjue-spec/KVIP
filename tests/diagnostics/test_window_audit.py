# SPDX-License-Identifier: Apache-2.0
import csv
import hashlib
import json
from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pytest

from kvip.diagnostics.archive import TensorArchive
from kvip.diagnostics.incremental_topk import select, scan_windows
from kvip.diagnostics.recent_threshold import recent_positions
from kvip.diagnostics.scan_options import INCREMENTAL_INPLACE_VARIANTS
from kvip.diagnostics.system_profile import inspect_artifacts, summarize_timeline
from kvip.diagnostics.window_audit import audit_snapshot, profile_attribution, query_work, run


def profile_csv(root, rows):
    path = root / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv"
    path.parent.mkdir(parents=True)
    with path.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.writer(target)
        writer.writerow(["Name", "Start Time(us)", "Duration(us)", "Wait Time(us)", "Block Num"])
        for name, start, duration in rows:
            writer.writerow([name, str(Decimal("1794930353418401.437") + Decimal(str(start))) + "\t",
                             str(duration) + "\t", "0.022\t", "32\t"])
    return path


def test_csv_epoch_precision_and_overlap(tmp_path):
    profile_csv(tmp_path, [("v9_init_kernel_1", 0, 16.728), ("v9_li_kernel_0", 16.750, 295.931),
                          ("v9_streaming_resolve_update_5", 312.703, 303.609)])
    artifacts = inspect_artifacts(tmp_path, "none")
    assert artifacts["status"] == "exports_present_review_required"
    result = summarize_timeline(tmp_path, artifacts)
    assert result["source_format"] == "kernel_details.csv"
    assert result["device"]["span_us"] == pytest.approx(616.312, abs=1e-6)
    assert result["kernels"][1]["start_us"] == pytest.approx(16.750, abs=1e-6)
    assert result["included_in_headline_timing"] is False
    profile_csv(tmp_path / "other", [("v9_li_kernel", 0, 1)])
    assert summarize_timeline(tmp_path, inspect_artifacts(tmp_path, "none"))["status"] == "unavailable"


def test_csv_does_not_sum_overlapping_kernels(tmp_path):
    profile_csv(tmp_path, [("v9_li_kernel_0", 0, 10), ("v9_li_kernel_1", 1, 10)])
    result = summarize_timeline(tmp_path, inspect_artifacts(tmp_path, "none"))
    assert result["li"] == dict(count=2, sum_us=20, span_us=11)


@pytest.mark.parametrize("distribution", ["zero", "ties", "random", "ascending"])
def test_work_matches_full_history_topk_and_keeps_causal_seed_exclusions(distribution):
    rng = np.random.default_rng(19)
    scores = rng.integers(0, 65536 if distribution == "random" else 4, 4091, dtype=np.uint16)
    if distribution == "zero":
        scores[:] = 0
    if distribution == "ascending":
        scores[:] = np.arange(len(scores), dtype=np.uint16)
    seed = np.r_[np.arange(512), np.arange(4020, len(scores))]
    windows = np.array_split(np.arange(512, 4096)[::-1], 8)
    eligible = np.arange(4096) % 3 == 0
    rows, expected = query_work(scores, seed, windows, eligible, k=128)
    assert len(rows) == 8 and expected["expected_ids"].shape == (9, 128)
    assert sum(row["valid_new_positions"] for row in rows) + len(seed) == len(scores)
    _, _, threshold = select(scores, np.arange(len(scores)), k=128)
    assert expected["expected_thresholds"][-1] == threshold
    assert len(set(expected["expected_ids"][-1])) == 128
    assert min(scores[expected["expected_ids"][-1]]) == threshold


def snapshot(root):
    n, r = 32768, 16384
    manifest = dict(geometry=dict(batch=1, resident=r, heads=32, history=n))
    archive = TensorArchive(root, manifest)
    values = np.random.default_rng(7).integers(0, 100, (1, 4, n), dtype=np.uint16)
    archive.save("post.score_keys", values)
    archive.save("pre.query_ends", np.array([[n-3, n-2, n-1, n]], dtype=np.int32))
    archive.save("pre.threshold_bounds", np.array([[n-128, n]], dtype=np.int32))
    archive.save("pre.tail_bounds", np.array([[n-64, n]], dtype=np.int32))
    archive.save("pre.row_owners", np.array([1], dtype=np.int32))
    archive.save("pre.hot_valid", np.ones((2, r), dtype=np.int32))
    archive.save("pre.reserved_hot", np.full((1, r), -1, dtype=np.int32))
    (root / "manifest.json").write_text(json.dumps(manifest))
    return hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()


def test_exported_queries_reproduce_all_eight_windows(tmp_path):
    root = tmp_path / "snapshot"
    snapshot(root)
    summary = audit_snapshot(root, tmp_path / "audit", samples=3)
    assert summary["query_windows"] == 32 and len(summary["probe_samples"]) == 3
    assert sum(summary["branches"].values()) == 32
    assert summary["probe_queries"] == 4
    with np.load(tmp_path / "audit/probe_inputs.npz", allow_pickle=False) as bundle:
        for q in range(4):
            ends = bundle["query_ends"][0]
            windows = scan_windows(32768, 16384, ends, True)
            seed = recent_positions(16384, bundle["threshold_bounds"][0], int(ends[q]))
            _, expected = query_work(bundle["scores"][0, q, :ends[q]], seed,
                                     windows, bundle["eligible"][0])
            for name, values in expected.items():
                np.testing.assert_array_equal(values, bundle[name][0, q])


def diagnostic(root, group=7):
    cases = {}
    for name, li, streaming in (("reverse", 295.931, 303.609),
                                ("reverse-inc-register", 408.355, 224.279),
                                ("reverse-inc-inplace", 417.849, 221.634)):
        folder = root / name
        profile_csv(folder, [("v9_init_kernel_1", 0, 16), ("v9_li_kernel_0", 16.022, li),
                            ("v9_streaming_resolve_update_5", 16.044 + li, streaming)])
        cases[name] = dict(system_profiles=dict(passes=[dict(
            metric="none", directory=str(folder), graph_replays=1, artifacts=inspect_artifacts(folder, "none"))]))
    path = root / f"g{group}-diagnostic.json"
    path.write_text(json.dumps(dict(cases=cases)))
    return path


def test_measured_attribution_and_wrong_replay_count_rejected(tmp_path):
    path = diagnostic(tmp_path)
    stages = profile_attribution(path)
    assert stages["reverse-inc-register"]["stages"]["li_us"] - stages["reverse"]["stages"]["li_us"] == pytest.approx(112.424)
    data = json.loads(path.read_text())
    data["cases"]["reverse"]["system_profiles"]["passes"][0]["graph_replays"] = 2
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="single-graph"):
        profile_attribution(path)


def test_complete_offline_audit_uses_saved_results_and_real_inputs(tmp_path, capsys):
    snapshot_hash = snapshot(tmp_path / "inputs/g7-input")
    results = tmp_path / "results"
    results.mkdir()
    diagnostic(results)
    cases = {}
    for name in INCREMENTAL_INPLACE_VARIANTS:
        cases[name] = dict(
            production_source=dict(commit="a" * 40), snapshot_sha256=snapshot_hash,
            incremental_check=dict(status="exact_prefix_ranks_and_pending", prefix_lists_sha256="prefix"),
            selection_comparison=dict(ordered_output_sha256="ordered"),
            geometry=dict(rounds=9, token_bytes=656), timing=dict(mean_us=100., p50_us=99.),
            reads_samples=[dict(prefetch=2, unused_prefetch=1, unused_prefetch_bytes=656,
                                final=2, bytes=2624, useful_transfer_bytes=1968)])
    (results / "g7.json").write_text(json.dumps(dict(status="pass", execution_order=INCREMENTAL_INPLACE_VARIANTS, cases=cases)))
    run(SimpleNamespace(results=results, inputs=tmp_path / "inputs", output=tmp_path / "audit",
                        groups=[7], profile_groups=[7], expected_native="a" * 40))
    assert "AUDIT_PASS" in capsys.readouterr().out
    assert json.loads((tmp_path / "audit/summary.json").read_text())["status"] == "pass"
