# SPDX-License-Identifier: Apache-2.0
"""Admission counters preserve the captured selection, causality and credit rules."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from kvip.diagnostics.admission_survey import PROFILES, admit, main, survey
from kvip.diagnostics.archive import load_snapshot
from kvip.diagnostics.score_replay import merge_traces
from kvip.diagnostics.token_replay import materialize_score_trace
from score_archive_fixture import score_archive


@pytest.fixture
def snapshot(tmp_path):
    source = score_archive(real=True, width=656)
    previous = copy.deepcopy(source)
    previous.array = lambda name: previous.arrays[name]
    previous.manifest["metadata"]["context"]["round"] -= 1
    previous.arrays["pre.tail_bounds"][:, 0] -= 128
    merged = merge_traces([source], numa_node=2, previous=[previous])
    materialized = materialize_score_trace(merged, tmp_path / "snapshot", numa_node=2)
    archive = load_snapshot(materialized.directory)
    yield archive
    archive.close()


def mutable(archive):
    arrays = {name: np.array(archive.array(name), copy=True) for name in archive.manifest["tensors"]}
    return SimpleNamespace(manifest=copy.deepcopy(archive.manifest), arrays=arrays, array=lambda name: arrays[name])


def test_real_archive_selection_is_immutable_and_counts_conserve(snapshot):
    before = snapshot.array("post.logical_output").copy()
    report = survey(snapshot, list(PROFILES))
    np.testing.assert_array_equal(before, snapshot.array("post.logical_output"))
    assert report["status"] == "pass"
    demand = report["demand"]["selected_read_demand"]
    for row in report["profiles"]:
        assert row["useful_prefetch"] + row["final_missing"] == demand
        assert row["cold_prefetch"] + row["candidate_hot_prefetch"] == row["useful_prefetch"] + row["unused_prefetch"]
        assert row["prefetch_bytes"] == 656 * (row["useful_prefetch"] + row["unused_prefetch"])
        assert row["prefetch_bytes"] <= 2 * row["total_per_request"] * 656
    assert report["demand"]["abc_hit"] > 0
    assert "Not actual producer" in report["ordering"]


def test_boost_is_strict_and_cannot_change_exact_topk(snapshot):
    archive = mutable(snapshot)
    scores = archive.arrays["post.score_keys"]
    # Preserve all ranking/selection relations but reduce the key separations.
    scores[scores == 0xBF00] = 0x8001
    scores[scores == 0xBF80] = 0x8002
    scores[scores == 0xC000] = 0x8003
    report = survey(archive, ["window480", "boost1", "boost2"])
    zero, one, two = report["profiles"]
    assert zero["eligible_candidates"] > one["eligible_candidates"] > two["eligible_candidates"]
    assert one["selected_suppressed_by_boost"] == 0
    assert two["selected_suppressed_by_boost"] > 0
    assert two["final_missing"] > zero["final_missing"]
    assert two["candidate_hot_prefetch"] == zero["candidate_hot_prefetch"]
    assert len(report["selection_sha256"]) == 64


def test_credit_carry_never_revisits_rejected_candidates():
    gate = np.ones(6, dtype=bool)
    late = np.array([2, 2, 2, 2, 2, 2])
    assert admit(gate, late, 6, 2, False, [1, 1, 1]).sum() == 2
    assert admit(gate, late, 6, 2, True, [1, 1, 1]).sum() == 6
    early = np.zeros(6, dtype=int)
    # The native scanner advances past quota rejects; final repair handles them.
    assert admit(gate, early, 6, 2, False, [1, 1, 1]).sum() == 2
    assert admit(gate, early, 6, 2, True, [1, 1, 1]).sum() == 2
    assert admit(gate, early, 6, 2, False, [0, 0, 0]).sum() == 0


def test_credit_carry_respects_native_per_window_480_descriptor_cap():
    gate = np.ones(1024, dtype=bool)
    late = np.full(1024, 2)
    assert admit(gate, late, 1024, 480, True, [1, 1, 1]).sum() == 480


def test_resident_compact_lru_scan_uses_lru_order_not_sorted_physical_slots():
    gate = np.ones(3, dtype=bool)
    fetched = admit(gate, np.zeros(3), 1, 1, False, [1], np.array([2, 0, 1]))
    np.testing.assert_array_equal(fetched, [False, True, False])


def test_recent_seed_scores_are_not_prefetch_ready_in_resident_window(snapshot):
    archive = mutable(snapshot)
    archive.manifest["geometry"]["rounds"] = 1
    report = survey(archive, ["window480"])
    assert report["profiles"][0]["cold_prefetch"] == 0


@pytest.mark.parametrize("field", ["pre.threshold_bounds", "post.score_keys"])
def test_missing_required_fields_fail_closed(snapshot, field):
    archive = mutable(snapshot)
    del archive.arrays[field]
    with pytest.raises(ValueError, match="Missing"):
        survey(archive, ["baseline"])


@pytest.mark.parametrize("fault", ["nan", "future", "wrong_topk", "recent", "unreadable"])
def test_invalid_snapshot_cannot_silently_create_a_model(snapshot, fault):
    archive = mutable(snapshot)
    if fault == "nan":
        archive.arrays["post.score_keys"][0, 0, 5000] = 0xFFC1
    elif fault == "future":
        archive.arrays["post.logical_output"][0, -1] = 8191
    elif fault == "wrong_topk":
        archive.arrays["post.score_keys"][0, 0, 5002] = 0xC100
    elif fault == "recent":
        archive.arrays["pre.threshold_bounds"][0, 1] -= 1
    else:
        archive.arrays["pre.addresses"][:, 5000] = 0
    with pytest.raises(ValueError):
        survey(archive, ["baseline"])


def test_phase_weights_require_exact_window_geometry(snapshot):
    with pytest.raises(ValueError, match="Phase weights"):
        survey(snapshot, ["baseline"], [1])
    with pytest.raises(ValueError, match="Phase weights"):
        survey(snapshot, ["baseline"], [-1] * snapshot.manifest["geometry"]["rounds"])


def test_cli_reads_materialized_schema_and_writes_only_cpu_counts(snapshot, tmp_path, capsys):
    output = tmp_path / "admission.json"
    main(["--snapshot", str(snapshot.directory), "--profiles", "baseline", "boost2", "--output", str(output)])
    report = json.loads(output.read_text())
    assert [row["name"] for row in report["profiles"]] == ["baseline", "boost2"]
    assert "ADMISSION_CPU_PASS" in capsys.readouterr().out
    assert "NPU time" in report["scope"]


def test_rank2048_matches_the_unboosted_gate_without_changing_quotas(snapshot):
    report = survey(snapshot, ["baseline"], gate_ranks=[2048, 1792, 1536])
    original = report["profiles"][0]
    ranks = report["gate_profiles"]
    assert [row["gate_rank"] for row in ranks] == [2048, 1792, 1536]
    for name, value in original.items():
        if name != "name":
            assert ranks[0][name] == value
    for row in ranks:
        assert row["safe_rank"] == 2048
        assert row["gate_profile"] == "baseline"
        assert (row["total_per_request"], row["window_per_request"], row["carry_unused_credits"]) == (512, 64, False)
        assert row["admitted"] == row["useful_prefetch"] + row["unused_prefetch"]
        assert row["useful_prefetch"] + row["final_estimate"] == report["demand"]["selected_read_demand"]
    assert report["phase_weights"] == [1.0] * snapshot.manifest["geometry"]["rounds"]
    assert ranks[0]["mean_raise_steps"] == 0.0


def test_stricter_rank_keeps_resident_protection_and_repairs_selected_cold_ties(snapshot):
    before = snapshot.array("post.logical_output").copy()
    report = survey(snapshot, [], gate_ranks=[2048, 1536, 1])
    zero, strict, top = report["gate_profiles"]
    # Fixture's 2048th seed score is below the final threshold. Rank 1536
    # reaches the final boundary: strict admission excludes tied cold tokens,
    # but they remain exact demand and must appear in final repair.
    assert zero["cold_prefetch"] > strict["cold_prefetch"]
    assert strict["selected_suppressed_by_gate"] > 0
    assert strict["final_estimate"] > zero["final_estimate"]
    assert strict["candidate_hot_prefetch"] == zero["candidate_hot_prefetch"]
    assert strict["threshold_over_final_queries"] == 0
    assert strict["mean_raise_steps"] > 0
    # Top-1 comes only from the early seed. It may exceed final Kth; do not
    # clamp against future/full-history information or silently pick a rank.
    assert top["threshold_over_final_queries"] == snapshot.manifest["geometry"]["batch"] * 4
    np.testing.assert_array_equal(before, snapshot.array("post.logical_output"))


@pytest.mark.parametrize("ranks", [[0], [2049], [-1], [True], [1536.5], [2048, 2048]])
def test_invalid_gate_rank_fails_closed(snapshot, ranks):
    with pytest.raises(ValueError, match="Gate ranks"):
        survey(snapshot, [], gate_ranks=ranks)


def test_rank_capacity_profile_matches_the_original_768_control(snapshot):
    report = survey(snapshot, ["total768"], gate_ranks=[2048, 1536], gate_profile="total768")
    control, = report["profiles"]
    same, strict = report["gate_profiles"]
    for name, value in control.items():
        if name != "name":
            assert same[name] == value
    assert strict["gate_profile"] == "total768"
    assert (strict["total_per_request"], strict["window_per_request"], strict["carry_unused_credits"]) == (768, 128, True)
    with pytest.raises(ValueError, match="Gate profile"):
        survey(snapshot, [], gate_ranks=[2048], gate_profile="boost1")


def test_gate_rank_cli_only_prints_requested_counterfactuals(snapshot, tmp_path, capsys):
    output = tmp_path / "rank-gates.json"
    main(["--snapshot", str(snapshot.directory), "--gate-ranks", "2048", "1792", "1536",
          "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["profiles"] == []
    assert len(report["gate_profiles"]) == 3
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 6
    assert lines[2].startswith("THRESHOLD recent_equal=")
    assert all("RANK_GATE" in row and "final_estimate=" in row and
               "threshold_over_final_queries=" in row and "mean_raise_steps=" in row for row in lines[3:])
    assert "npu_timing=false" in lines[0]
