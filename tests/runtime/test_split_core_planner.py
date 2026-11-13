"""Check workload accounting against independent conservation/geometry facts."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("adaptive_plan", ROOT / "experiments/split_core/adaptive_plan.py")
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)


def case(batch, miss):
    return next(c for c in plan.customer_cases() if (c["batch"], c["miss_percent"]) == (batch, miss))


def test_exact_customer_matrix_and_axis_coverage():
    result = plan.build()
    assert len(result["base_cases"]) == 42
    assert len(result["rows"]) == 336
    assert len({(r["case"], r["aic_budget"], r["heads"], r["token_bytes"]) for r in result["rows"]}) == 336
    assert [case(1, m)["miss_per_query"] for m in (3, 5, 10, 15, 20, 50)] == [61, 102, 205, 307, 410, 1024]


def test_payload_head_and_batch_scaling_are_independent():
    base = plan.account(case(1, 20))
    payload = plan.account(case(1, 20), token_bytes=2624)
    heads = plan.account(case(1, 20), heads=64)
    batch = plan.account(case(8, 20))
    assert base["fixture_cold_union_per_request"] == 820
    assert base["conditional_no_prefetch_cold_payload_bytes_batch"] == 537920
    assert payload["conditional_no_prefetch_cold_payload_bytes_batch"] == 4 * 537920
    assert payload["qk_math_mac_batch"] == base["qk_math_mac_batch"] == 2**31
    assert heads["qk_math_mac_batch"] == 2**32
    assert batch["qk_math_mac_batch"] == 8 * base["qk_math_mac_batch"]
    assert base["score_capacity_bytes_batch"] == 1024**2
    assert base["classification_hot_metadata_logical_read_bytes_batch"] == 512 * 1024


@pytest.mark.parametrize("b,cap", [(1, 64), (2, 192), (4, 448), (8, 448), (16, 448)])
def test_requested_windows_differ_from_actual_rounds(b, cap):
    row = plan.account(case(b, 50))
    assert row["cold_prefetch_admission_upper_conditional_per_request"] == cap
    assert row["fixture_cold_union_per_request"] == 2048
    assert row["actual_repair_descriptors"] is None


def test_small_miss_caps_admission_and_payload_is_not_per_query_sum():
    row = plan.account(case(4, 3))
    assert row["cold_prefetch_admission_upper_conditional_per_request"] == 122
    assert row["conditional_no_prefetch_cold_payload_bytes_batch"] == 4 * 122 * 656


def test_current_limits_and_proposed_budget_are_distinct():
    assert plan.account(case(16, 50), 28, 56)["native_full_pipeline_supported"]
    assert plan.account(case(16, 50), 28, 56)["shape_history_shards"] == 1
    assert plan.account(case(16, 50), 32, 64)["shape_history_shards"] == 2
    large = plan.account(case(24, 10), 28, 56)
    assert large["tail_po_plus_consumer_proposed_aiv_count"] == 56
    assert large["tail_po_plus_consumer_budget_only_fits"]
    assert not large["native_full_pipeline_supported"]
    biggest = plan.account(case(32, 10), 28, 56)
    assert biggest["shape_actual_rounds"] is None
    assert biggest["cold_prefetch_admission_upper_conditional_per_request"] is None
    assert "PrepareShape" in biggest["unsupported_reasons"]
    assert biggest["tail_one_owner_proposed_aiv_count"] == 40


def test_more_shards_can_exceed_full_score_interface_bytes():
    row = plan.account(case(1, 10))
    assert row["proposed_p2_candidate_write_read_bytes_batch"] == 192 * 1024
    assert row["proposed_p32_candidate_write_read_bytes_batch"] == 3 * 1024**2
    assert row["score_write_plus_final_read_bytes_batch"] == 2 * 1024**2
    assert row["t8_local_tasks"] == 32
    assert row["t8_merge_tasks"] == 4
    assert row["t8_candidate_capacity_bytes_batch"] == 384 * 1024


def test_generated_delivery_is_reproducible_and_has_no_fabricated_timings(tmp_path):
    plan.write(tmp_path)
    root = ROOT / "docs/research/split_core/results"
    for name in ("adaptive-workloads.json", "adaptive-workloads.csv", "adaptive-standard42.csv"):
        assert (tmp_path / name).read_bytes() == (root / name).read_bytes()
    for row in plan.build()["rows"]:
        for unknown in (
            "actual_prefetch_submitted",
            "actual_prefetch_useful",
            "actual_index_events",
            "actual_repair_descriptors",
            "physical_dram_bytes",
            "consumer_ready_ticks",
            "predicted_speedup",
            "winning_policy",
        ):
            assert row[unknown] is None


def test_out_of_scope_geometry_is_rejected():
    malformed = {**case(1, 3), "history": 65536}
    with pytest.raises(ValueError, match="128K/Q4"):
        plan.account(malformed)
