"""CPU-only source-model checks; no NPU or KVAF required."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
from varlen_partition_model import (  # noqa: E402
    SCENARIOS, current_qli_partition, optimistic_request_s2_split,
)


def test_full_b4_uses_two_q2_groups_per_request():
    q, k = SCENARIOS["balanced_128k"]
    data = current_qli_partition(q, k, 32, "auto")
    assert data["nominal_cores"] == data["active_cores"] == 8
    assert data["idle_nominal_cores"] == 0
    assert [task["query_count"] for task in data["tasks"]] == [2] * 8
    assert [task["request"] for task in data["tasks"]] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert data["max_serial_s2_tiles_per_core"] == 1024
    assert data["max_query_tile_units_per_core"] == 2048


def test_mixed_q_count_releases_half_nominal_cores():
    q, k = SCENARIOS["mixed_qk"]
    data = current_qli_partition(q, k, 32, "auto")
    assert data["nominal_cores"] == 8
    assert data["active_cores"] == 4
    assert data["idle_nominal_cores"] == 4
    assert [(t["request"], t["query_begin"], t["query_count"]) for t in data["tasks"]] == [
        (0, 0, 2), (0, 2, 2), (1, 0, 2), (2, 0, 1)
    ]


def test_q4_is_one_request_per_core_and_not_s2_parallel():
    q, k = SCENARIOS["mixed_32_64_96_128k"]
    data = current_qli_partition(q, k, 32, "q4")
    assert data["active_cores"] == data["nominal_cores"] == 4
    assert data["max_serial_s2_tiles_per_core"] == 1024
    assert data["max_query_tile_units_per_core"] == 4096


def test_hypothetical_balanced_has_no_gain_at_same_eight_aic():
    q, k = SCENARIOS["balanced_128k"]
    result = optimistic_request_s2_split(q, k, 8)
    assert result["quota"] == [2, 2, 2, 2]
    assert result["ideal_max_query_tile_units"] == 2048
    assert result["global_lower_bound"] == 2048


def test_hypothetical_mixed_length_reassignment():
    q, k = SCENARIOS["mixed_32_64_96_128k"]
    result = optimistic_request_s2_split(q, k, 8)
    assert result["quota"] == [1, 2, 2, 3]
    assert result["ideal_max_query_tile_units"] == 1536
    assert result["global_lower_bound"] == 1280


def test_hypothetical_skew_has_larger_opportunity():
    q, k = SCENARIOS["extreme_8_8_8_128k"]
    result = optimistic_request_s2_split(q, k, 8)
    assert result["quota"] == [1, 1, 1, 5]
    assert result["ideal_max_query_tile_units"] == 820
    assert result["global_lower_bound"] == 608


def test_hypothetical_variable_q_reassignment():
    q, k = SCENARIOS["mixed_qk"]
    result = optimistic_request_s2_split(q, k, 8)
    assert result["quota"] == [5, 2, 1, 0]
    assert result["ideal_max_query_tile_units"] == 820
    assert result["global_lower_bound"] == 736


@pytest.mark.parametrize("q,k,cores", [
    ((4, 2), (131072,), 8),
    ((5,), (131072,), 8),
    ((4,), (131071,), 8),
    ((4, 4, 4, 4), (8192,) * 4, 3),
    ((4,), (0,), 8),
])
def test_reject_bad_geometry(q, k, cores):
    with pytest.raises(ValueError):
        current_qli_partition(q, k, cores)
