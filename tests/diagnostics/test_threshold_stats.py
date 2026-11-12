# SPDX-License-Identifier: Apache-2.0
"""CPU threshold policies preserve causal selection and request-local accounting."""

import copy

import numpy as np
import pytest

from kvip.diagnostics.threshold_stats import POLICIES, _rolling_mask, analyze_trace, seed_state
from kvip.diagnostics.recent_threshold import range_state, threshold_bounds, recent_positions
from kvip.diagnostics.token_trace import trace_summary
from test_score_replay import predecessor, score_trace
from test_token_replay import token_trace as _source_fixture


ZERO, HALF, ONE, TWO = 0x8000, 0xBF00, 0xBF80, 0xC000


@pytest.fixture(name="token_trace", params=[656, 2624])
def _token_fixture(request):
    return _source_fixture.__wrapped__(request)


def _trace(base):
    trace = score_trace(base, real=False)
    layers = trace.manifest["geometry"]["token_bytes"] // 656
    trace.manifest["metadata"]["layer_names"] = [f"model.layers.{index}" for index in range(layers)]
    return trace


def _refresh(trace):
    """Keep hand-authored scores and the trace's recorded threshold/summary consistent."""
    arrays, g = trace.arrays, trace.manifest["geometry"]
    scores, records = arrays["post.score_keys"], arrays["post.threshold_records"]
    records.fill(0)
    for row in range(g["batch"]):
        for query in range(4):
            end = int(arrays["pre.query_ends"][row, query])
            initial_end = min(g["resident"], end)
            initial = int(np.partition(scores[row, query, :initial_end], -2048)[-2048])
            final = int(np.partition(scores[row, query, :end], -2048)[-2048])
            records[row, query, 0] = [initial_end, initial_end, initial, 1]
            records[row, query, -1] = [end, g["history"], final, 4]
    trace.manifest["summary"] = trace_summary(trace)
    return trace


def _expanded(base, *, history=8192, resident=4096, selected=None):
    """A valid larger history with identity Index positions and two distinct owners."""
    base = _trace(base)
    b = base.manifest["geometry"]["batch"]
    if selected is None:
        selected = np.arange(2048, dtype=np.int32)
    assert len(selected) == 2048
    begin = history - 128
    ends = np.tile(np.arange(begin + 1, begin + 5, dtype=np.int32), (b, 1))
    ids = np.tile(np.arange(history, dtype=np.int32), (b, 1))
    base.manifest["geometry"].update(history=history, resident=resident, output_stride=resident + 384)
    base.manifest["metadata"]["context"].update(min_context=begin, max_context=history)
    base.arrays.update(
        {
            "pre.logical_ids": ids,
            "pre.hot_valid": np.ones((b, resident), dtype=np.uint8),
            "pre.address_valid": (ids < begin).astype(np.uint8),
            "pre.query_ends": ends,
            "pre.tail_bounds": np.tile([begin, begin + 4], (b, 1)).astype(np.int32),
            "pre.lru_slots": np.tile(np.arange(2048, dtype=np.int32), (b, 1)),
            "pre.ddr_page_slots": np.stack([np.arange(history // 128, dtype=np.int32) + row * 512 for row in range(b)]),
            "post.logical_output": np.tile(np.asarray(selected, dtype=np.int32), (b * 4, 1)),
        }
    )
    scores = np.full((b, 4, history), ZERO, dtype=np.uint16)
    for row in range(b):
        for q in range(4):
            scores[row, q, selected] = ONE
            scores[row, q, ends[row, q] :] = 0xC100
    base.arrays.update(
        {
            "post.score_keys": scores,
            "post.threshold_records": np.zeros((b, 4, (history + 16383) // 16384 + 2, 4), dtype=np.int32),
            "post.prefetch_records": np.empty((0, 3), dtype=np.int32),
            "post.final_records": np.empty((0, 3), dtype=np.int32),
        }
    )
    return _refresh(base)


def _report(trace, previous=None):
    if previous is None:
        previous = predecessor(trace)
    return analyze_trace(trace, seed_state(previous))


def test_recent_threshold_includes_retired_abc_and_current_query_without_future(token_trace):
    selected = np.array([*range(2046), 7936, 8064], dtype=np.int32)
    trace = _expanded(token_trace, history=8192, resident=4096, selected=selected)
    trace.arrays["pre.tail_bounds"][:] = [8064, 8068]
    trace.arrays["pre.query_ends"][:] = [8065, 8066, 8067, 8068]
    previous = predecessor(trace)
    previous.arrays["pre.tail_bounds"][:] = [7936, 8064]
    # A lower-scored readable extra lies outside the exact final selection.
    trace.arrays["post.score_keys"][:, :, selected] = TWO
    trace.arrays["post.score_keys"][:, :, 5000] = ONE
    trace.arrays["post.score_keys"][:, :, 8100:] = 0xC100
    requests, queries = _report(_refresh(trace), previous)
    for item in queries:
        assert item["initial_key"] == ZERO
        assert item["recent_key"] == item["final_key"] == TWO
        assert item["recent_begin"] == 7936
        assert item["recent_end"] == item["end"]
        assert item["retired_abc_positions"] == item["retired_abc_included"] == 128
        assert item["previous_topk_positions"] == item["previous_topk_covered_by_recent"] == 2048
    assert all(row["hot_extra_candidates"] == 1 and row["recent_extra_candidates"] == 0 for row in requests)
    assert all(row["recent_extra_bytes"] == 0 and row["recent_equal_queries"] == 4 for row in requests)


def test_recent_bounds_follow_request_identity_and_reject_permuted_recent(token_trace):
    trace = _expanded(token_trace)
    previous = predecessor(trace)
    previous.arrays["pre.tail_bounds"][0, 0] = 7808
    previous.arrays["pre.tail_bounds"][1, 0] = 7936
    expected = threshold_bounds(trace, range_state(previous))
    actual = threshold_bounds(trace, range_state(_swap_previous_rows(previous)))
    np.testing.assert_array_equal(actual, expected)
    assert actual[:, 0].tolist() == [7808, 7936]
    trace.arrays["pre.logical_ids"][0, [7900, 7901]] = [7901, 7900]
    with pytest.raises(ValueError, match="identity physical"):
        threshold_bounds(trace, range_state(previous))


def test_recent_prefix_overlap_is_deduplicated_and_negative_bf16_order_is_respected(token_trace):
    np.testing.assert_array_equal(recent_positions(4096, [4000, 4100], 4098), np.arange(4098))
    trace = _expanded(token_trace, selected=[*range(2046), 7936, 8064])
    previous = predecessor(trace)
    previous.arrays["pre.tail_bounds"][:, 0] = 7936
    scores = trace.arrays["post.score_keys"]
    scores.fill(0x3F7F)  # sortable -4
    scores[:, :, [*range(2046), 7936, 8064]] = 0x407F  # sortable -1
    scores[:, :, 6000] = 0x3FFF  # sortable -2, false under old resident -4
    requests, queries = _report(_refresh(trace), previous)
    assert all(row["recent_key"] == row["final_key"] == 0x407F for row in queries)
    assert all(row["initial_value"] == -4 and row["recent_value"] == -1 for row in queries)
    assert all(row["hot_extra_candidates"] == 1 and row["recent_extra_candidates"] == 0 for row in requests)


def test_strict_threshold_ties_use_q4_selected_union_without_cross_request_dedup(token_trace):
    trace = _trace(token_trace)
    # Two unselected final-boundary ties and one lower-scoring token are shared
    # by all four queries. Selected boundary ties must never become "extra".
    trace.arrays["post.score_keys"][:, :, [3600, 3601]] = ONE
    trace.arrays["post.score_keys"][:, :, 3602] = HALF
    requests, queries = _report(_refresh(trace))

    assert len(requests) == 2 and len(queries) == 8
    for row in requests:
        assert row["selected_unique"] == 2051
        assert row["cold_demand"] == 4
        assert row["hot_candidates"] == 2050
        assert row["hot_selected_candidates"] == 2047
        assert row["hot_extra_candidates"] == 3
        assert row["hot_extra_ties"] == 2
        assert row["hot_extra_below"] == 1
        assert row["hot_cold_extra"] == 3
        assert row["hot_candidate_hot_extra"] == 0
        assert row["hot_extra_bytes"] == 3 * row["token_bytes"]
        assert row["final_candidates"] == row["final_extra_candidates"] == 0
        assert row["seed_hot_candidates"] == 0
    assert sum(row["hot_extra_candidates"] for row in requests) == 6
    assert all(query["initial_key"] == ZERO and query["final_key"] == ONE for query in queries)


def test_observed_transfers_keep_duplicates_and_are_not_policy_predictions(token_trace):
    trace = _trace(token_trace)
    trace.arrays["post.prefetch_records"] = np.array(
        [[0, 3500, 3500], [0, 3600, 3600], [0, 3600, 3600]], dtype=np.int32
    )
    trace.arrays["post.score_keys"][:, :, 3600] = ONE
    requests, _ = _report(_refresh(trace))
    first, second = requests

    assert first["actual_prefetch"] == 3
    assert first["actual_unused_prefetch"] == 2
    assert first["actual_unused_final_ties"] == 2
    assert first["actual_unused_below_final"] == 0
    assert first["actual_unused_bytes"] == 2 * first["token_bytes"]
    assert first["hot_extra_candidates"] == 1
    assert second["actual_prefetch"] == second["actual_unused_bytes"] == 0
    assert second["hot_extra_candidates"] == 1


def test_candidate_domain_excludes_future_abc_unpublished_and_protected_hot(token_trace):
    selected = [*range(2045), 5000, 5001, 5002]
    trace = _expanded(token_trace, selected=selected)
    a = trace.arrays
    for row in range(2):
        for q in range(4):
            a["post.score_keys"][row, q, selected] = TWO
        # 2045 is an LRU victim, 2048 is protected hot, 5500 is readable cold,
        # 6000 is unpublished, and the final four history tokens are ABC.
        a["post.score_keys"][row, :, [2045, 2048, 5500, 6000]] = ONE
        a["pre.address_valid"][row, 6000] = 0
        begin, stop = a["pre.tail_bounds"][row]
        a["post.score_keys"][row, :, begin:stop] = ONE
        a["pre.address_valid"][row, begin:stop] = 1
        # Give future positions valid DDR addresses as well; causality alone
        # must still suppress the already high future scores in this fixture.
        a["pre.address_valid"][row, stop:] = 1
    requests, _ = _report(_refresh(trace))

    for row in requests:
        assert row["hot_candidates"] == 2050
        assert row["hot_selected_candidates"] == 2048
        assert row["hot_extra_candidates"] == 2
        assert row["hot_cold_extra"] == 1
        assert row["hot_candidate_hot_extra"] == 1


def test_initial_threshold_scores_all_resident_positions_even_when_hot_invalid(token_trace):
    trace = _expanded(token_trace)
    trace.arrays["pre.hot_valid"][:, :2048] = 0
    requests, queries = _report(_refresh(trace))

    assert all(query["initial_positions"] == 4096 for query in queries)
    assert all(query["initial_valid_hot"] == 2048 for query in queries)
    assert all(query["initial_key"] == ONE for query in queries)
    assert all(query["seed_hot_key"] == query["final_key"] == ONE for query in queries)
    assert all(row["cold_demand"] == 2048 for row in requests)
    assert all(row["hot_candidates"] == 0 for row in requests)


def test_rolling_prefix_updates_only_future_admission_and_keeps_early_extras(token_trace):
    selected = np.arange(8000, 10048, dtype=np.int32)
    trace = _expanded(token_trace, history=49152, selected=selected)
    scores = trace.arrays["post.score_keys"]
    scores[:, :, selected] = TWO
    scores[:, :, 5000:5010] = ONE  # Read before the first 16K update.
    scores[:, :, 20000:20010] = ONE  # Suppressed by that completed prefix.
    requests, queries = _report(_refresh(trace))

    for row in requests:
        assert row["hot_extra_candidates"] == 20
        assert row["rolling_candidates"] == 2058
        assert row["rolling_selected_candidates"] == 2048
        assert row["rolling_extra_candidates"] == 10
        assert row["rolling_extra_below"] == 10
        assert row["rolling_cold_extra"] == 10
        assert row["rolling_extra_bytes"] == 10 * row["token_bytes"]
        assert row["final_candidates"] == row["seed_hot_candidates"] == 0
    assert all(query["rolling_updates"] == 2 for query in queries)
    assert all(query["initial_key"] <= query["seed_hot_key"] <= query["final_key"] for query in queries)


def _swap_previous_rows(previous):
    """Change packed request order while retaining each request's TND TopK."""
    swapped = copy.deepcopy(previous)
    swapped.array = lambda name: swapped.arrays[name]
    for name, values in tuple(swapped.arrays.items()):
        if name != "pre.metadata" and name.startswith("pre.") and values.shape[0] == 2:
            swapped.arrays[name] = values[::-1].copy()
    metadata = swapped.arrays["pre.metadata"]
    metadata[2] = ((1 - metadata[2] // 4) * 4) + metadata[2] % 4
    metadata[4].view(np.int32)[:2] = swapped.arrays["pre.row_owners"]
    requests = swapped.manifest["metadata"]["requests"]
    for item in requests:
        row = int(np.flatnonzero(swapped.arrays["pre.row_owners"] == item["owner"])[0])
        item.update(query_begin=row * 4, query_stop=(row + 1) * 4)
    swapped.manifest["metadata"]["requests"] = requests[::-1]
    return swapped


def test_previous_topk_follows_request_identity_after_packed_rows_reorder(token_trace):
    trace = _trace(token_trace)
    previous = predecessor(trace)
    # TND rows 0..3 belong to packed request 1 in the source fixture.
    previous.arrays["post.logical_output"][:4, 0] = 3600
    previous.arrays["post.score_keys"][1, :, 2048] = ZERO  # logical token 0
    previous.arrays["post.score_keys"][1, :, 3600] = ONE
    _refresh(previous)
    expected = _report(trace, previous)
    actual = _report(trace, _swap_previous_rows(previous))

    assert actual == expected
    assert {q["seed_hot_key"] for q in actual[1] if q["packed_row"] == 0} == {ONE}
    assert {q["seed_hot_key"] for q in actual[1] if q["packed_row"] == 1} == {ZERO}


@pytest.mark.parametrize("fault", ["missing", "gap", "identity", "dp", "group"])
def test_missing_or_unrelated_predecessor_is_rejected(token_trace, fault):
    trace = _trace(token_trace)
    previous = predecessor(trace)
    if fault == "gap":
        previous.manifest["metadata"]["context"]["round"] -= 1
    elif fault == "identity":
        previous.manifest["metadata"]["requests"][0]["request_id"] = "new-request"
    elif fault in ("dp", "group"):
        previous.manifest["metadata"]["dp_rank" if fault == "dp" else "group"] += 1
    seed = None if fault == "missing" else seed_state(previous)
    with pytest.raises(ValueError, match="preceding|lifetime"):
        analyze_trace(trace, seed)


def test_seed_state_does_not_retain_scores_and_owns_topk_copy(token_trace):
    trace = _trace(token_trace)
    seed = seed_state(trace)
    retained = {key: value.copy() for key, value in seed["requests"].items()}
    trace.arrays["post.logical_output"].fill(0)
    assert set(seed) == {"dp_rank", "group", "scheduler_round", "layer_names", "requests", "recent"}
    for key, value in seed["requests"].items():
        np.testing.assert_array_equal(value, retained[key])
        assert value.ndim == 1


def test_policy_accounting_reconciles_selected_extra_ties_and_bytes(token_trace):
    trace = _trace(token_trace)
    trace.arrays["post.score_keys"][:, :, 3600] = ONE
    trace.arrays["post.score_keys"][:, :, 3601] = HALF
    requests, queries = _report(_refresh(trace))
    for row in requests:
        for policy in POLICIES:
            assert row[f"{policy}_candidates"] == (
                row[f"{policy}_selected_candidates"] + row[f"{policy}_extra_candidates"]
            )
            assert row[f"{policy}_extra_candidates"] == (row[f"{policy}_extra_ties"] + row[f"{policy}_extra_below"])
            assert row[f"{policy}_extra_bytes"] == row[f"{policy}_extra_candidates"] * row["token_bytes"]
    assert all(query["initial_key"] <= query["seed_hot_key"] <= query["final_key"] for query in queries)


def test_future_logical_id_in_physical_causal_prefix_is_rejected(token_trace):
    trace = _trace(token_trace)
    seed = seed_state(predecessor(trace))
    trace.arrays["pre.logical_ids"][0, [100, 4095]] = [4095, 100]
    with pytest.raises(ValueError, match="future logical"):
        analyze_trace(trace, seed)


def test_compact_rolling_histogram_matches_exact_prefix_order_statistics():
    rng = np.random.default_rng(30)
    keys = np.concatenate(
        [rng.integers(base, base + 2048, 16384, dtype=np.uint16) for base in (0xA000, 0xB000, 0xC000)]
    )
    initial = int(np.partition(keys[:16384], -2048)[-2048])
    actual, updates = _rolling_mask(keys, 16384, initial)
    expected = keys > initial
    boundary = int(np.partition(keys[:32768], -2048)[-2048])
    expected[32768:] = keys[32768:] > boundary
    np.testing.assert_array_equal(actual, expected)
    assert updates == 1
