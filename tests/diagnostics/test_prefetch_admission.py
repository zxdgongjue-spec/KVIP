# SPDX-License-Identifier: Apache-2.0
"""CPU-only accounting against real Row word offsets, not device timing."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from kvip.diagnostics.prefetch_admission import mean_admission, prefetch_admission


def rows(posted, suppressed, *, total=8, transport=2048, visited=None):
    posted, suppressed = np.asarray(posted), np.asarray(suppressed)
    batch, rounds = posted.shape
    values = np.zeros((batch, 256), dtype=np.int32)
    values[:, 16 : 16 + rounds] = posted
    values[:, 48 : 48 + rounds] = suppressed
    values[:, 8] = values[:, 9] = posted.sum(axis=1)
    values[:, 10] = suppressed.sum(axis=1)
    values[:, 208:211] = (8192, total, transport)
    visited = np.ones_like(posted, dtype=bool) if visited is None else np.asarray(visited, dtype=bool)
    values[:, 214 : 214 + rounds] = np.where(visited, np.cumsum(visited, axis=1), 0)
    values[:, 246] = visited.sum(axis=1)
    return values


def test_strict_window_limit_ties_exhaustion_and_unused_budget_are_distinct():
    value = rows([[1, 4, 1], [4, 4, 0]], [[0, 3, 0], [1, 2, 0]])
    original = value.copy()
    assert prefetch_admission(value, window_limit=4, rounds=3) == dict(
        rounds=3,
        window_limit=4,
        packed_rows=2,
        active_rows=2,
        visited_epochs=6,
        unvisited_epochs=0,
        window_limited_epochs=2,
        other_or_tied_limited_epochs=1,
        budget_exhausted_epochs=1,
        suppressed_candidates=6,
        remaining_admissible_budget=2,
        rows_suppressed_with_budget_left=1,
        posted_by_epoch=[5, 8, 1],
        suppressed_by_epoch=[1, 5, 0],
    )
    np.testing.assert_array_equal(value, original)


def test_hitting_a_cap_without_rejected_candidates_is_not_attributed_to_that_cap():
    value = rows([[4, 0]], [[0, 0]], visited=[[True, False]])
    result = prefetch_admission(value, window_limit=4, rounds=2)
    assert result["window_limited_epochs"] == result["other_or_tied_limited_epochs"] == 0
    assert result["unvisited_epochs"] == 1 and result["budget_exhausted_epochs"] == 0
    assert result["remaining_admissible_budget"] == 4


def test_transport_limit_and_padding_do_not_look_like_stranded_total_budget():
    value = rows([[2, 0], [0, 0]], [[4, 0], [0, 0]], total=8, transport=2, visited=[[True, True], [False, False]])
    result = prefetch_admission(value.view(np.uint32), window_limit=4, rounds=2, active_rows=np.array([True, False]))
    assert result["active_rows"] == 1
    assert result["other_or_tied_limited_epochs"] == result["budget_exhausted_epochs"] == 1
    assert result["window_limited_epochs"] == result["remaining_admissible_budget"] == 0
    assert result["rows_suppressed_with_budget_left"] == 0


def test_zero_budget_does_not_invent_suppressed_candidates_in_unscanned_windows():
    result = prefetch_admission(rows([[0, 0, 0]], [[0, 0, 0]], total=0), window_limit=64, rounds=3)
    assert result["budget_exhausted_epochs"] == 3
    assert result["suppressed_candidates"] == result["window_limited_epochs"] == 0


def test_allowance_follows_recorded_service_order_instead_of_epoch_number():
    value = rows([[1, 4, 0]], [[2, 3, 0]], total=5)
    value[0, 214:217] = (2, 1, 3)
    result = prefetch_admission(value, window_limit=4, rounds=3)
    assert result["window_limited_epochs"] == 1
    assert result["other_or_tied_limited_epochs"] == 1
    assert result["budget_exhausted_epochs"] == 1
    assert result["posted_by_epoch"] == [1, 4, 0]


def test_padding_epoch_visits_are_counted_but_its_unused_budget_is_excluded():
    value = rows([[4, 0], [0, 0]], [[1, 0], [0, 0]])
    result = prefetch_admission(value, window_limit=4, rounds=2, active_rows=np.array([True, False]))
    assert result["packed_rows"] == 2 and result["active_rows"] == 1
    assert result["visited_epochs"] == 4
    assert result["remaining_admissible_budget"] == 4


@pytest.mark.parametrize(
    "word,value",
    [(8, 3), (9, 3), (10, 1), (17, 2), (18, 1), (50, 1), (216, 1), (214, 2), (246, 0), (16, -1), (209, 1)],
)
def test_corrupt_row_counters_are_rejected(word, value):
    data = rows([[2, 0]], [[0, 0]])
    data[0, word] = value
    with pytest.raises(ValueError):
        prefetch_admission(data, window_limit=4, rounds=2)


def test_suppression_requires_full_positive_allowance_and_a_visited_window():
    for data in (rows([[2]], [[1]]), rows([[0]], [[1]], total=0), rows([[1]], [[0]], visited=[[False]])):
        with pytest.raises(ValueError):
            prefetch_admission(data, window_limit=4, rounds=1)


@pytest.mark.parametrize(
    "change",
    [
        dict(rounds=0),
        dict(rounds=True),
        dict(rounds=33),
        dict(window_limit=0),
        dict(window_limit=481),
        dict(active_rows=[1]),
    ],
)
def test_invalid_geometry_is_rejected(change):
    with pytest.raises(ValueError):
        prefetch_admission(rows([[0]], [[0]]), **(dict(window_limit=4, rounds=1) | change))


def test_mean_retains_vectors_and_old_reports_without_changing_samples():
    first = prefetch_admission(rows([[1, 4, 1]], [[0, 3, 0]]), window_limit=4, rounds=3)
    second = prefetch_admission(rows([[2, 4, 0]], [[0, 2, 0]]), window_limit=4, rounds=3)
    samples = [dict(prefetch_admission=first), dict(prefetch_admission=second)]
    before = deepcopy(samples)
    result = mean_admission(samples)
    assert result["posted_by_epoch"] == [1.5, 4.0, 0.5]
    assert result["suppressed_by_epoch"] == [0.0, 2.5, 0.0]
    assert result["window_limited_epochs"] == result["rows_suppressed_with_budget_left"] == 1
    assert samples == before and mean_admission([dict(prefetch=2)]) is None
    with pytest.raises(ValueError, match="Mixed"):
        mean_admission([samples[0], {}])
    samples[1]["prefetch_admission"]["window_limit"] = 8
    with pytest.raises(ValueError, match="geometry"):
        mean_admission(samples)


def test_fused_accounting_reuses_the_existing_row_readback(monkeypatch):
    import torch
    from kvip.diagnostics import replay_measure

    data = rows([[2, 0]], [[0, 0]])
    data[0, 247] = 1
    native_rows = torch.from_numpy(data)
    completion = torch.zeros((2, 1216), dtype=torch.uint8)
    completion.view(torch.int64)[:, 0] = 0x4B5641464247444F
    rotating = SimpleNamespace(
        rows=native_rows,
        rounds=2,
        row_status=torch.zeros((1, 32), dtype=torch.int32),
        final_posts=torch.zeros((1, 4, 32), dtype=torch.int32),
        prefetch_status=completion,
        final_status=completion.clone(),
        mapping=torch.tensor([[4, -1, 5, 4 + 8192]], dtype=torch.int32),
        arguments=[None] * 9 + [torch.tensor([[[0, 3]]], dtype=torch.int32)],
        hot_valid=torch.ones((1, 4), dtype=torch.int32),
    )
    readbacks = []
    original = replay_measure.cpu_array

    def read(torch, value):
        readbacks.append(value)
        return original(torch, value)

    monkeypatch.setattr(replay_measure, "cpu_array", read)
    result = replay_measure.fused_reads(
        torch,
        rotating,
        SimpleNamespace(status=torch.zeros(1, dtype=torch.int32)),
        "urma",
        2624,
        window_limit=4,
        active_rows=np.array([True]),
    )
    assert sum(value is native_rows for value in readbacks) == 1
    assert result["prefetch_admission"]["posted_by_epoch"] == [2, 0]
    assert result["prefetch_admission"]["remaining_admissible_budget"] == 6
