# SPDX-License-Identifier: Apache-2.0
"""Validate completed native Row counters against fixed and carried admission."""

from copy import deepcopy

import numpy as np
import pytest

from kvip.diagnostics.prefetch_admission import mean_admission, prefetch_admission


def rows(posted, suppressed, *, total=1024, transport=2048, visited=None, order=None):
    posted, suppressed = np.asarray(posted), np.asarray(suppressed)
    batch, rounds = posted.shape
    values = np.zeros((batch, 256), dtype=np.int32)
    values[:, 16:16 + rounds] = posted
    values[:, 48:48 + rounds] = suppressed
    values[:, 8] = values[:, 9] = posted.sum(axis=1)
    values[:, 10] = suppressed.sum(axis=1)
    values[:, 208] = 8192
    values[:, 209] = total
    values[:, 210] = transport
    visited = np.ones_like(posted, dtype=bool) if visited is None else np.asarray(visited, dtype=bool)
    values[:, 214:214 + rounds] = np.where(visited, np.cumsum(visited, axis=1), 0) if order is None else order
    values[:, 246] = visited.sum(axis=1)
    return values


def test_legacy_window_limit_and_ties_remain_compatible():
    data = rows([[1, 4, 1], [4, 4, 0]], [[0, 3, 0], [1, 2, 0]], total=8)
    before = data.copy()
    report = prefetch_admission(data, window_limit=4, rounds=3)
    assert "admission_controls" not in report
    assert report["window_limited_epochs"] == 2
    assert report["other_or_tied_limited_epochs"] == 1
    assert report["budget_exhausted_epochs"] == 1
    assert report["remaining_admissible_budget"] == 2
    assert report["posted_by_epoch"] == [5, 8, 1]
    np.testing.assert_array_equal(data, before)


def test_controls_override_fixed_geometry_and_uniform_equals_expanded():
    data = rows([[100, 128, 0], [0, 128, 100]], [[0, 2, 0], [0, 1, 0]], total=512)
    with pytest.raises(ValueError, match="row budget"):
        prefetch_admission(data, window_limit=64, rounds=3)
    arguments = dict(window_limit=64, rounds=3)
    uniform = prefetch_admission(data, **arguments, admission_controls=[512, 128, 1, 0])
    expanded = prefetch_admission(data.view(np.uint32), **arguments,
                                 admission_controls=np.array([[512, 128, 1, 0]] * 2, dtype=np.int32))
    assert uniform == expanded
    assert uniform["window_limit"] == 128
    assert uniform["window_limited_epochs"] == 2
    assert uniform["admission_controls"] == [[512, 128, 1, 0]] * 2


def test_carried_quota_subtracts_all_prior_posts():
    data = rows([[0, 10, 374, 128]], [[0, 0, 5, 3]])
    report = prefetch_admission(data, window_limit=64, rounds=4,
                                admission_controls=[1024, 128, 0, 1])
    assert report["window_limited_epochs"] == 2
    assert report["remaining_admissible_budget"] == 512
    assert report["posted_by_epoch"] == [0, 10, 374, 128]
    with pytest.raises(ValueError, match="row budget"):
        prefetch_admission(data, window_limit=64, rounds=4,
                           admission_controls=[1024, 128, 0, 0])


def test_carry_window_uses_logical_epoch_but_prior_uses_service_order():
    data = rows([[0, 256, 128]], [[2, 1, 3]], order=[[2, 1, 3]])
    report = prefetch_admission(data, window_limit=64, rounds=3,
                                admission_controls=[1024, 128, 0, 1])
    # Epoch 1 is first and has 256 credits. Epoch 0 then has zero, despite
    # remaining row capacity; its examined candidates can all be suppressed.
    assert report["window_limited_epochs"] == 3
    assert report["budget_exhausted_epochs"] == 1
    assert report["suppressed_candidates"] == 6


def test_carried_allowance_keeps_480_descriptor_hard_cap():
    data = rows([[0, 0, 0, 480, 160]], [[0, 0, 0, 1, 1]])
    report = prefetch_admission(data, window_limit=64, rounds=5,
                                admission_controls=[1024, 128, 0, 1])
    assert report["window_limited_epochs"] == 2
    corrupt = rows([[0, 0, 0, 481, 159]], [[0, 0, 0, 1, 1]])
    with pytest.raises(ValueError, match="row budget"):
        prefetch_admission(corrupt, window_limit=64, rounds=5,
                           admission_controls=[1024, 128, 0, 1])


def test_each_owner_and_transport_have_independent_limits():
    data = rows([[64, 32], [128, 100]], [[1, 1], [1, 1]], total=[96, 512], transport=[2048, 228])
    controls = [[96, 64, 0, 0], [512, 128, 2, 1]]
    report = prefetch_admission(data, window_limit=64, rounds=2, admission_controls=controls)
    assert report["window_limit"] is None
    assert report["window_limited_epochs"] == 2
    assert report["other_or_tied_limited_epochs"] == 2
    assert report["remaining_admissible_budget"] == 0


def test_zero_quota_is_unvisited_and_padding_budget_not_counted():
    data = rows([[0, 0], [128, 0]], [[0, 0], [1, 0]], total=[0, 512], visited=[[False, False], [True, False]])
    report = prefetch_admission(data, window_limit=64, rounds=2,
                                admission_controls=[[0, 128, 0, 1], [512, 128, 0, 1]],
                                active_rows=np.array([True, False]))
    assert report["visited_epochs"] == 1 and report["budget_exhausted_epochs"] == 0
    assert report["remaining_admissible_budget"] == 0
    data[1, 209] = 500
    with pytest.raises(ValueError, match="actual row totals"):
        prefetch_admission(data, window_limit=64, rounds=2,
                           admission_controls=[[0, 128, 0, 1], [512, 128, 0, 1]],
                           active_rows=np.array([True, False]))
    with pytest.raises(ValueError, match="must not visit"):
        prefetch_admission(rows([[0]], [[0]], total=0), window_limit=64, rounds=1,
                           admission_controls=[0, 128, 0, 1])


@pytest.mark.parametrize("controls", [
    [], [[1024, 128, 0, 1]] * 2, [1024, 128, 0], [1024.0, 128, 0, 1],
    [True, True, False, True], [-1, 128, 0, 1], [2049, 128, 0, 1],
    [1024, 0, 0, 1], [1024, 481, 0, 1], [1024, 128, -1, 1],
    [1024, 128, 65536, 1], [1024, 128, 0, 2],
    np.array([2**64 - 1, 128, 0, 1], dtype=np.uint64),
])
def test_invalid_control_shapes_types_and_native_bounds(controls):
    with pytest.raises(ValueError, match="controls"):
        prefetch_admission(rows([[0]], [[0]]), window_limit=64, rounds=1,
                           admission_controls=controls)


@pytest.mark.parametrize("word,value", [(8, 0), (9, 0), (10, 1), (17, 1), (49, 1),
                                        (215, 1), (214, 2), (246, 0), (16, -1), (209, 512)])
def test_adaptive_accounting_still_rejects_corrupt_native_rows(word, value):
    data = rows([[128]], [[0]])
    data[0, word] = value
    with pytest.raises(ValueError):
        prefetch_admission(data, window_limit=64, rounds=1,
                           admission_controls=[1024, 128, 0, 1])


def test_means_preserve_controls_and_reject_control_changes():
    a = prefetch_admission(rows([[128, 1]], [[1, 0]]), window_limit=64, rounds=2,
                           admission_controls=[1024, 128, 0, 1])
    b = prefetch_admission(rows([[128, 3]], [[2, 0]]), window_limit=64, rounds=2,
                           admission_controls=[[1024, 128, 0, 1]])
    samples = [{"prefetch_admission": a}, {"prefetch_admission": b}]
    before = deepcopy(samples)
    report = mean_admission(samples)
    assert report["admission_controls"] == [[1024, 128, 0, 1]]
    assert report["posted_by_epoch"] == [128.0, 2.0]
    assert samples == before
    samples[1]["prefetch_admission"]["admission_controls"][0][2] = 1
    with pytest.raises(ValueError, match="geometry"):
        mean_admission(samples)
    legacy = prefetch_admission(rows([[1]], [[0]]), window_limit=64, rounds=1)
    assert "admission_controls" not in mean_admission([{"prefetch_admission": legacy}])
    assert mean_admission([{}]) is None
    with pytest.raises(ValueError, match="fields"):
        mean_admission([{"prefetch_admission": legacy}, {"prefetch_admission": a}])
