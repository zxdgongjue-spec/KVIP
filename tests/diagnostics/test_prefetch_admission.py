"""Validate native admission counters when nine publications exhaust 512 slots."""

import numpy as np
import pytest

from kvip.diagnostics.prefetch_admission import prefetch_admission, mean_admission


def native_row(posted, suppressed, *, limits=(2048, 512, 2048), orders=None):
    rounds = len(posted)
    row = np.zeros((1, 256), np.uint32)
    row[0, 8:10] = sum(posted)
    row[0, 10] = sum(suppressed)
    row[0, 16:16+rounds] = posted
    row[0, 48:48+rounds] = suppressed
    row[0, 208:211] = limits
    row[0, 214:214+rounds] = orders or list(range(1, rounds + 1))
    row[0, 246] = rounds
    return row


def test_ninth_publication_can_examine_candidates_after_total_budget_exhaustion():
    row = native_row([64]*8 + [0], [10]*8 + [137])
    out = prefetch_admission(row, window_limit=64, rounds=9)
    assert out["remaining_admissible_budget"] == 0
    assert out["window_limited_epochs"] == 7
    assert out["other_or_tied_limited_epochs"] == 2
    assert out["budget_exhausted_epochs"] == 1
    assert out["exhausted_with_observed_candidates"] == 1
    assert out["exhausted_suppressed_candidates"] == 137
    assert out["suppressed_candidates"] == 217
    assert mean_admission([{"prefetch_admission": out}]*2)["exhausted_suppressed_candidates"] == 137


@pytest.mark.parametrize("limits", [(125, 512, 2048), (2048, 125, 2048), (2048, 512, 125)])
def test_partial_remaining_budget_and_zero_allowance_are_both_valid(limits):
    out = prefetch_admission(native_row([64, 61, 0], [1, 3, 20], limits=limits), window_limit=64, rounds=3)
    assert out["posted_by_epoch"] == [64, 61, 0]
    assert out["exhausted_suppressed_candidates"] == 20


def test_zero_observations_do_not_claim_that_the_native_scan_was_skipped():
    out = prefetch_admission(native_row([64]*8 + [0], [0]*9), window_limit=64, rounds=9)
    assert out["budget_exhausted_epochs"] == 1
    assert out["exhausted_with_observed_candidates"] == 0


def test_budget_is_reconstructed_from_recorded_service_order():
    row = native_row([0, 64, 64], [17, 3, 4], limits=(2048, 128, 2048), orders=[3, 1, 2])
    out = prefetch_admission(row, window_limit=64, rounds=3)
    assert out["exhausted_suppressed_candidates"] == 17


def test_real_over_budget_and_inconsistent_suppression_remain_errors():
    with pytest.raises(ValueError, match="exceeds an actual row budget"):
        prefetch_admission(native_row([64]*8 + [1], [0]*9), window_limit=64, rounds=9)
    with pytest.raises(ValueError, match="row=0 epoch=0 posted=63.*allowance=64"):
        prefetch_admission(native_row([63], [10]), window_limit=64, rounds=1)
    row = native_row([64]*8 + [0], [0]*8 + [100])
    row[0, 10] += 1
    with pytest.raises(ValueError, match="suppression counters disagree"):
        prefetch_admission(row, window_limit=64, rounds=9)
