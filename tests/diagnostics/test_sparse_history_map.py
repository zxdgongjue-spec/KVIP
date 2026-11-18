# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from kvip.diagnostics.replay_measure import validate_sparse_history_map


RESIDENT, CAPACITY = 4, 3
# First tail clips to capacity, second clips to history, third is empty.
BOUNDS = np.array([[5, 10], [10, 15], [8, 8]], dtype=np.int32)


def fixture(write_tail):
    words = np.full((3, 12), RESIDENT, dtype=np.int32)
    if write_tail:
        # Explicit reference independent of the checker: slot = token % 3.
        words[0, 5:8] = RESIDENT + 8192 + np.array([2, 0, 1])
        words[1, 10:12] = RESIDENT + 8192 + np.array([1, 2])
    return words


def verify(words, bounds, write_tail):
    validate_sparse_history_map(words, bounds, RESIDENT, CAPACITY,
                                write_tail=write_tail, step_name="reordered-low")


@pytest.mark.parametrize("write_tail", [False, True])
def test_exact_contract_handles_clipped_empty_and_repacked_rows(write_tail):
    order = [2, 0, 1]
    verify(fixture(write_tail)[order], BOUNDS[order], write_tail)


@pytest.mark.parametrize("write_tail", [False, True])
def test_no_cross_acceptance_of_legacy_and_untouched_contracts(write_tail):
    with pytest.raises(AssertionError, match="reordered-low"):
        verify(fixture(not write_tail), BOUNDS, write_tail)


@pytest.mark.parametrize("row,token", [(0, 6), (0, 8), (1, 9), (2, 8)])
@pytest.mark.parametrize("write_tail", [False, True])
def test_one_corrupt_word_fails_inside_outside_and_past_tail(write_tail, row, token):
    words = fixture(write_tail)
    words[row, token] ^= 1
    with pytest.raises(AssertionError, match="reordered-low"):
        verify(words, BOUNDS, write_tail)


def test_legacy_oracle_detects_wrong_request_bounds_after_repack():
    with pytest.raises(AssertionError, match="reordered-low"):
        verify(fixture(True)[[2, 0, 1]], BOUNDS, True)


@pytest.mark.parametrize("bad", [(-1, 4), (8, 5), (1, 2**32)])
@pytest.mark.parametrize("write_tail", [False, True])
def test_invalid_bounds_do_not_alias_python_indices(bad, write_tail):
    bounds = BOUNDS.astype(np.int64)
    bounds[0] = bad
    with pytest.raises(ValueError, match="Invalid ABC bounds"):
        verify(fixture(write_tail), bounds, write_tail)
