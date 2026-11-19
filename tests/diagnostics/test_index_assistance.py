# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from kvip.diagnostics.index_assistance import survey_index_helpers


@pytest.mark.parametrize("other_row_exit,expected_gap", [(215, 5), (227, 0)])
def test_last_index_row_can_differ_from_heaviest_or_last_role(other_row_exit, expected_gap):
    rows = np.full((64, 16), 120, dtype=np.uint64)
    marks = np.zeros((64, 32), dtype=np.uint64)
    marks[:, 3] = 90
    marks[:, 1] = 210
    # The most moves belong to row 7. Row 3 has the last Index finish.
    marks[7, 16] = 99
    marks[3::16, 16] = (1, 2, 3, 4)
    marks[3::16, 17] = (3, 0, 5, 0)
    rows[35, 8] = 200
    marks[35, 3] = 130
    marks[3, 1] = 220
    marks[4, 1] = other_row_exit
    # Only two foreign lanes exited before 200. Row admission at 130 clips
    # the otherwise earlier first core. The target's own lane never qualifies.
    marks[1, 1] = 110
    marks[32, 1] = 170
    marks[19, 1] = 135
    assert survey_index_helpers(rows, marks) == dict(
        last_index_row=3, heaviest_row=7, last_row_pairs=10, last_row_singles=8,
        heaviest_pairs=99, heaviest_singles=0, freed_lanes=2,
        longest_free_window_ticks=70, free_lane_ticks=100,
        last_row_only_exit_gap_ticks=expected_gap)
