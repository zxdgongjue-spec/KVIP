# SPDX-License-Identifier: Apache-2.0
"""Regression checks for the reference used to audit on-device route history."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'support'))
from ranked_route_replay import expected_mode


def test_last_success_and_current_query_geometry_select_route():
    owners = np.arange(1, 17)
    feedback = np.tile([7, 10, 100, 4], (17, 1))
    ends = np.full((16, 4), 131072)
    assert expected_mode(None, owners, ends, 656, 7, 11) == 0
    assert expected_mode(feedback, owners[::-1], ends, 656, 7, 11) == 1
    assert expected_mode(feedback, owners, ends // 2, 656, 7, 11) == 0
    assert expected_mode(feedback, owners, ends, 2624, 7, 11) == 0
    assert expected_mode(feedback, owners, ends, 656, 8, 11) == 0
    assert expected_mode(feedback, owners, ends, 656, 7, 12) == 0
    feedback[16, 2] = 5000
    assert expected_mode(feedback, owners, ends, 656, 7, 11) == 0


def test_rejection_of_incomplete_predecessor_forces_whole_graph_high():
    owners = np.arange(1, 17)
    ends = np.full((16, 4), 130048)
    feedback = np.tile([3, 12, 20, 4], (17, 1))
    feedback[2, 3] = 1
    assert expected_mode(feedback, owners, ends, 656, 3, 13) == 0



def test_independent_policy_rejects_corrupt_demand_and_out_of_range_queries():
    owners = np.arange(1, 17)
    feedback = np.tile([3, 12, 20, 4], (17, 1))
    ends = np.full((16, 4), 130048)
    feedback[8, 2] = -1
    assert expected_mode(feedback, owners, ends, 656, 3, 13) == 0
    feedback[8, 2] = 20
    ends[8, 2] = 131329
    assert expected_mode(feedback, owners, ends, 656, 3, 13) == 0
