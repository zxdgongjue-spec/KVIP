# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from kvip.diagnostics.fetch_rank_check import marker_summary


def test_middle_doorbell_is_first_useful_and_empty_turns_count():
    events = np.zeros((9, 6, 16), dtype='<u8')
    events[1:, :, 13:15] = [100, 125]
    events[1, 0, [10, 11, 12, 15]] = [3, 104, 118, 113]
    posts = [[[] for _ in range(9)] for _ in range(16)]
    for row in (0, 6, 12):
        posts[row][1] = [row]
    final = [{6} for _ in posts]
    result = marker_summary(events.view(np.uint8), posts, final, 1)
    assert result['first'] == dict(n=1, mean=4, p50=4)
    assert result['useful'] == dict(n=1, mean=13, p50=13)
    assert result['returned'] == dict(n=48, mean=25, p50=25)
    events[1, 0, 15] = 119
    with pytest.raises(ValueError, match='not monotonic'):
        marker_summary(events.view(np.uint8), posts, final, 1)


def test_live_submission_is_checked_even_if_no_new_rank_was_ready():
    from kvip.diagnostics.fetch_rank_check import check_state

    versions = np.zeros((1, 9), dtype=np.int32)
    posts = [[[] for _ in range(9)]]
    tables = [[[] for _ in range(4)]]
    assert check_state(None, versions, posts, tables, None, fixed=False)['lists'] == 9
    posts[0][2] = [57]
    with pytest.raises(ValueError, match='Q4 submission mismatch'):
        check_state(None, versions, posts, tables, None, fixed=False)
