# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from kvip.diagnostics.fetch_rank_check import marker_summary


def test_stage_statistics_accept_array_and_empty_samples():
    from kvip.diagnostics.fetch_rank_check import stats

    assert stats(np.array([1., 2., 6.])) == dict(n=3, mean=3., p50=2.)
    assert stats(np.array([])) == dict(n=0)


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


def test_rank_metadata_bit_preserves_ids_order_and_pending_submissions():
    from kvip.diagnostics.fetch_rank_check import check_state
    from kvip.diagnostics.fetch_window_topk import publication

    ids = np.arange(2048, dtype='<u4')
    scores = np.arange(4096, 2048, -1, dtype='<u2')
    table = [(scores, ids, int(scores[-1])) for _ in range(9)]
    ranks = np.zeros((1, 4, 9, 3104), dtype='<u4')
    ranks[..., :2] = [scores[-1], 2047]
    ranks[..., 32:2080] = ids | np.where(ids % 3 == 0, 0x80000000, 0).astype('<u4')
    ranks[..., 2080:] = scores.view('<u4')
    eligible = (ids % 3 == 0)[None]
    sent = set()
    lists = [ids[eligible[0]].tolist()] * 4
    posts = [[publication(lists, sent, 64, 512) for _ in range(9)]]
    args = (np.arange(1, 10)[None], posts, [[table] * 4], eligible)
    result = check_state(ranks, *args, fixed=True)
    assert (result['ranks'], result['lists']) == (36, 9)
    ranks[0, 0, 0, 32] ^= 1
    with pytest.raises(ValueError):
        check_state(ranks, *args, fixed=True)


def test_lazy_rank_skips_unneeded_queries_but_still_checks_every_post():
    from kvip.diagnostics.fetch_rank_check import check_state
    from kvip.diagnostics.fetch_window_topk import publication

    ids = np.arange(2048, dtype='<u4')
    scores = np.arange(4096, 2048, -1, dtype='<u2')
    table = [(scores, ids, int(scores[-1])) for _ in range(9)]
    ranks = np.zeros((1, 4, 9, 3104), dtype='<u4')
    # Q0 fills each 64-entry window. Q1-3 and the exhausted ninth turn
    # cannot submit; their rank banks have never been initialized.
    for epoch in range(8):
        row = ranks[0, 0, epoch]
        row[:3] = [scores[-1], 2047, epoch+1]
        row[32:2080] = ids | np.uint32(0x80000000)
        row[2080:] = scores.view('<u4')
    sent = set()
    posts = [[publication([ids.tolist()]*4, sent, 64, 512) for _ in range(9)]]
    args = (np.arange(1, 10)[None], posts, [[table]*4], np.ones((1, 2048), dtype=bool))
    result = check_state(ranks, *args, fixed=False, lazy=True)
    assert (result['ranks'], result['lists']) == (8, 9)
    posts[0][4][0] += 1
    with pytest.raises(ValueError, match='Q4 submission mismatch'):
        check_state(ranks, *args, fixed=False, lazy=True)
