# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for the ranked Index diagnostic endpoints."""
import numpy as np
import pytest

from ranked_five_report import FLEXIBLE_MAGIC
from ranked_help_report import matched_samples, tail_endpoints


def test_fourth_field_owner_and_final_get_return():
    ticks = np.zeros((2, 16*256+64*32+64*8), dtype=np.uint64)
    index = ticks[:, :16*64].reshape(2, 16, 64)[..., 32:]
    index[..., 22:25] = [90, 201, 202]
    # The second invocation finishes Get later. Its unsigned hardware counters
    # must still produce a negative Index-minus-Get difference.
    fetch = ticks[:, 16*128:16*256].reshape(2, 16, 128)
    fetch[..., 99] = 400
    fetch[..., 104:106] = [202, 204]
    workers = fetch[:, :, :96].reshape(2, 16, 4, 24)
    workers[..., 3:5] = [95, 100]
    workers[1, ..., 4] = 300
    workers[..., 16] = 1
    workers[..., 7:9] = [330, 350]  # empty conflict phase
    workers[..., 9] = 380  # publication includes dependency and residency work

    fields = ticks[:, 16*256+64*32:].reshape(2, 64, 8)
    for slot in range(64):
        row, field = divmod(slot, 3) if slot < 48 else (slot-48, 3)
        owner = row*3 if field == 3 else slot
        arrival, start, end = (120, 130, 200) if field == 3 else (80, 100, 120)
        fields[:, slot] = [FLEXIBLE_MAGIC, row, owner<<32 | field, 10, arrival, end, start, 0]
    raw = {'times': ticks, 'cycles_per_us': 2}
    tail = tail_endpoints(raw, 'flexible')
    assert (tail['index_later'], tail['get_later']) == (1, 1)
    assert [call['index_minus_nonempty_get_us'] for call in tail['calls']] == [51., -49.]
    assert tail['fourth_owner_busy_rows'] == {'producer_finalizer': 32}
    assert tail['row_handoff_gap_us'] == {'n': 32, 'p50': 5., 'p95': 5.}
    assert tail['join_minus_last_field_us']['p50'] == .5
    fields[0, 48, 2] = 3<<32 | 3
    with pytest.raises(ValueError, match='physical owner'):
        tail_endpoints(raw, 'flexible')


def test_tied_topk_matches_only_equal_fingerprints():
    def arm(digests, spans):
        return dict(digests=digests, samples=[dict(us=s, stages_us=dict(li=s, resolve_update=s)) for s in spans])
    before = arm(['a', 'a', 'b'], [100., 1000., 60.])
    after = arm(['b', 'c', 'a'], [50., 1., 95.])
    m = matched_samples(before, after)
    assert m['pairs'] == [(0, 2), (2, 0)]
    assert m['span_p50_saved_us'] == 7.5
