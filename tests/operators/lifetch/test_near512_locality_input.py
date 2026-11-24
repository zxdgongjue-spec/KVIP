from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / 'support'))
from near512_locality_input import expand_locality
from workloads import ReplayRound


def donor():
    batch, history, hot = 2, 8192, 4096
    n = np.arange(history, dtype=np.int32)
    layout = np.tile(n, (batch, 1))
    layout[:, :64], layout[:, 5120:5184] = layout[:, 5120:5184].copy(), layout[:, :64].copy()
    ends = np.array([[8033, 8034, 8035, 8036], [8097, 8098, 8099, 8100]], np.int32)
    tails = np.c_[np.full(batch, 7936), ends[:, -1]]
    # Sortable BF16 values below 32768 encode negative scores. Zero from a
    # padded Key would incorrectly beat every one of the artificial winners.
    scores = np.empty((batch, 4, history), np.uint16)
    keys = np.zeros((batch, history, 128), np.uint8)
    for b in range(batch):
        keys[b, :, :4] = n.view(np.uint8).reshape(-1, 4)
        keys[b, :, 4] = b
        for q in range(4):
            values = 16000 - n // (3 + q)
            values[5250 + q:5270 + q] = 20000
            values[7936:ends[b, q]] = 21000
            scores[b, q] = values
    address = np.ones((batch, history), np.uint8)
    address[:, 7936:] = 0  # ABC may be visible before the DDR page is published.
    shared = tuple((0, page) if page < 62 else None for page in range(history // 128))
    return ReplayRound(1, 1, ((0,'r0',0), (1,'r1',1)),
                       dict(history=history, resident=hot, batch=batch), keys, scores[:,0].astype(np.float32),
                       np.empty((2,4,32,128),np.uint8), np.empty((2,4,32),np.uint16), np.ones((2,4,32),np.float32),
                       ends, tails, address, (shared, shared), layout, np.ones((batch, hot),np.uint8),
                       np.tile(n[:2048],(batch,1)), (), scores, n < ends[:,:,None])


def test_injected_keys_do_not_change_any_winner_or_eligible_cutoff_tie():
    source = donor()
    cap = 16384
    large, audit = expand_locality(source, cap)
    np.testing.assert_equal(large.query_ends, source.query_ends + 8192)
    np.testing.assert_equal(large.tail_bounds, source.tail_bounds + 8192)
    assert large.captured_hot_valid is source.captured_hot_valid and large.captured_lru is source.captured_lru
    assert large.query is source.query and large.weights is source.weights
    for b in range(2):
        m = np.asarray(audit["source_page_to_long_page"][b])
        lifted = (m[:,None]*128 + np.arange(128)).ravel()
        np.testing.assert_equal(large.captured_layout[b, lifted], lifted[source.captured_layout[b]])
        np.testing.assert_equal(np.sort(large.captured_layout[b]), np.arange(cap))
        assert [large.page_keys[b][page] for page in m] == list(source.page_keys[b])
        fill = np.setdiff1d(np.arange(cap), lifted)
        assert len(fill) == 8192 and (fill >= 4096).all() and max(fill) < large.tail_bounds[b,0]
        np.testing.assert_equal(large.logical_address_valid[b,lifted], source.logical_address_valid[b])
        assert large.logical_address_valid[b,fill].all()
        assert large.logical_score_valid[b][:,fill].all()
        np.testing.assert_equal(large.logical_score_valid[b][:,lifted],source.logical_score_valid[b])
        from_row = large.logical_keys[b,:,4]
        assert np.all(from_row == b)
        ids = np.ascontiguousarray(large.logical_keys[b,:,:4]).view(np.int32).ravel()
        np.testing.assert_equal(ids[lifted],np.arange(len(lifted)))
        np.testing.assert_equal(large.logical_scales[b],source.logical_scales[b,ids])
        np.testing.assert_equal(large.logical_score_keys[b], source.logical_score_keys[b][:,ids])
        assert np.all(source.logical_address_valid[b,ids[fill]])
        for q in range(4):
            old = source.logical_score_keys[b,q]
            valid = source.logical_score_valid[b,q]
            cutoff = np.partition(old[valid],-2048)[-2048]
            hot_floor = np.partition(old[source.captured_layout[b,:4096]],-2048)[-2048]
            assert max(large.logical_score_keys[b,q,fill]) < hot_floor <= cutoff < 32768
            expected = lifted[(old >= cutoff) & valid]
            actual = np.flatnonzero((large.logical_score_keys[b,q] >= cutoff) & large.logical_score_valid[b,q])
            np.testing.assert_equal(actual, expected)
    # Extra payloads stay distinct across requests; original shared pages are preserved.
    fill0 = {p for p in large.page_keys[0] if p and p[0] < 0}
    fill1 = {p for p in large.page_keys[1] if p and p[0] < 0}
    assert len(fill0) == len(fill1) == 64 and fill0.isdisjoint(fill1)


def test_uses_strictly_below_cutoff_tokens_even_when_an_entire_page_is_never_safe():
    source = donor()
    source.logical_score_keys[:,:,::128] = 30000
    expanded, details = expand_locality(source, 16384)
    assert all(row["eligible_full_pages"] == 0 for row in details["background"])
    assert expanded.logical_scales.shape == (2,16384)
    source.logical_score_keys.fill(20000)
    with pytest.raises(ValueError,match="below all four hot cutoffs"):
        expand_locality(source, 16384)


def test_final_cutoff_still_guards_rows_with_a_partially_populated_hot_cache():
    source = donor()
    source.captured_hot_valid[:, 3:] = 0
    enlarged, proof = expand_locality(source, 16384)
    assert all(r["hot_cutoffs_bf16_key"] == [None] * 4 for r in proof["background"])
    for row, pages in enumerate(proof["source_page_to_long_page"]):
        filled_pages = np.setdiff1d(np.arange(128), pages)
        scores = enlarged.logical_score_keys[row].reshape(4,128,128)[:, filled_pages, :]
        assert np.all(scores.max(axis=(1,2)) < proof["background"][row]["final_cutoffs_bf16_key"])
