# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import numpy as np
import pytest

import near512_profile as prof


def make_example():
    history = 2304
    long_history = 2816
    g = dict(batch=1, history=history, resident=1024, output_stride=1280)
    old = SimpleNamespace(geometry=g, captured_layout=np.arange(history)[None,:],
                          captured_hot_valid=np.ones((1,1024), bool), tail_bounds=np.array([[2176, 2304]]))
    scores = np.full(history, 2, np.uint16)   # Cutoff ties can move between residency classes.
    scores[0] = 9
    scores[1] = 1
    old.logical_score_keys = np.broadcast_to(scores, (1,4,history))
    old.logical_score_valid = np.ones((1,4,history), bool)
    old.logical_score_valid[0,2,-1] = False
    old_ids = np.broadcast_to(np.r_[0, np.arange(2,2049)], (1,4,2048)).copy()
    pages = np.r_[np.arange(8), np.arange(8,history//128)+4][None,:]
    def move(ids):
        return pages[0,ids//128]*128 + ids%128
    new = SimpleNamespace(geometry=dict(g, history=long_history), captured_layout=np.arange(long_history)[None,:],
                          captured_hot_valid=old.captured_hot_valid.copy(), tail_bounds=old.tail_bounds + 512)
    return old, new, pages, old_ids, move


def test_boundary_ties_and_mapped_residency():
    old, new, pages, ids, move = make_example()
    prof.bind_source_check(old,new,pages)
    old_counts = dict(unique=2048, hot=1023, abc=0, cold=1025)
    # A tied cold token is replaced by a tied ABC token; Q still has the stock cutoff.
    alternate = ids.copy()
    alternate[:,:,-1] = 2180
    assert np.array_equal(new.check_donor_topk(move(alternate)), alternate)
    compared = prof.compare_profiles(dict(native_unique=old_counts),new,move(alternate))
    assert compared['short'] == old_counts
    assert compared['delta'] == dict(unique=0, hot=0, abc=1, cold=-1)
    assert compared['checked_queries'] == 4
    for where, replacement, why in [((0,0,0), move(2200), 'beyond cutoff ties'),
                                    ((0,0,1), move(1), 'beyond cutoff ties'),
                                    ((0,0,1), 1024, 'Added history'),
                                    ((0,2,1), move(2303), 'invisible'),
                                    ((0,1,1), move(0), 'repeated')]:
        bad = move(ids)
        bad[where] = replacement
        with pytest.raises(ValueError,match=why):
            new.check_donor_topk(bad)
    new.captured_hot_valid[0,0] = False
    prof.bind_source_check(old,new,pages)
    with pytest.raises(ValueError,match='classification'):
        new.check_donor_topk(move(ids))


def test_original_comparison_requires_verified_serial_and_all_20_requests(tmp_path, monkeypatch):
    old, new, pages, ids, move = make_example()
    old.validate_topk = lambda selected: np.testing.assert_array_equal(selected, ids)
    folder = tmp_path/'mainline_checks.json'
    sha = prof.topk_fingerprint(ids)
    fields = {'query': 'aa'}
    check = dict(group=7,calls=20,queries=1280,serial_set_digest=sha,
                 native_reference=dict(origin='completed-owner-archive', identity=dict(inputs=fields), archive_path='owner-root'))
    (tmp_path/'score_topk_check.json').write_text(json.dumps(check))
    traffic = dict(prefetch=[25]*20,final=list(range(20)))
    proof = dict(capture_operand_hashes=fields,history_cap=dict(measured_requests=traffic))
    folder.write_text(json.dumps(proof))
    monkeypatch.setattr(prof, 'archived_oracle', lambda path, group: (ids, dict(serial_set_digest=sha)))
    ref = prof.archive_profile(old,7,folder)
    assert ref['native_unique']['cold'] == 1025
    comparison = prof.traffic_comparison(ref,[{'prefetch':24,'final':11}]*10)
    assert comparison['delta'] == dict(prefetch=-1, final=1.5)
    traffic['final'].pop()
    folder.write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='incomplete 128K traffic'):
        prof.archive_profile(old,7,folder)
