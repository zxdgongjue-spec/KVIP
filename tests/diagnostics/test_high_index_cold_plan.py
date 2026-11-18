# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from kvip.diagnostics.high_index_cold_plan import MAGIC, decode_choices, expected_choice, verify_choices
from kvip.diagnostics.high_index_profile import verify_permutation


def test_raw_sources_and_chunk_local_victims_define_eligibility():
    src = np.arange(4096, 4096 + 4096)
    dst = np.arange(len(src)) % 512
    assert expected_choice(src, dst, [1024, 2, 1, 0, 0], 1024, 16384) == (
        True, 1027, (1539, 0))
    # Reusing the first raw cold source in Q1 rejects after the whole Q1 phase is submitted.
    dup = src.copy()
    dup[2050] = src[0]
    assert expected_choice(dup, dst, [1024, 2, 1, 3, 4], 1024, 16384) == (False, 1027, None)
    broken = dst.copy()
    broken[511] = 0
    assert expected_choice(src, broken, [1024, 2, 1, 3, 4], 1024, 16384) == (False, 1024, None)
    assert expected_choice(src, dst, [0]*5, 16385, 210944) == (False, 0, None)
    assert expected_choice(src, dst, [0]*5, 16384, 210945) == (False, 0, None)
    assert expected_choice(src, dst, [0]*5, 16384, 210944) == (True, 0, (0,0))


def test_fast_fallback_and_mode_oracle_share_one_batch():
    marks = np.zeros((64,32), np.uint64)
    marks[32:48,23:25] = [MAGIC, 1]
    marks[33,24] = 2
    marks[32:34,25] = [3,3]
    marks[32:48,21] = 100
    marks[32:48,26] = 99
    rows = decode_choices(marks, expected=True)
    flags = np.zeros((16,16,32), np.int32)
    flags[0:2,[0,12,13],1] = 1
    src = np.zeros((16,10240), np.int32)
    dst = src.copy()
    src[0,[0,2048,2049]] = [7,8,9]
    dst[0,[0,2048,2049]] = [0,0,1]
    src[1,[0,2048,2049]] = [7,8,7]
    dst[1,[0,2048,2049]] = [0,1,2]
    before = np.tile(np.arange(16),(16,1))
    after = before.copy()
    after[0,[0,7,8,1,9]] = [8,0,7,9,1]
    after[1,[0,1,2,7,8]] = [2,8,7,0,1]
    assert verify_permutation(flags, src, dst, before, after, range(16))[:2] == [2,2]
    prec = dict(row_moves=[5,5]+[0]*14, row_pairs=[1,1]+[0]*14)
    profile = dict(cold_plan=rows, preplan=prec)
    verify_choices(profile,flags,src,dst,4,16)
    assert rows["fast_rows"] == [0,*range(2,16)] and rows["fallback_rows"] == [1]
    assert (rows["fast_events"],rows["fallback_events"]) == (3,3)
    rows["mode"][1] = 1
    with pytest.raises(ValueError,match="differs from CPU eligibility"):
        verify_choices(profile,flags,src,dst,4,16)


def test_choices_require_positive_build_evidence_and_planner_ownership():
    marks = np.zeros((64,32),np.uint64)
    assert decode_choices(marks, expected=False) is None
    with pytest.raises(ValueError, match="Missing"):
        decode_choices(marks, expected=True)
    marks[:,23] = MAGIC
    with pytest.raises(ValueError, match="Invalid"):
        decode_choices(marks, expected=True)
