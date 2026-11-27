# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

import b24_two_row_attach as attach
from four_rank_final_map import archived_assignments, compare, row_assignment
from test_fixed_q4_packed_timing import fixture
import fixed_q4_packed_timing as timing


def data():
    return {f"r0_{stage}{q}": np.empty((0, 5), dtype="<i4") for stage in ("w","q") for q in range(4)}


def entries(pairs):
    return np.array([[s, 100+s, 1, 2, d] for s, d in pairs], dtype="<i4")


def test_complete_map_from_old_journal_requires_unique_result():
    records = data()
    records["r0_q1"] = entries([(4, 1), (5, 2)])
    records["r0_q3"] = entries([(7, 3)])
    assert archived_assignments(records, 1) == [row_assignment([7, 4, 5], [3, 1, 2])]
    # 1 is a destination of Q1 and a source of Q4. Swapping the order
    # can change which hot entry fills each vacated slot.
    records["r0_q3"] = entries([(1, 3), (6, 9)])
    a = row_assignment([4, 5, 1, 6], [1, 2, 3, 9])
    b = row_assignment([1, 6, 4, 5], [3, 9, 1, 2])
    assert a != b and archived_assignments(records, 1) == [None]


def test_controlled_time_requires_ordered_work_and_final_map_on_all_calls():
    check = dict(ordered_match=True, actual_window_sha256="w", actual_final_sha256="q")
    traffic = dict(row_window_prefetch=[[1]*4], row_query_final=[[2]*4])
    base = dict(timed_journal_checks=[check] * 10, traffic=[traffic] * 10,
                expected_window_sha256="w", expected_final_sha256="q",
                mean_us=dict(LI=200.,tail=102.),graph_mean_us=313.)
    alt = dict(base, timed_journal_checks=[dict(check, index_assignment_by_row=["map"])]*10,
               graph_mean_us=291.,mean_us=dict(LI=201.,tail=89.))
    assert compare(base,alt,["map"])["controlled_savings_us"] == dict(graph=22.,LI=-1.,tail=13.)
    assert compare(base,alt,[None])["map_unknown_rows"] == 10
    assert compare(base,alt,["different"])["controlled_savings_us"] is None
    old = dict(base, timed_journal_checks=[dict(check,index_assignment_by_row=["map"])]*10)
    assert compare(old,alt,[None])["guarded_pass"]
    alt["timed_journal_checks"] = [alt["timed_journal_checks"][0]] * 9 + [dict(check, actual_final_sha256="changed",index_assignment_by_row=["map"])]
    assert compare(old,alt,[None])["controlled_savings_us"] is None


def test_profiler_rejects_one_old_tail_among_ten(monkeypatch):
    monkeypatch.setattr(timing, "fixed_stage_records", timing.fixed_stage_records)
    attach.install_profiler()
    cold = "v9_li_large_four_ranks"
    events = fixture(["v9_li_large_hot", cold, attach.KERNEL])
    opts = dict(fused=True, batch=32, expected_li23=cold)
    assert timing.packed_account(events,"scope",1000,**opts)["calls"] == 10
    next(e for e in events if e["name"].startswith(attach.KERNEL))["name"] = "v9_parallel_terminal_large_0"
    with pytest.raises(ValueError,match="Unattributed"):
        timing.packed_account(events,"scope",1000,**opts)
