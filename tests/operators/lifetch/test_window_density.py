# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
from window_density import summarize


def event(epoch, arrived, after_wait, finished, tokens, bell=0):
    return [after_wait, finished, epoch << 32, 1, arrived, 0, 0, 0, 0, tokens, tokens,
            bell, bell, after_wait, after_wait, 0]


def test_overlap_and_link_backlog_are_separate():
    events = [[event(0, 100, 100, 130, 3, 110), event(0, 105, 105, 120, 0)],
              [event(1, 125, 131, 145, 2, 142), event(1, 140, 140, 155, 0)]]
    li = dict(clock_mhz=1., per_query_cycle_stamps=[[[1, 5, 100, 160]] * 4] * 2)
    stats = summarize(events, li, 16000)
    assert stats["adjacent_epochs_arrival_envelope_overlap_us"] == [5]
    assert stats["adjacent_epochs_distinct_row_service_overlap_us"] == [0]
    assert stats["epochs"][1]["rows_arrived_before_previous_end"] == 1
    assert stats["epochs"][1]["mean_exposed_previous_service_us"] == 2.5
    other = summarize(events, li, 656, sibling_checks=[[(110, 133)], [(121, 125)]])["epochs"][1]
    assert other["sibling_rows_arrived_before_previous_end"] == 1
    assert other["mean_sibling_exposed_previous_us"] == 10
    assert other["mean_sibling_check_us"] == 13.5
    lanes = dict(sibling=[[(105, 110)], [(109, 121)]], post_owner=[[(131, 132)], [(124, 130)]])
    over = summarize(events, li, 656, score_spans=lanes)["epochs"][1]
    assert over["sibling_score_overlaps_previous_service_rows"] == 2
    assert over["sibling_mean_score_overlap_previous_service_us"] == 8
    assert over["post_owner_score_overlaps_previous_service_rows"] == 0
    assert stats["longest_inter_post_us"] == 32
    informative = summarize(events, li, 16000, winners=[[2, 4], []], sources=[[1, 5, 0, 2, 4], []])
    assert [w["used_by_final_topk"] for w in informative["epochs"]] == [0, 2]
    assert informative["first_useful_post_us"] == 42
    assert informative["mean_useful_lead_to_li_end_us"] == 18
    assert stats["ideal_link_by_payload_GBps"][16]["busy_during_li_us"] == 5
    assert stats["ideal_link_by_payload_GBps"][16]["drain_after_li_us"] == 0
    assert summarize(events, li, 320000)["ideal_link_by_payload_GBps"][16]["drain_after_li_us"] == 50


def test_no_posts_and_invalid_doorbell():
    li = dict(clock_mhz=2., per_query_cycle_stamps=[[[1, 5, 100, 200]] * 4])
    empty = summarize([[event(0, 101, 103, 140, 0)]], li, 656)
    assert empty["first_post_us"] is None
    assert empty["ideal_link_by_payload_GBps"][16]["idle_during_li_us"] == 50
    with pytest.raises(ValueError, match="Post outside service"):
        summarize([[event(0, 101, 103, 140, 4, 90)]], li, 656)


def test_concurrent_rows_use_union_of_service_intervals():
    old = [event(0, 100, 100, end, 0) for end in (130, 190, 180)]
    new = [event(1, begin, begin, end, 0) for begin, end in ((140, 170), (195, 198), (185, 195))]
    li = dict(clock_mhz=1., per_query_cycle_stamps=[[[1, 5, 100, 200]] * 4] * 3)
    stats = summarize([old, new], li, 656)
    assert stats["adjacent_epochs_distinct_row_service_overlap_us"] == [35]
    assert stats["epochs"][1]["rows_with_zero_post"] == 3
    new[0][3] |= 1 << 32
    with pytest.raises(ValueError, match="Skipped service"):
        summarize([old, new], li, 656)


def test_split_role_scores_can_cross_two_earlier_windows():
    from copy import deepcopy
    from window_density import split_role_intervals

    clocks = [0] * 128
    events = [[], [], [], []]
    li = dict(clock_mhz=2., per_query_cycle_stamps=[[[1, 3, 5, 150]] * 4, [[2, 4, 6, 152]] * 4])
    for r in (0, 1):
        clocks[r * 64:r * 64 + 6] = [11, 55, 56, 86, 87, 110]
        for i, (w, b, e) in enumerate([(10, 10, 40), (45, 60, 73), (74, 87, 99), (100, 111, 141 - r)]):
            clocks[r * 64 + 16 + i * 3:r * 64 + 19 + i * 3] = [w,b,e]
            events[i].append(event(i, b + 1, b + 2, e - 1, 0))
    result = split_role_intervals(clocks, events, li)
    a = result["rows"][0]
    assert (result["last_li3_row"], result["last_window_row"]) == (1, 0)
    assert a["score_us"] == [22., 15., 11.5] and a["wait_ready_us"] == [7.5, 6.5, 5.5]
    assert a["score_overlapped_by_service_us"] == [14.5, 6.5, 6.]
    assert a["wait_for_unfinished_score_us"] == [5., 6., 5.]
    late = deepcopy(clocks)
    late[1] = 65
    with pytest.raises(ValueError, match="score clocks|ready or service"):
        split_role_intervals(late, events, li)
    wrong = deepcopy(events)
    wrong[1][0][1] = 95
    with pytest.raises(ValueError, match="ready or service"):
        split_role_intervals(clocks, wrong, li)
    early = deepcopy(clocks)
    early[2:4] = [46, 86]
    early[:2] = [11, 44]
    early[4:6] = [72, 110]
    early[3] = 71
    # Score segment 3 runs through window 1 and 2, separated by a ready wait.
    a = split_role_intervals(early, events, li)["rows"][0]
    assert a["score_service_overlap_matrix_us"][2] == [0., .5, 6., 0.]


def test_shared_final_rank_clocks_match_the_post_owner():
    from copy import deepcopy
    from window_density import split_role_intervals

    clocks = [0] * 64
    clocks[:8] = [10, 40, 42, 62, 63, 106, 115, 128]
    intervals = [(8, 8, 30), (31, 41, 52), (53, 64, 78), (79, 108, 143)]
    events = []
    for epoch, (wait, ready, end) in enumerate(intervals):
        clocks[16 + 3 * epoch:19 + 3 * epoch] = [wait, ready, end]
        events.append([event(epoch, ready + 1, ready + 2, end - 1, 0)])
    clocks[28:31] = [114, 125, 132]
    events[-1][0][0] = 133
    events[-1][0][13:15] = [114, 132]
    li = dict(clock_mhz=1., per_query_cycle_stamps=[[[1, 3, 7, 149]] * 4])
    rank = split_role_intervals(clocks, events, li, shared_final=True)["rows"][0]["shared_final"]
    assert (rank["v0_rank_us"], rank["v1_rank_us"], rank["rank_overlap_us"], rank["v1_join_us"]) == (13, 11, 10, 7)
    incomplete = clocks.copy()
    incomplete[7] = 0
    with pytest.raises(ValueError, match="Invalid shared final"):
        split_role_intervals(incomplete, events, li, shared_final=True)
    wrong_writer = deepcopy(events)
    wrong_writer[3][0][13] = clocks[6]
    with pytest.raises(ValueError, match="Invalid shared final"):
        split_role_intervals(clocks, wrong_writer, li, shared_final=True)
