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
