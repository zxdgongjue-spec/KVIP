# SPDX-License-Identifier: Apache-2.0
"""Checks for the three-arm W1 TopK trace contract."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'experiments/lifetch_mr23_service'))
from window_q_trace import analyze_window_q  # noqa: E402


def span(phase, tid, ts, dur, **detail):
    return dict(ph='X', tid=tid, ts=ts, dur=dur, args=dict(
        core_type='AIC' if tid >= 64 else 'AIV', kernel_id=22, window_epoch=1, phase_id=phase, **detail))


OWNER_AIVS = {1: (4,), 2: (8, 9), 4: (16, 17, 24, 25),
              8: (32, 33, 40, 41, 48, 49, 56, 57)}


def window_trace(workers, batch=2):
    spans = [span(21, 64, 0, 20), *(span(22, tid, 0, 30) for tid in range(8 * batch))]
    if batch == 1:
        spans.append(span(150, 5, 25, 40, service_lane=1))
    width = 4 // workers
    for request, owner in enumerate(OWNER_AIVS[batch]):
        spans.extend([
            span(150, owner, 25, 50, service_lane=request),
            span(154, owner, 70, 2, request_or_service_lane=request),
        ])
        if workers > 1:
            spans.extend([
                span(251, owner, 25, 4, request_row=request),
                span(252, owner, 45, 10, request_row=request),
            ])
        for first in range(0, 4, width):
            slot = first // width
            peers = (2, 3, 6) if batch == 1 else tuple(
                8 * (request // 2) + n + request % 2 for n in (2, 4, 6))
            worker = owner if slot == 0 else peers[slot - 1]
            if workers > 1:
                spans.append(span(253, worker, 35, 12, request_row=request, query=first))
            for query in range(first, first + width):
                if (request, query) != (0, 0):
                    spans.append(span(155, worker, 35 + 3 * (query - first), 3,
                                      request_row=request, query=query))
    return {'traceEvents': spans}


@pytest.mark.parametrize('workers', (1, 2, 4))
def test_w1_assignment_and_tail(workers):
    metrics, _ = analyze_window_q(window_trace(workers), workers)
    assert metrics['w1_topk_worker_aivs'] == 2 * workers
    assert metrics['w1_service_owner_aivs'] == 2
    assert metrics['w1_after_last_aiv_score_scan_us'] == 45
    assert metrics['w1_max_request_topk_elapsed_us'] == 12 // workers
    assert metrics['w1_per_q_topk_spans'] == 7


@pytest.mark.parametrize('batch', (1, 4, 8))
@pytest.mark.parametrize('workers', (1, 4))
def test_w1_matrix_assignments(batch, workers):
    metrics, _ = analyze_window_q(window_trace(workers, batch), workers, batch)
    assert metrics['w1_topk_worker_aivs'] == batch * workers
    assert metrics['w1_service_owner_aivs'] == batch
    assert metrics['w1_empty_service_lanes'] == (batch == 1)
    assert metrics['w1_per_q_topk_spans'] == batch * 4 - 1


def test_bs1_empty_lane_cannot_submit():
    source = window_trace(4, 1)
    source['traceEvents'].append(span(154, 5, 70, 2, request_or_service_lane=1))
    with pytest.raises(ValueError, match='Only the W1 request owner'):
        analyze_window_q(source, 4, 1)


def test_a_topk_helper_cannot_submit_wqe():
    source = window_trace(4)
    next(e for e in source['traceEvents'] if e['args']['phase_id'] == 154)['tid'] = 2
    with pytest.raises(ValueError, match='Only the W1 request owner'):
        analyze_window_q(source, 4)


def test_all_expected_slices_must_publish():
    source = window_trace(4)
    event = next(e for e in source['traceEvents'] if e['args']['phase_id'] == 253)
    source['traceEvents'].remove(event)
    with pytest.raises(ValueError, match='Incomplete or duplicate'):
        analyze_window_q(source, 4)


def test_wqe_waits_for_worker_join():
    source = window_trace(2)
    next(e for e in source['traceEvents'] if e['args']['phase_id'] == 154)['ts'] = 51
    with pytest.raises(ValueError, match='before the TopK join'):
        analyze_window_q(source, 2)
