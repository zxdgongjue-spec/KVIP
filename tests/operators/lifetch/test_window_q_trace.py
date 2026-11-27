# SPDX-License-Identifier: Apache-2.0
"""Checks for the three-arm W1 TopK trace contract."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'experiments/lifetch_mr23_service'))
from window_q_trace import analyze_window_q, window_prefetch_get_counts  # noqa: E402


def span(phase, tid, ts, dur, *, epoch=1, **detail):
    return dict(ph='X', tid=tid, ts=ts, dur=dur, args=dict(
        core_type='AIC' if tid >= 64 else 'AIV', kernel_id=22, window_epoch=epoch, phase_id=phase, **detail))


OWNER_AIVS = {1: (4,), 2: (8, 9), 4: (16, 17, 24, 25),
              8: (32, 33, 40, 41, 48, 49, 56, 57),
              16: (32, 33, 36, 37, 40, 41, 44, 45, 48, 49, 52, 53, 56, 57, 60, 61)}


def window_trace(workers, batch=2):
    score_aivs = (4 if batch == 16 else 8) * batch
    spans = [span(21, 64, 0, 20), *(span(22, tid, 0, 30) for tid in range(score_aivs))]
    if batch == 1:
        spans.append(span(150, 5, 25, 40, service_lane=1))
    width = 4 // workers
    for request, owner in enumerate(OWNER_AIVS[batch]):
        if batch == 16:
            spans.append(span(150, owner - 32, 0, 10, epoch=0, service_lane=request))
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
            if batch == 16:
                peers = (owner - 30, owner + 2, owner - 32)
            elif batch == 1:
                peers = (2, 3, 6)
            else:
                peers = tuple(8 * (request // 2) + n + request % 2 for n in (2, 4, 6))
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


@pytest.mark.parametrize('batch', (1, 4, 8, 16))
@pytest.mark.parametrize('workers', (1, 2, 4))
def test_w1_matrix_assignments(batch, workers):
    metrics, _ = analyze_window_q(window_trace(workers, batch), workers, batch)
    assert metrics['w1_topk_worker_aivs'] == batch * workers
    assert metrics['w1_service_owner_aivs'] == batch
    assert metrics['w1_empty_service_lanes'] == (batch == 1)
    assert metrics['w1_reused_w0_owner_aivs'] == (16 if batch == 16 and workers == 4 else 0)
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


def test_bs16_w0_owner_must_finish_before_helping_w1():
    source = window_trace(4, 16)
    w0 = next(e for e in source['traceEvents'] if e['args']['window_epoch'] == 0)
    w0['dur'] = 40
    with pytest.raises(ValueError, match='before finishing W0 Service'):
        analyze_window_q(source, 4, 16)


def bs16_four_window_trace(workers):
    spans = [span(21, 64, 35, 12, epoch=3), *(span(22, tid, 36, 12, epoch=3) for tid in range(64))]
    for row, w1_owner in enumerate(OWNER_AIVS[16]):
        cores = (w1_owner - 32, w1_owner, w1_owner - 30, w1_owner + 2)
        for epoch, core in enumerate(cores):
            start = 12 * epoch if epoch < 3 else 49
            duration = 8 if epoch < 3 else 41
            spans.append(span(150, core, start, duration, epoch=epoch, service_lane=row))
            if epoch:
                spans.append(span(151, core, start, epoch, epoch=epoch, service_lane=row))
        w3_owner = cores[3]
        spans.append(span(154, w3_owner, 80, 2, epoch=3, request_or_service_lane=row))
        if workers == 4:
            spans.extend([
                span(251, w3_owner, 50, 2, epoch=3, request_row=row),
                span(252, w3_owner, 59, 10, epoch=3, request_row=row),
            ])
        for query, core in enumerate((w3_owner, *cores[:3])):
            topk_core = core if workers == 4 else w3_owner
            start = 55 if workers == 4 else 55 + query * 3
            if workers == 4:
                spans.append(span(253, topk_core, start, 4, epoch=3, request_row=row, query=query))
            spans.append(span(155, topk_core, start, 3, epoch=3, request_row=row, query=query))
    spans.extend(span(180, 64, 12 * epoch, 0.5 * epoch, epoch=epoch) for epoch in range(1, 4))
    return {'traceEvents': spans}


@pytest.mark.parametrize('workers', (1, 4))
def test_bs16_w4_last_window_workers_and_waits(workers):
    metrics, scopes = analyze_window_q(bs16_four_window_trace(workers), workers, 16, windows=4)
    assert metrics['w3_topk_worker_aivs'] == workers * 16
    assert metrics['w3_reused_earlier_owner_aivs'] == (48 if workers == 4 else 0)
    assert metrics['w3_after_last_aic_qk_scan_us'] == 43
    assert metrics['w3_aiv_score_span_us'] == 12
    assert metrics['w3_max_request_topk_elapsed_us'] == (3 if workers == 4 else 12)
    assert metrics['w3_max_owner_topk_join_us'] == (10 if workers == 4 else 0)
    assert metrics['w2_max_owner_predecessor_wait_us'] == 2
    assert metrics['w3_max_owner_predecessor_wait_us'] == 3
    assert metrics['w3_sampled_aic_credit_wait_spans'] == 1
    assert metrics['w3_max_sampled_aic_credit_wait_us'] == 1.5
    assert metrics['w0_max_sampled_aic_credit_wait_us'] is None
    assert {event['details']['window_epoch'] for event in scopes} == {0, 1, 2, 3}


@pytest.mark.parametrize('phase', (251, 252, 253))
def test_bs16_w4_cannot_split_earlier_window(phase):
    source = bs16_four_window_trace(4)
    next(e for e in source['traceEvents'] if e['args']['phase_id'] == phase)['args']['window_epoch'] = 2
    with pytest.raises(ValueError, match='only in W3'):
        analyze_window_q(source, 4, 16, windows=4)


@pytest.mark.parametrize('epoch', (0, 1, 2))
def test_bs16_w4_helpers_wait_for_original_service(epoch):
    source = bs16_four_window_trace(4)
    service = next(e for e in source['traceEvents']
                   if e['args']['phase_id'] == 150 and e['args']['window_epoch'] == epoch)
    service['dur'] = 60
    with pytest.raises(ValueError, match=f'before finishing W{epoch} Service'):
        analyze_window_q(source, 4, 16, windows=4)


def test_window_prefetch_counts_keep_effective_windows():
    records = [dict(request=0, prefetch_requests=5, prefetch_by_window=[0, 5]),
               dict(request=0, prefetch_requests=7, prefetch_by_window=[2, 5])]
    assert window_prefetch_get_counts(records, 2, calls=2) == {
        'w0_prefetch_get_requests_per_call': 1, 'w1_prefetch_get_requests_per_call': 5,
    }
    with pytest.raises(ValueError, match='invalid window prefetch'):
        window_prefetch_get_counts(records, 4)
