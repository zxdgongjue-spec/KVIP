# SPDX-License-Identifier: Apache-2.0
"""Regression checks for Cold P2/P4 trace attribution."""

import csv
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'experiments/lifetch_mr23_service'))
from lifetch_cold_trace_comparison import summarize_scopes, write_comparison  # noqa: E402
from lifetch_msprof_timeline import ROUTE_PHASES  # noqa: E402
from lifetch_unified_trace_comparison import write_unified_comparison  # noqa: E402


def span(kernel, phase, start, duration, lane=0, window=1):
    return dict(ph='X', pid=2, tid=lane, ts=start, dur=duration,
                args=dict(kernel_entry=kernel, phase_id=phase, window_epoch=window))


def test_scope_union_does_not_sum_overlapping_lanes_or_other_kernels():
    kernel = 'v9_li_kernel_cold_partitioned'
    document = {'traceEvents': [
        span(kernel, 218, 2, 3), span(kernel, 218, 4, 5, lane=1),
        span(kernel, 218, 9, 2, lane=1, window=2),
        span('v9_li_kernel', 218, 2, 25),
        span(kernel, 192, 1, 1),
    ]}
    rows = {(row['window'], row['phase_id']): row for row in summarize_scopes(document, kernel, 4)}
    assert rows[None, 218]['source_union_us'] == 9
    assert rows[None, 218]['max_lane_union_us'] == 7
    assert rows[1, 218]['source_union_us'] == 7
    assert rows[1, 218]['event_median_us'] == 4
    assert rows[1, 192]['sampled'] is True
    assert rows[3, 151]['events'] == 0


def capture(route, cold_kernel, cold_us, wait, prefetch):
    fixture = dict(fixture_sha256='same_fixture', library_sha256='same_bridge',
                   geometry={'batch': 8}, miss_percent=10, soc='Ascend950PR_9599', aics=32,
                   windows=4, requested_windows=4, prefetch_tokens=512, prefetch_rank_start=1024,
                   warmup=2, service_lanes=8)
    return dict(
        report=dict(**fixture, cold_route=route, local_counts=True, hot_parts=2,
                    cold_parts=4 if wait else 2, total_prefetch=prefetch, total_final_get=1,
                    requests=[dict(request=0, prefetch_by_window=[0, prefetch, 0, 0],
                                   final_get_by_query=[1, 0, 0, 0])]),
        intervals=[
            dict(name='Hot Score', kernel='v9_li_kernel', begin_us=0, end_us=20, duration_us=20),
            dict(name='Cold Score + rotating Service', kernel=cold_kernel, begin_us=40,
                 end_us=40 + cold_us, duration_us=cold_us),
            dict(name='Terminal', kernel='v9_parallel_terminal', begin_us=50 + cold_us,
                 end_us=60 + cold_us, duration_us=10),
        ],
        source={'traceEvents': [span(cold_kernel, 218, 3, wait)] if wait else []},
    )


def test_pair_preserves_native_delta_waits_and_different_get_work(tmp_path):
    captures = {'p2': capture('p2', 'v9_li_kernel_2', 100, 0, 1),
                'cold_p4': capture('cold_p4', 'v9_li_kernel_cold_partitioned', 110, 4, 2)}
    report = write_comparison(tmp_path, captures, local_counts=True)
    assert report['cold_service_native_delta_us'] == 10
    assert report['routes']['cold_p4']['cold_mix_groups'] == 32
    assert report['historical_reference']['delta_us'] == 18.58
    assert report['get_comparison'] == dict(status='different_request_totals', changed_requests=1)
    with (tmp_path / 'cold_trace_scopes.csv').open() as stream:
        wait = next(row for row in csv.DictReader(stream) if row['scope'] == 'partitioned_half_wait')
    assert (wait['p2_events'], wait['cold_p4_events'], wait['delta_source_union_us']) == ('0', '1', '4.0')
    captures['cold_p4']['report']['library_sha256'] = 'wrong_bridge'
    with pytest.raises(ValueError, match='different fixture or build fields'):
        write_comparison(tmp_path, captures, local_counts=True)


def unified_capture(route, score, cold, prefetch):
    (hot, threshold, cold_kernel, terminal) = (kernel for _, kernel, _ in ROUTE_PHASES[route])
    fixture = capture(route, cold_kernel, cold, 0, prefetch)
    fixture['report'].update(local_counts=False, hot_parts=4 if route == 'full_uniform' else 2)
    next_start = 1000
    intervals = []
    for (_, kernel, label), duration in zip(ROUTE_PHASES[route], (score, 10, cold, 60), strict=True):
        intervals.append(dict(name=label, kernel=kernel, begin_us=next_start,
                              end_us=next_start + duration, duration_us=duration))
        next_start += duration + 5
    fixture['intervals'] = intervals
    fixture['source']['traceEvents'] = [
        dict(span(hot, 1, 0, score), name='Score body'),
        dict(span(threshold, 23, score + 5, 10), name='Resident threshold'),
        dict(span(cold_kernel, 154, score + 30, 3), name='Prefetch WQE submission'),
        dict(span(cold_kernel, 223, score + 33, 1), name='Prefetch doorbell'),
        dict(span(terminal, 1, score + 20 + cold, 60), name='Terminal body'),
    ]
    if route != 'p2':
        fixture['source']['traceEvents'].append(dict(span(cold_kernel, 220, score + 20, 2),
                                                     name='Cold T2 score read'))
    return fixture


def test_unified_report_keeps_independent_clocks_and_absent_scopes(tmp_path):
    captures = {'p2': unified_capture('p2', 20, 100, 1),
                'full_p2': unified_capture('full_p2', 90, 110, 1),
                'full_uniform': unified_capture('full_uniform', 30, 105, 2)}
    report = write_unified_comparison(tmp_path, captures, local_counts=False)
    native = report['routes']['p2']
    uniform = report['routes']['full_uniform']
    assert native['score_to_kv_ready_device_us'] == 205
    assert native['source_first_doorbell_us'] == 53
    assert uniform['delta_score_to_kv_ready_device_us'] == 15
    assert uniform['source_first_doorbell_to_terminal_us'] == 92
    assert uniform['get_comparison']['status'] == 'different_request_totals'
    with (tmp_path / 'unified_trace_scopes.csv').open() as stream:
        read = next(row for row in csv.DictReader(stream) if row['phase_id'] == '220')
    assert (read['p2_events'], read['full_uniform_events'], read['full_uniform_delta_source_union_us']) == ('', '1', '')
    captures['full_uniform']['source']['traceEvents'] = [
        event for event in captures['full_uniform']['source']['traceEvents'] if event['args']['phase_id'] != 223]
    with pytest.raises(ValueError, match='missing Terminal or observed prefetch doorbell'):
        write_unified_comparison(tmp_path, captures, local_counts=False)
