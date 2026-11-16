"""Transfer rates use one replay's global span, never summed core durations."""
import pytest

from diagnostics import final_transfer_scope, summarize
from reporting import add_profile_metrics, atomic_json
from types import SimpleNamespace


def capture(width=656):
    cores = [[10, 1000] + [0] * 30 for _ in range(14)]
    requests = [dict(final_counts=[dict(total=0, safe=0, conflict=0) for _ in range(4)]) for _ in range(2)]
    for b, q, safe, conflict, begin, end in ((0, 0, 2, 0, 100, 200),
                                          (1, 0, 3, 0, 110, 210),
                                          (0, 1, 0, 4, 400, 500)):
        requests[b]['final_counts'][q] = dict(total=safe+conflict, safe=safe, conflict=conflict)
        core = cores[6 + 2*q + b]
        core[5:7] = [begin, end]  # UBmem whole worker interval.
        if conflict:
            core[10:12] = [begin, end]
    raw = dict(batch=2, workers_per_query=2, resolver_cores=3, token_bytes=width,
               clock_mhz=10, cores=cores)
    return raw, requests


@pytest.mark.parametrize('transport', ['urma', 'ubmem'])
@pytest.mark.parametrize('width', [656, 2624])
def test_global_span_and_actual_group_width(transport, width):
    result = final_transfer_scope(*capture(width), transport)
    assert result['entries'] == 9 and result['bytes'] == 9 * width
    assert result['span_us'] == 40
    assert result['GBps'] == pytest.approx(9 * width / 40 / 1000)
    assert sum(group['entries'] for group in result['groups']) == 9
    assert len(result['groups']) == 8  # Empty Q3/Q4 groups retained.
    assert 'not first WQE' in result['note']


def test_nonzero_work_requires_actual_timestamps():
    raw, requests = capture()
    raw['cores'][6][5] = 0
    with pytest.raises(ValueError, match='Get timestamps'):
        final_transfer_scope(raw, requests, 'urma')


def test_empty_tasks_are_na_not_zero_bandwidth():
    raw, requests = capture()
    for request in requests:
        request['final_counts'] = [dict(total=0, safe=0, conflict=0) for _ in range(4)]
    result = final_transfer_scope(raw, requests, 'urma')
    assert result['entries'] == 0 and result['GBps'] is None and result['span_us'] is None


def test_level_one_reports_final_and_combined_scope_without_summing_overlap(tmp_path, monkeypatch):
    cores = [dict(role='resolver+finalize b0', stages=[dict(name='Q1.resolve', start_us=3, end_us=7)]),
             dict(role='resolver+finalize b1', stages=[dict(name='Q1.resolve', start_us=6, end_us=9)])]
    monkeypatch.setattr('diagnostics.decode_timeline', lambda raw: dict(cores=cores))
    raw, details = capture()
    details[0]['core_timeline_raw'] = raw
    details[0]['prefetch_scope'] = dict(bytes=656, first_doorbell=50,
        last_cq_observed=150, interval_cycles=100)
    atomic_json(tmp_path / 'full_attribution.json', dict(samples=[dict(
        round=0, final=9, algorithm=dict(totals={}, requests=[]), streaming_cycles=details,
        accounting=dict(stages_us=dict(resolve_update=100)))]))
    config = SimpleNamespace(iterations=1, level=1, stages=('fetch',), transport='urma')
    result = summarize(tmp_path, config)
    assert result['trace'] == 'COMPLETE'
    sample = result['captures'][0]['samples'][0]
    assert sample['combined_observed_bandwidth']['span_us'] == 45
    assert sample['combined_observed_bandwidth']['bytes'] == 10 * 656
    assert 'metric=final_get_bandwidth' in (tmp_path / 'diagnostic_summary.log').read_text()
    row = dict(report=str(tmp_path / 'report.json'))
    add_profile_metrics([row], result)
    assert row['fused_profiled_resolver_span_p50_us'] == 6
    assert row['fused_profiled_final_get_bytes_mean'] == 9 * 656
    assert row['fused_profiled_transfer_span_p50_us'] == 45
    assert row['fused_profiled_transfer_GBps_p50'] == pytest.approx(10 * 656 / 45 / 1000)


def test_invalid_cross_core_order_cannot_produce_global_bandwidth(tmp_path, monkeypatch):
    raw, details = capture()
    details[0]['core_timeline_raw'] = raw
    def invalid(raw):
        raise ValueError('cross-core publication order violated')
    monkeypatch.setattr('diagnostics.decode_timeline', invalid)
    atomic_json(tmp_path / 'full_attribution.json', dict(samples=[dict(
        round=0, final=9, algorithm=dict(totals={}, requests=[]), streaming_cycles=details,
        accounting=dict(stages_us=dict(resolve_update=100)))]))
    config = SimpleNamespace(iterations=1, level=1, stages=('fetch',), transport='urma')
    result = summarize(tmp_path, config)
    assert result['trace'] == 'INCOMPLETE'
    assert 'final_get_bandwidth' not in result['captures'][0]['samples'][0]
