import csv
import json

import pytest

from reporting import CorrectnessRecorder, comparison_row, save_summary, timing_summary


def test_incorrect_output_retains_measured_performance(tmp_path):
    row = comparison_row('b16', timing_summary([490, 500, 510]),
                         timing_summary([340, 350, 360]), 'FAIL')
    assert row['reduction'] == pytest.approx(.3)
    assert row['fused_mean_us'] == 350
    assert not row['accepted']
    save_summary(tmp_path, [row])
    assert json.loads((tmp_path / 'summary.json').read_text())['cases'][0] == row
    text = (tmp_path / 'summary.log').read_text()
    assert all(value in text for value in ('FAIL', '500.00', '350.00', '+30.00%'))


def test_missing_timing_is_not_zero_or_success():
    row = comparison_row('timeout', timing_summary([20]), None, 'NOT_RUN', 'FAILED')
    assert row['fused_mean_us'] is None
    assert row['reduction'] is None
    assert row['timing'] == 'INCOMPLETE'
    assert not row['accepted']


def test_only_correctness_assertions_can_be_recorded_and_continued():
    checks = CorrectnessRecorder()
    def incorrect():
        raise AssertionError('wrong KV')
    def device_failure():
        raise RuntimeError('stream failed')
    assert not checks.check('bytes', incorrect)
    assert checks.status == 'FAIL'
    with pytest.raises(RuntimeError, match='stream failed'):
        checks.check('device', device_failure)


@pytest.mark.parametrize('samples', [[], [0], [-1], [float('nan')], [float('inf')]])
def test_invalid_timing_rejected(samples):
    with pytest.raises(ValueError):
        timing_summary(samples)


def test_missing_batch_report_cannot_be_success(tmp_path):
    from types import SimpleNamespace
    from reporting import atomic_json, collect_results, timing_summary
    config = SimpleNamespace(transport='urma', mode='bench', batches=(1, 2), fetch_jettys_per_aiv=1)
    atomic_json(tmp_path / 'one/report.json', dict(batch=1, correctness='PASS', fetch_jettys_per_aiv=2,
        fetch_path='parallel_terminal', effective_fetch_jettys_per_aiv=1,
        native_misses_per_request=[0], fused_read_counts=[dict(prefetch=0, final=0)], timing=dict(
            native_chain=timing_summary([10]), prototype_with_update=timing_summary([8]))))
    rows = collect_results(tmp_path, config, 0)
    assert len(rows) == 2
    assert rows[0]['accepted'] and not rows[1]['accepted']
    assert rows[0]['fetch_jettys_per_aiv'] == 2
    assert rows[0]['fetch_path'] == 'parallel_terminal'
    assert rows[0]['effective_fetch_jettys_per_aiv'] == 1
    assert rows[0]['serial_read_bytes'] == rows[0]['fused_total_bytes_mean'] == 0
    assert 'fused_total_entries_max' not in rows[1]
    assert rows[1]['batch'] == 2 and rows[1]['execution'] == 'INCOMPLETE'


def test_late_worker_failure_does_not_erase_completed_timing(tmp_path):
    from types import SimpleNamespace
    from reporting import atomic_json, collect_results, timing_summary
    config = SimpleNamespace(transport='urma', mode='bench', batches=(1,))
    atomic_json(tmp_path / 'one/report.json', dict(batch=1, correctness='PASS', timing=dict(
        native_chain=timing_summary([10]), prototype_with_update=timing_summary([8]))))
    rows = collect_results(tmp_path, config, -11)
    assert rows[0]['fused_mean_us'] == 8 and rows[1]['execution'] == 'FAILED'


def test_trace_failure_preserves_unprofiled_partial_record(tmp_path):
    from types import SimpleNamespace
    from reporting import atomic_json, collect_results, timing_summary
    config = SimpleNamespace(transport='urma', mode='diagnose', batches=(2,))
    atomic_json(tmp_path / 'case/completed_timings.json', dict(batch=2, correctness='PASS', timing=dict(
        native_chain=timing_summary([100]), prototype_with_update=timing_summary([80]))))
    rows = collect_results(tmp_path, config, 1)
    assert rows[0]['serial_mean_us'] == 100 and rows[0]['fused_mean_us'] == 80
    assert rows[0]['timing'] == 'COMPLETE' and not rows[0]['accepted']
    assert rows[0]['execution'] == 'INCOMPLETE'


@pytest.mark.parametrize('graph_batch', [False, True])
def test_diagnose_csv_joins_stage_and_transfer_evidence_by_case(tmp_path, monkeypatch, graph_batch):
    from execution import dispatch
    from reporting import atomic_json
    from workloads import configuration

    def launch(job, directory):
        for batch in job.batches:
            case = directory / f'case{batch}'
            atomic_json(case / 'report.json', dict(
                batch=batch, correctness='PASS', native_misses_per_request=[7] * batch,
                fused_read_counts=[dict(prefetch=3 * batch, final=6 * batch), dict(prefetch=5 * batch, final=4 * batch)],
                **({} if graph_batch else dict(fused_attribution=f'../trace{batch}/attribution.json')),
                timing=dict(native_chain=timing_summary([110]), native_get_only=timing_summary([40]),
                            prototype_with_update=timing_summary([87]))))
            serial_source = (case / 'serial/diagnose/round-1/native_attribution.json' if graph_batch
                             else case / 'native_attribution.json')
            fused_source = (case / 'fused/diagnose/round-1/full_attribution.json' if graph_batch
                            else directory / f'trace{batch}/attribution.json')
            atomic_json(serial_source, dict(samples=[dict(
                round=i, us=120, accounting=dict(stages_us=dict(li=20 * batch, lidu_prepare=10, fetch=45),
                                                inter_kernel_gap_us=2)) for i in range(job.iterations)]))
            atomic_json(fused_source, [dict(
                round=i, prefetch=0, final=0, algorithm=dict(totals={}, requests=[]), us=95,
                accounting=dict(stages_us=dict(init=4, li=15 * batch, resolve_update_fetch=30)))
                for i in range(job.iterations)])
        atomic_json(directory / 'worker-result.json', dict(exit_code=0))
        return 0

    monkeypatch.setattr('execution.launch', launch)
    config = configuration(['diagnose', '--transport', 'ubmem', '--kvaf-server', 'test',
                            '--batches', '1,2', '--iterations', '2', '--stages', 'fetch', '--output', str(tmp_path)])
    assert dispatch(config) == 0
    with next(tmp_path.glob('*/results.csv')).open(newline='') as stream:
        first, second = list(csv.DictReader(stream))
    assert first['serial_p50_us'] == '110.0'
    assert first['serial_profiled_total_p50_us'] == '120.0'
    assert first['serial_profiled_li_p50_us'] == '20.0'
    assert first['serial_profiled_lidu_prepare_p50_us'] == '10.0'
    assert first['serial_profiled_get_p50_us'] == '45.0'
    assert first['serial_get_only_p50_us'] == '40.0'
    assert first['fused_profiled_li_p50_us'] == '15.0'
    assert second['fused_profiled_li_p50_us'] == '30.0'
    assert first['fused_prefetch_entries_min'] == '3'
    assert first['fused_final_entries_max'] == '6'
    assert first['fused_profiled_transfer_GBps_p50'] == ''
    assert first['fused_profiled_transfer_span_samples'] == '0'
    assert second['serial_read_entries'] == '14'
    assert second['fused_total_bytes_mean'] == str(18.0 * 656)
    assert second['fused_read_count_samples'] == '2'
