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
    assert 'FAIL | 500.000 | 350.000 | 30.00%' in (tmp_path / 'summary.log').read_text()


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
    config = SimpleNamespace(transport='urma', mode='bench', batches=(1, 2))
    atomic_json(tmp_path / 'one/report.json', dict(batch=1, correctness='PASS', timing=dict(
        native_chain=timing_summary([10]), prototype_with_update=timing_summary([8]))))
    rows = collect_results(tmp_path, config, 0)
    assert len(rows) == 2
    assert rows[0]['accepted'] and not rows[1]['accepted']
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
