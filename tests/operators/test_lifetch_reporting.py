"""CPU regressions for the unified LIFetch reporting module."""
import json

import execution as execution
import pytest
from reporting import CorrectnessRecorder, atomic_json, comparison_row, save_summary, timing_summary
from workloads import configuration


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


def test_late_assertion_overrides_prior_pass_but_preserves_timings(tmp_path):
    from types import SimpleNamespace

    from reporting import atomic_json, collect_results, record_case_failure
    atomic_json(tmp_path / 'completed_timings.json', dict(batch=4, correctness='PASS', timing=dict(
        native_chain=timing_summary([100]), prototype_with_update=timing_summary([80]))))
    record_case_failure(tmp_path, AssertionError('TopK changed'))
    rows = collect_results(tmp_path, SimpleNamespace(transport='urma', mode='diagnose', batches=(4,)), 1)
    assert rows[0]['execution'] == 'FAILED' and rows[0]['correctness'] == 'FAIL'
    assert rows[0]['fused_mean_us'] == 80 and not rows[0]['accepted']


def test_bench_routes_requested_workload_and_saves_table(monkeypatch, tmp_path):
    calls = []
    def worker(config, directory):
        calls.append(config)
        for batch in config.batches:
            atomic_json(directory / f'case{batch}/report.json', dict(batch=batch, correctness='PASS', timing=dict(
                native_chain=timing_summary([100, 110]), prototype_with_update=timing_summary([80, 90]))))
        atomic_json(directory / 'worker-result.json', dict(exit_code=0))
        return 0
    monkeypatch.setattr(execution, 'launch', worker)
    assert execution.dispatch(configuration(['bench', '--transport', 'ubmem', '--batches', '2,16', '--seed', '19',
                        '--kvaf-server', 'host:1', '--output', str(tmp_path)])) == 0
    assert calls[0].batches == (2, 16) and calls[0].seed == 19
    result = json.loads(next(tmp_path.glob('*/summary.json')).read_text())['cases'][0]
    assert result['serial_mean_us'] == 105 and result['fused_mean_us'] == 85


def test_check_init_uses_only_init_job(monkeypatch, tmp_path):
    calls = []
    def worker(config, directory):
        calls.append(config.operation)
        atomic_json(directory / 'checks.json', [dict(case='init', correctness='PASS')])
        atomic_json(directory / 'worker-result.json', dict(exit_code=0))
        return 0
    monkeypatch.setattr(execution, 'launch', worker)
    assert execution.dispatch(configuration(['check', '--suite', 'init', '--kvaf-server', 'host:1', '--output', str(tmp_path)])) == 0
    assert calls == ['init']


def test_crashed_check_stops_remaining_device_cases(monkeypatch, tmp_path):
    calls = []
    def worker(config, directory):
        directory.mkdir(parents=True)
        calls.append(config)
        return -15
    monkeypatch.setattr(execution, 'launch', worker)
    assert execution.dispatch(configuration(['check', '--suite', 'basic', '--kvaf-server', 'host:1', '--output', str(tmp_path)])) == 1
    assert len(calls) == 1
    rows = json.loads(next(tmp_path.glob('*/summary.json')).read_text())['cases']
    assert [row['execution'] for row in rows] == ['FAILED', *['NOT_RUN'] * 4]


def test_source_layout_and_entry_import_without_device_modules():
    import ast
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    source = root / 'tests/operators/lifetch'
    expected = {'workloads', 'execution', 'correctness', 'performance', 'diagnostics', 'reporting'}
    assert {path.stem for path in (source / 'support').glob('*.py')} == expected
    assert {path.name for path in source.rglob('*.py')} == {'test_lifetch.py', *{f'{name}.py' for name in expected}}
    # A removed helper cannot be silently supplied by a developer's PYTHONPATH.
    for path in source.rglob('*.py'):
        tree = ast.parse(path.read_text())
        assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                       and node.func.id in {'exec', 'eval'} for node in ast.walk(tree))
    command = [sys.executable, '-c',
               "import runpy,sys; sys.path.insert(0,sys.argv[1]); "
               "runpy.run_path(sys.argv[2],run_name='lifetch_import_test'); import execution; "
               "assert not {'acl','torch','torch_npu','kvaf'} & set(sys.modules)",
               str(root / 'python'), str(source / 'test_lifetch.py')]
    subprocess.run(command, check=True, capture_output=True, text=True)
    help_result = subprocess.run([sys.executable, str(source / 'test_lifetch.py'), '--help'],
                                 check=True, capture_output=True, text=True)
    assert 'convert-trace' in help_result.stdout and 'diagnose' in help_result.stdout
