# SPDX-License-Identifier: Apache-2.0
"""Public commands route workloads and preserve partial worker evidence."""
import importlib.util
import json
from pathlib import Path

import execution
from reporting import atomic_json, timing_summary

SOURCE = Path(__file__).resolve().parents[1] / 'test_lifetch.py'
spec = importlib.util.spec_from_file_location('unified_lifetch_test', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


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
    assert module.main(['bench', '--transport', 'ubmem', '--batches', '2,16', '--seed', '19',
                        '--kvaf-server', 'host:1', '--output', str(tmp_path)]) == 0
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
    assert module.main(['check', '--suite', 'init', '--kvaf-server', 'host:1', '--output', str(tmp_path)]) == 0
    assert calls == ['init']


def test_crashed_check_stops_remaining_device_cases(monkeypatch, tmp_path):
    calls = []
    def worker(config, directory):
        directory.mkdir(parents=True)
        calls.append(config)
        return -15
    monkeypatch.setattr(execution, 'launch', worker)
    assert module.main(['check', '--suite', 'basic', '--kvaf-server', 'host:1', '--output', str(tmp_path)]) == 1
    assert len(calls) == 1
    rows = json.loads(next(tmp_path.glob('*/summary.json')).read_text())['cases']
    assert [row['execution'] for row in rows] == ['FAILED', *['NOT_RUN'] * 4]
