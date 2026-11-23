"""Configuration adapters cannot silently change workloads or enable tracing."""
import pytest

from execution import runner_options
from workloads import configuration


def test_benchmark_options_preserve_workload_and_do_not_profile():
    args = configuration(['bench', '--batches', '1,2,4,8,16', '--seed', '93',
                          '--miss-percent', '15', '--iterations', '7', '--fetch-jettys-per-aiv', '1'])
    worker = runner_options(args)
    assert worker.batches == '1,2,4,8,16'
    assert worker.seed == 93 and worker.miss_percent == 15
    assert worker.iterations == 7 and worker.trace_rounds == 0
    assert worker.q_workers == 2 and worker.fetch_jettys_per_aiv == 1


def test_diagnostic_repeats_follow_iterations():
    args = configuration(['diagnose', '--iterations', '7', '--stages', 'update'])
    worker = runner_options(args)
    assert worker.trace_rounds == 7
    assert worker.stages == ('update',)


def test_real_never_falls_back_to_synthetic():
    import pytest
    args = configuration(['bench', '--input', 'real', '--trace-root', '/trace',
                          '--group', '10', '--round-range', '51:52'])
    with pytest.raises(ValueError, match='continuous replay'):
        runner_options(args)


def test_worker_dependency_environment_is_explicit_and_local(tmp_path):
    import os
    from execution import ROOT, worker_environment
    package = tmp_path / 'build-npu/python/kvaf/__init__.py'
    package.parent.mkdir(parents=True)
    package.write_text('')
    inherited = {'PYTHONPATH': '/previous/kvaf', 'ASCEND_HOME_PATH': '/cann'}
    environment = worker_environment(tmp_path, inherited)
    assert environment['PYTHONPATH'].split(os.pathsep)[:2] == [
        str(ROOT / 'python'), str(tmp_path / 'build-npu/python')]
    assert environment['ASCEND_HOME_PATH'] == '/cann'
    assert inherited['PYTHONPATH'] == '/previous/kvaf'


def test_missing_selected_kvaf_does_not_fall_back(tmp_path):
    import pytest
    from execution import worker_environment
    with pytest.raises(FileNotFoundError, match='Selected KVAF'):
        worker_environment(tmp_path, {'PYTHONPATH': '/other/working/kvaf'})


def test_supervisor_passes_selected_environment_to_real_child(tmp_path):
    import json
    import os
    import sys
    from execution import supervise
    observed = tmp_path / 'observed.json'
    marker = tmp_path / 'worker-result.json'
    program = (
        'import json, os, pathlib, sys; '
        'pathlib.Path(sys.argv[1]).write_text(json.dumps(dict(selected=os.environ["KVIP_TEST_SELECTED"]))); '
        'pathlib.Path(sys.argv[2]).write_text(json.dumps(dict(exit_code=0)))'
    )
    environment = dict(os.environ, KVIP_TEST_SELECTED='explicit-root')
    code = supervise([sys.executable, '-c', program, str(observed), str(marker)], tmp_path,
                     environment=environment)
    assert code == 0
    assert json.loads(observed.read_text()) == {'selected': 'explicit-root'}


def test_fetch_only_is_automatic_only_for_level_three_fetch():
    for arguments, expected in [
        (['bench'], False),
        (['diagnose', '--level', '2'], False),
        (['diagnose', '--level', '3', '--stages', 'update'], False),
        (['diagnose', '--level', '3', '--stages', 'fetch'], True),
        (['diagnose', '--level', '3'], True),
    ]:
        worker = runner_options(configuration(arguments))
        assert worker.diagnostic_fetch_only is expected


def test_bench_matrix_rejects_invalid_percentages():
    import pytest
    for value in ('nan', 'inf', '3,nan', '3,101', '-1,3', '3,3.0', '3,'):
        with pytest.raises(SystemExit):
            configuration(['bench', f'--miss-percent={value}'])


def test_matrix_dispatch_collects_twenty_pairs_without_npu(tmp_path, monkeypatch):
    import json
    import execution
    from reporting import atomic_json, timing_summary

    def launch(job, directory):
        directory.mkdir(parents=True)
        assert job.miss_percent == (3, 5, 10, 15)
        for miss in job.miss_percent:
            for batch in job.batches:
                atomic_json(directory / f'miss{miss}' / f'bs{batch}' / 'report.json', dict(
                    batch=batch, miss_percent=miss, correctness='PASS', timing=dict(
                        native_chain=timing_summary([100, 110]),
                        prototype_with_update=timing_summary([80, 90]))))
        atomic_json(directory / 'worker-result.json', dict(exit_code=0))
        return 0

    monkeypatch.setattr(execution, 'launch', launch)
    for transport in ('urma', 'ubmem'):
        output = tmp_path / transport
        config = configuration(['bench', '--transport', transport, '--batches', '1,2,4,8,16',
                                '--miss-percent', '3,5,10,15', '--kvaf-server', 'test:1',
                                '--output', str(output)])
        assert execution.dispatch(config) == 0
        summary, = output.rglob('summary.json')
        rows = json.loads(summary.read_text())['cases']
        assert len(rows) == 20
        assert [(row['miss_percent'], row['batch']) for row in rows] == [
            (miss, bs) for miss in (3, 5, 10, 15) for bs in (1, 2, 4, 8, 16)]
        assert {(row['batch'], row['miss_percent']) for row in rows} == {
            (bs, miss) for bs in (1, 2, 4, 8, 16) for miss in (3, 5, 10, 15)}
        assert all(row['transport'] == transport and row['accepted'] for row in rows)
        assert all(row['serial_mean_us'] == 105 and row['fused_mean_us'] == 85 for row in rows)
        assert 'Invocation distribution (us)' in summary.with_name('summary.log').read_text()


def test_nested_miss_progress_reaches_supervised_worker(tmp_path):
    from execution import progress
    (tmp_path / 'config.json').write_text('{}')
    child = tmp_path / 'miss15'
    child.mkdir()
    progress(child, 'BS=8: paired benchmark')
    assert (child / 'stage.log').read_text().strip() == 'BS=8: paired benchmark'
    assert (tmp_path / 'stage.log').read_text().strip() == 'miss15: BS=8: paired benchmark'


@pytest.mark.parametrize('mode', ['bench', 'diagnose'])
def test_synthetic_urma_worker_reuses_the_performance_pool(tmp_path, monkeypatch, mode):
    import json
    import sys
    from types import SimpleNamespace
    import execution
    from reporting import atomic_json

    build = tmp_path / 'build-npu'
    monkeypatch.setitem(sys.modules, 'kvaf', SimpleNamespace(__file__=str(build / 'python/kvaf/__init__.py')))
    workers = []

    def run_worker(options, directory, *, shared_pool):
        pool = shared_pool
        assert isinstance(pool, execution.BenchPool)
        pool.workers = 40
        pool.borrows += 1
        assert options.batches == [16]
        assert options.iterations == 2 and options.mode == mode
        workers.append((pool, options.miss_percent))
        return 0

    monkeypatch.setattr(execution, 'run_synthetic_batch', run_worker)
    miss = '5,12.5' if mode == 'bench' else '12.5'
    config = configuration([mode, '--transport', 'urma', '--kvaf-root', str(tmp_path),
                            '--batches', '16', '--miss-percent', miss, '--iterations', '2'])
    path = tmp_path / 'run/config.json'
    atomic_json(path, dict(vars(config), kvaf_root=str(config.kvaf_root), output=str(config.output)))
    assert execution.run_config(path) == 0
    stats = json.loads(path.with_name('pool.json').read_text())
    assert stats['runtime_workers'] == 40 and stats['cases'] == len(workers)
    assert [miss for _, miss in workers] == ([5, 12.5] if mode == 'bench' else [12.5])
    assert all(pool is workers[0][0] for pool, _ in workers)


@pytest.mark.parametrize('show_details', [False, True])
@pytest.mark.parametrize('exit_code', [0, 1])
def test_supervisor_details_saved_and_failures_visible(tmp_path, capsys, show_details, exit_code):
    import sys
    from execution import supervise

    (tmp_path / 'cases.log').write_text('DETAIL_COUNTERS prefetch=128\n')
    program = (
        'import json, pathlib, sys; '
        'pathlib.Path(sys.argv[1]).write_text(json.dumps({"exit_code": int(sys.argv[2])})); '
        'print("FAILURE_REASON" if int(sys.argv[2]) else "worker detail"); '
        'sys.exit(int(sys.argv[2]))'
    )
    code = supervise([sys.executable, '-c', program,
                      str(tmp_path / 'worker-result.json'), str(exit_code)],
                     tmp_path, show_case_details=show_details)
    output = capsys.readouterr().out
    assert code == exit_code
    assert ('DETAIL_COUNTERS' in output) == show_details
    assert 'DETAIL_COUNTERS' in (tmp_path / 'summary.log').read_text()
    if exit_code:
        assert 'FAILURE_REASON' in output
