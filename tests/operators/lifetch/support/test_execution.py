"""Configuration adapters cannot silently change workloads or enable tracing."""
from execution import runner_options
from workloads import configuration


def test_benchmark_options_preserve_workload_and_do_not_profile():
    args = configuration(['bench', '--batches', '1,2,4,8,16', '--seed', '93',
                          '--miss-percent', '15', '--iterations', '7'])
    worker = runner_options(args)
    assert worker.batches == '1,2,4,8,16'
    assert worker.seed == 93 and worker.miss_percent == 15
    assert worker.iterations == 7 and worker.trace_rounds == 0
    assert worker.streaming and worker.streaming_fetch and worker.fused_final
    assert worker.prepare_update_metadata and worker.q_workers == 2


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
    from runner import supervise
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
        assert worker.fetch_only_compare is False  # Never bypass full-graph timing.
