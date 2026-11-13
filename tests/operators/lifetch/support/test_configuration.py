import pytest

from workloads import CHECK_SUITES, configuration


def test_defaults_and_check_has_no_geometry_options():
    check = configuration(['check'])
    assert check.transport == 'urma' and check.device == 0
    assert check.suite == CHECK_SUITES[:-1]
    bench = configuration(['bench'])
    assert bench.seed == 43 and bench.batches == (1, 4, 8, 16)
    assert bench.queries == 4
    assert bench.fetch_jettys_per_aiv == 2
    with pytest.raises(SystemExit):
        configuration(['check', '--batches', '16'])


def test_real_preserves_source_selection_and_rejects_even_explicit_default_geometry():
    args = ['bench', '--input', 'real', '--trace-root', '/trace', '--group', '10', '--round-range', '51:51']
    parsed = configuration(args)
    assert parsed.round_range == (51, 51) and parsed.batches is None
    for key, value in [('seed', '43'), ('queries', '4'), ('batches', '16'), ('history', '131072')]:
        with pytest.raises(SystemExit):
            configuration(args + ['--' + key, value])


@pytest.mark.parametrize('args', [
    ['check', '--suite', 'real'], ['check', '--suite', 'all,basic'],
    ['bench', '--queries', '2'], ['bench', '--batches', '17'],
    ['bench', '--batches', '1,1'], ['bench', '--trace-root', '/trace'],
    ['bench', '--input', 'real'], ['bench', '--seed', '-1'],
    ['diagnose', '--level', 'stage'], ['diagnose', '--stages', 'unknown'],
    ['bench', '--fetch-jettys-per-aiv', '3'],
    ['bench', '--transport', 'ubmem', '--fetch-jettys-per-aiv', '1'],
    ['diagnose', '--input', 'real', '--fetch-jettys-per-aiv', '1'],
    ['diagnose', '--stages', 'all,fetch'],
    ['diagnose', '--algorithm-stats'], ['diagnose', '--fetch-bandwidth'],
])
def test_invalid_or_removed_interfaces_fail_early(args):
    with pytest.raises(SystemExit):
        configuration(args)


def test_diagnose_reuses_bench_config_and_has_only_two_extra_selectors():
    args = configuration(['diagnose', '--level', '3', '--stages', 'update,fetch'])
    assert args.stages == ('update', 'fetch')
    assert args.iterations == 20 and args.seed == 43
    assert 'algorithm_stats' not in vars(args)


@pytest.mark.parametrize('mode', ['bench', 'diagnose'])
def test_synthetic_urma_accepts_single_jetty(mode):
    args = configuration([mode, '--batches', '16', '--miss-percent', '12.5', '--fetch-jettys-per-aiv', '1'])
    assert args.fetch_jettys_per_aiv == 1


@pytest.mark.parametrize('flag', ['--early-fetch', '--avoid-prefetch-slots', '--legacy-fetch-layout',
                                   '--parallel-fetch-finalize', '--optimized-resolve', '--trace-rounds'])
def test_old_pipeline_flags_are_not_public(flag):
    with pytest.raises(SystemExit):
        configuration(['bench', flag])
