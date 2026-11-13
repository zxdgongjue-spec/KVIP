"""Diagnostic summaries never combine different replays into a timeline."""
import json
from types import SimpleNamespace

from diagnostics import summarize
from reporting import atomic_json


def test_stage_distributions_keep_raw_repeats_and_selection(tmp_path):
    samples = [dict(round=i, prefetch=10+i, final=3, algorithm=dict(totals={}, requests=[]), accounting=dict(
        stages_us=dict(init=2, li=value, resolve_update=7))) for i, value in enumerate((10, 30))]
    atomic_json(tmp_path / 'case/full_attribution.json', dict(samples=samples))
    config = SimpleNamespace(iterations=2, level=1, stages=('li',), transport='urma')
    report = summarize(tmp_path, config)
    assert report['trace'] == 'COMPLETE'
    capture = report['captures'][0]
    assert capture['stage_summary'] == {'li': dict(mean_us=20., p50_us=20., min_us=10., max_us=30., samples=2)}
    assert [row['repeat'] for row in capture['samples']] == [0, 1]
    assert json.loads((tmp_path / 'diagnostic.json').read_text()) == report


def test_missing_timestamp_and_repeat_are_incomplete_not_zero(tmp_path):
    atomic_json(tmp_path / 'case/attribution.json', [dict(round=0, accounting=dict(error='missing Get'))])
    config = SimpleNamespace(iterations=2, level=2, stages=('fetch',), transport='ubmem')
    report = summarize(tmp_path, config)
    assert report['trace'] == 'INCOMPLETE'
    assert report['captures'][0]['stage_summary'] == {}
    assert any('missing Get' in message for message in report['missing'])
    assert any('expected 2' in message for message in report['missing'])


def test_fetch_only_requires_both_paths_and_all_queries(tmp_path):
    config = SimpleNamespace(iterations=1, level=3, stages=('fetch',), transport='urma')
    missing = summarize(tmp_path, config)
    assert 'Fetch-only comparison results' in missing['missing']
    rows = [dict(query=query, version=version, aiv=8 if query == 'Q4_all' else 2,
                 jettys_per_aiv=2, count=0, bytes=0, mean_us=12., p50_us=11.,
                 p99_us=13., effective_GBps=None, samples_us=[11., 13.])
            for query in ('Q1', 'Q2', 'Q3', 'Q4', 'Q4_all')
            for version in ('native', 'fused')]
    atomic_json(tmp_path / 'case/fetch_only.json', dict(status='PASS', rows=rows))
    complete = summarize(tmp_path, config)
    assert not any('Fetch-only' in item for item in complete['missing'])
    assert len(complete['fetch_only'][0]['rows']) == 10
    assert 'FETCH-ONLY Q4_all fused' in (tmp_path / 'diagnostic_summary.log').read_text()
    atomic_json(tmp_path / 'case/fetch_only.json', dict(status='PASS', rows=rows[:-1]))
    incomplete = summarize(tmp_path, config)
    assert any('incomplete Fetch-only' in item for item in incomplete['missing'])


def test_ubmem_fetch_only_unsupported_is_explicit(tmp_path):
    config = SimpleNamespace(iterations=1, level=3, stages=('fetch',), transport='ubmem')
    result = summarize(tmp_path, config)
    assert any('Fetch-only: UNSUPPORTED' in item for item in result['unavailable'])
    assert 'Fetch-only comparison results' not in result['missing']


def test_real_rounds_keep_separate_repeat_distributions(tmp_path):
    for source, durations in ((51, (10, 20)), (52, (30, 50))):
        samples = [dict(round=repeat, source_round=source, scheduler_round=source + 100,
                        identities=[[2, 'request', 1]], correctness='PASS',
                        algorithm=dict(totals=dict(prefetch=source), requests=[]),
                        accounting=dict(stages_us=dict(li=duration)))
                   for repeat, duration in enumerate(durations)]
        atomic_json(tmp_path / f'round-{source}/full_attribution.json', dict(samples=samples))
    config = SimpleNamespace(iterations=2, level=1, stages=('li',), transport='urma')
    result = summarize(tmp_path, config)
    assert result['trace'] == 'COMPLETE'
    assert [capture['stage_summary']['li']['mean_us'] for capture in result['captures']] == [15, 40]
    assert [capture['samples'][0]['source_round'] for capture in result['captures']] == [51, 52]
    assert 'source_round=51' in (tmp_path / 'diagnostic_summary.log').read_text()
    assert 'source_round=52' in (tmp_path / 'diagnostic_summary.log').read_text()


def test_level_one_requests_intervals_and_zero_bandwidth(tmp_path):
    for transport in ('urma', 'ubmem'):
        root = tmp_path / transport
        sample = dict(round=0, prefetch=0, final=0, algorithm=dict(
            totals=dict(prefetch=0, final=0), requests=[dict(request=0,
                initial_hot_valid=16384, selected_unique=2048, initial_cold=0,
                prefetch=0, final=0, KV_read_bytes=0)]), accounting=dict(
            stages_us=dict(li=10), intervals_us=dict(li=[[100, 110]]),
            inter_kernel_gap_us=2, boundary_residual_us=3))
        atomic_json(root / 'full_attribution.json', dict(samples=[sample]))
        report = summarize(root, SimpleNamespace(iterations=1, level=1,
                            stages=('li', 'fetch'), transport=transport))
        assert report['trace'] == 'COMPLETE'
        log = (root / 'diagnostic_summary.log').read_text()
        assert 'request=0 initial_hot_valid=16384 selected_unique=2048 initial_cold=0' in log
        assert 'start_us=0.000 end_us=10.000 duration_us=10.000' in log
        assert 'GB/s=N/A' in log
        assert 'window candidates' not in log
        assert not (root / 'timeline.csv').exists()


def test_level_one_missing_nonzero_transfer_timestamps_fails(tmp_path):
    atomic_json(tmp_path / 'full_attribution.json', dict(samples=[dict(
        round=0, prefetch=10, final=5, algorithm=dict(totals={}, requests=[]),
        accounting=dict(stages_us=dict(resolve_update=20)))]))
    result = summarize(tmp_path, SimpleNamespace(iterations=1, level=1,
                       stages=('fetch',), transport='ubmem'))
    assert result['trace'] == 'INCOMPLETE'
    assert any('final_get_bandwidth unavailable' in value for value in result['missing'])
