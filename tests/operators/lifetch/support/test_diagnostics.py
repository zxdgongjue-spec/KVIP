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


def test_byte_backed_topk_is_decoded_as_int32(monkeypatch):
    import numpy as np
    import diagnostics

    class Tensor:
        def __init__(self, data):
            self.data = data
        def cpu(self):
            return self
        def numpy(self):
            return self.data

    selected = np.arange(4 * 4 * 2048, dtype=np.int32).reshape(4, 4, 2048)
    rotating = SimpleNamespace(runtime_shape=SimpleNamespace(batch=4, resident=16384, tokenBytes=656),
        rounds=8, arguments=[None] * 9 + [Tensor(selected.view(np.uint8))],
        mapping=Tensor(np.zeros((4, 1))), rows=Tensor(np.zeros((4, 256), dtype=np.uint32)),
        final_posts=Tensor(np.zeros((4, 4, 6))))
    monkeypatch.setattr(diagnostics, 'algorithm_counts', lambda mapping, tokens, *args, **kw: tokens)
    assert np.array_equal(diagnostics.algorithm_after(rotating, {}), selected)


def test_level_one_terminal_is_tables_not_raw_dump(tmp_path, capsys):
    atomic_json(tmp_path / 'full_attribution.json', dict(samples=[dict(round=0,
        prefetch=0, final=0, algorithm=dict(totals={}, requests=[]),
        accounting=dict(stages_us=dict(li=12)))]))
    summarize(tmp_path, SimpleNamespace(iterations=1, level=1, stages=('li',), transport='urma'))
    output = capsys.readouterr().out
    assert 'Stage distribution' in output and 'Per-request counts' in output
    assert 'Effective bandwidth' in output
    assert 'ACCOUNTING repeat=' not in output
    assert output.index('ALGORITHM / KV TRAFFIC') < output.index('TIME BREAKDOWN')
    assert 'N/A: ' not in output and 'INCOMPLETE: ' not in output
    assert (tmp_path / 'level1_summary.log').read_text().strip() == output.strip()


def test_level_two_tables_keep_replay_core_and_query_identity():
    from diagnostics import level_two_tables
    request = dict(batch=0, producer=[dict(resolve=20, output=10, pack=5, publish=5)],
        update=[10, 30], update_wait=[5, 10], li=[],
        core_timeline_raw=dict(batch=1, resolver_cores=3, workers_per_query=2, clock_mhz=10))
    core = dict(block=3, role='fetch Q1 lane0', start_us=1., end_us=9., duration_us=8.,
        stages=[dict(name='safe.KVAF_Get', start_us=2., end_us=8., duration_us=6.,
                     start_cycles=20, end_cycles=80)],
        KVAF_counters=dict(safe_post_cycles=10, safe_CQ_cycles=20,
                          conflict_post_cycles=0, conflict_CQ_cycles=0))
    sample = dict(repeat=2, requests=[request], timeline=dict(cores=[core], last_block=3,
        observed_span_us=9., clock_mhz=10), algorithm=dict(requests=[dict(request=0,
        windows=[dict(window=1, posted=2, bytes=1312, unused=1, cumulative_posted=2, cumulative_unused=1)],
        queries=[dict(query=1, topk=2048, unique=2048, cross_q_duplicate=0,
                      new_unique=2048, initial_local=2046, final_posted=1)])]))
    result = dict(level=2, transport='urma', trace='COMPLETE', missing=[], unavailable=[],
        captures=[dict(source='test', stage_summary={}, samples=[sample])])
    sample['final_get_bandwidth'] = dict(groups=[dict(block=3, query=1, lane=0, entries=1, bytes=656)])
    output = level_two_tables(result)
    assert 'Per-Fetch-block effective bandwidth' in output
    assert '0.109' in output  # 656 bytes / 6 us, decimal GB/s.
    for value in ('samples=1', 'fetch Q1 lane0', 'Window prefetch / unused',
                  'Per-Q demand / repair', 'Per-Q resolver / update', '1.000', '2.000'):
        assert value in output
    assert 'URMA group counters' not in output
    assert 'Core work / wait intervals' not in output
    assert 'LI TopK classification' not in output
    assert output.index('ALGORITHM / KV TRAFFIC') < output.index('TIME BREAKDOWN')
    result['captures'][0]['samples'] = [sample, sample, sample]
    aggregated = level_two_tables(result)
    for title in ('Active cores', 'Per-Fetch-block effective bandwidth', 'Per-Q resolver / update'):
        assert aggregated.count(title) == 1
    assert '3/3' in aggregated
    result['level'] = 3
    assert 'URMA group counters' in level_two_tables(result)
    assert 'Core work / wait intervals' in level_two_tables(result)
    result['transport'] = 'ubmem'
    assert 'URMA group counters' not in level_two_tables(result)


def test_level_three_prefetch_counts_and_time_are_separated():
    from diagnostics import level_two_tables
    event = dict(event=2, core=7, epoch=1, bitmap_bytes=4096, candidate_count=80,
                 posted_count=32, scan_cycles=100, candidate_cycles=200, post_cycles=300)
    request = dict(batch=0, core_timeline_raw=dict(clock_mhz=100), prefetch_services=[event])
    result = dict(level=3, transport='ubmem', trace='COMPLETE', captures=[dict(source='trace', stage_summary={},
        samples=[dict(repeat=0, requests=[request], algorithm=dict(requests=[]))])])
    text = level_two_tables(result)
    algorithm, timing = text.split('TIME BREAKDOWN')
    assert 'Prefetch service work' in algorithm
    assert 'Prefetch service steps' not in algorithm
    assert 'Prefetch service steps' in timing
    assert 'submit/copy' in timing and '3.000' in timing and 'Cycles' in timing
    result['level'] = 2
    assert 'Prefetch service' not in level_two_tables(result)


def test_level_two_repeats_are_aggregated_without_mutating_raw_samples():
    from copy import deepcopy
    from diagnostics import level_two_tables, aggregate_diagnostic_rows
    samples = [dict(repeat=i, requests=[], algorithm=dict(requests=[dict(request=0,
        selected_unique=100, initial_cold=10, prefetch=value, final=3,
        KV_read_bytes=656 * (value + 3), unused_prefetch=1, windows=[], queries=[])]))
        for i, value in enumerate((4, 6, 8))]
    result = dict(level=2, transport='urma', trace='COMPLETE', captures=[dict(
        source='case', stage_summary={}, samples=samples)])
    original = deepcopy(result)
    output = level_two_tables(result)
    assert output.count('Demand and actual reads') == 1
    assert output.count('Window prefetch / unused') == 1
    assert output.count('Per-Q demand / repair') == 1
    assert '6.000 [4.000, 8.000]' in output
    assert 'repeat=' not in output
    assert result == original
    result['level'] = 3
    assert level_two_tables(result).count('Demand and actual reads') == 3
    assert aggregate_diagnostic_rows([[3, 'fetch', 2., '*'], [3, 'fetch', 4., '']], 2) == [
        [3, 'fetch', '3.000 [2.000, 4.000]', '1/2']]
