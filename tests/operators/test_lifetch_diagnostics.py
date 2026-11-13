"""CPU regressions for the unified LIFetch diagnostics module."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import diagnostics as module
import numpy as np
import pytest
from diagnostics import (
    account_native,
    algorithm_counts,
    analyze_prefetch_pressure,
    capture_rows,
    collect_replay,
    split_native_jetty_descriptors,
    summarize,
    summarize_mismatches,
)
from reporting import atomic_json
from test_lifetch_workloads import capture_file
from workloads import load_score_capture


def algorithm_counts_fixture():
    mapping = np.full((1, 4096), -1, dtype=np.int32)
    # Physical order differs from admission order across two windows.
    mapping[0, [0, 3000, 3001]] = [2050, 2048, 2049]
    selected = np.broadcast_to(np.arange(2048), (1, 4, 2048)).copy()
    selected[0, 1, 0] = 3000
    rows = np.zeros((1, 256), dtype=np.uint32)
    rows[0, 9], rows[0, 247] = 3, 1
    rows[0, 16:18] = [1, 2]
    posts = np.zeros((1, 4, 8), dtype=np.int32)
    posts[0, 1, 0] = 1
    return mapping, selected, rows, posts


def test_window_admission_order_query_union_and_exact_unused():
    result = algorithm_counts(*algorithm_counts_fixture(), 2048, 2, hot_valid=np.ones((1, 2048)), token_bytes=2624)
    row = result['requests'][0]
    assert row['selected_unique'] == 2049 and row['initial_cold'] == 1
    assert [item['unused'] for item in row['windows']] == [0, 1]
    assert [item['cumulative_unused'] for item in row['windows']] == [0, 1]
    assert [item['new_unique'] for item in row['queries']] == [2048, 1, 0, 0]
    assert [item['final_posted'] for item in row['queries']] == [0, 1, 0, 0]
    assert row['KV_read_bytes'] == 4 * 2624
    assert row['useful'] is None and row['redundant'] is None


@pytest.mark.parametrize('corrupt', ['ordinal', 'window', 'final'])
def test_inconsistent_identity_or_counts_are_not_silently_summarized(corrupt):
    mapping, selected, rows, posts = algorithm_counts_fixture()
    if corrupt == 'ordinal':
        mapping[0, 3000] = 2050
    elif corrupt == 'window':
        rows[0, 16] += 1
    else:
        posts[0, 1, 0] += 1
    with pytest.raises(ValueError, match='match'):
        algorithm_counts(mapping, selected, rows, posts, 2048, 2)


def test_missing_initial_state_is_not_assumed_all_hot():
    result = algorithm_counts(*algorithm_counts_fixture(), 2048, 2)
    assert result['totals']['initial_cold'] is None
    assert result['totals']['cold_miss_rate'] is None
    assert result['requests'][0]['queries'][0]['initial_local'] is None


def test_abc_is_local_and_not_counted_as_initial_cold():
    result = algorithm_counts(*algorithm_counts_fixture(), 2048, 2, hot_valid=np.ones((1, 2048)),
                              tail_bounds=np.array([[3000, 3002]]))
    assert result['requests'][0]['initial_hot_valid'] == 2048
    assert result['requests'][0]['initial_cold'] == 0

    assert result['requests'][0]['queries'][1]['initial_local'] == 1


def scored_capture(directory):
    path = capture_file(directory, real=False)
    capture = load_score_capture(path, require_real=False)
    a = capture.arrays
    a.update({
        'post.window_posted': np.array([[1, 1]], dtype=np.int32),
        'post.window_processed_ends': np.array([[[2048] * 4, [4096] * 4]], dtype=np.int32),
        'post.prefetch_records': np.array([[0, 0, 4095], [0, 3000, 1095]], dtype=np.int64),
        'post.prefetch_ordinals': np.array([1, 0], dtype=np.int32),
        'post.final_records': np.array([[0, 4095, 0, 0]], dtype=np.int64),
        'post.threshold_records': np.zeros((1, 4, 2, 4), dtype=np.int32),
    })
    a['post.threshold_records'][:, :, 0] = [2048, 2048, 0xBF80, 1]
    a['post.threshold_records'][:, :, 1] = [4096, 4096, 0xBF80, 4]
    return capture


def test_prefetch_admission_order_and_ties(tmp_path):
    capture = scored_capture(tmp_path)
    row, selected = capture_rows(capture)[0]
    assert selected == set(range(2048, 4096))
    assert (row['prefetch'], row['unused_prefetch'], row['final'], row['initial_cold']) == (2, 1, 1, 0)
    assert [window['unused'] for window in row['windows']] == [1, 0]
    assert [query['unique_new'] for query in row['queries']] == [2048, 0, 0, 0]
    assert row['q_jaccard'] == [[1.] * 4] * 4
    assert row['queries'][0]['cutoff_tie_population'] == 4096
    assert row['queries'][0]['selected_cutoff_margin']['mean'] == 0
    assert sum(row['queries'][0]['physical_position_hist']) == 2048


@pytest.mark.parametrize('mutation,error', [
    ('ordinal', 'ordinals'), ('identity', 'identity'), ('threshold', 'Threshold'),
    ('kind', 'threshold'), ('processed', 'bounds'), ('dtype', 'shapes'),
])
def test_invalid_evidence_is_rejected(tmp_path, mutation, error):
    capture = scored_capture(tmp_path)
    a = capture.arrays
    if mutation == 'ordinal':
        a['post.prefetch_ordinals'][:] = 0
    elif mutation == 'identity':
        a['post.prefetch_records'][0, 2] = 0
    elif mutation == 'threshold':
        a['post.threshold_records'][0, 0, 0, 2] = 0
    elif mutation == 'kind':
        a['post.threshold_records'][0, 0, 0, 3] = 7
    elif mutation == 'processed':
        a['post.window_processed_ends'][0, 1] = 1024
    elif mutation == 'dtype':
        a['post.threshold_records'] = a['post.threshold_records'].astype(np.float32)
    with pytest.raises(ValueError, match=error):
        capture_rows(capture)


def test_short_threshold_prefix_uses_native_zero_sentinel(tmp_path):
    capture = scored_capture(tmp_path)
    capture.arrays['post.threshold_records'][0, 0, 0] = [1024, 2048, 0, 1]
    assert capture_rows(capture)[0][0]['queries'][0]['thresholds'][0]['key'] == 0


def test_offline_cli_real_schema_and_temporal_statistics(tmp_path):
    dp = tmp_path / 'dp2'
    dp.mkdir()
    rounds = []
    for number in (1, 2):
        directory = dp / str(number)
        directory.mkdir()
        capture = scored_capture(directory)
        if number == 2:
            # All scores tie: replacing a selected token remains legal TopK.
            capture.arrays['post.topk'][:, :, -1] = 2048
        archive = directory / 'scores.npz'
        np.savez_compressed(archive, **capture.arrays)
        manifest = capture.manifest
        manifest['archive'].update(bytes=archive.stat().st_size,
                                   sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        manifest['arrays'] = {name: {'dtype': value.dtype.name, 'shape': list(value.shape)}
                              for name, value in capture.arrays.items()}
        manifest['metadata'].update(dp_rank=2, group=10,
            context={'round': 100 + number, 'global_decode_confirmed': True},
            requests=[{'request_id': 'private-request', 'owner': 0, 'query_begin': 0, 'query_stop': 4}])
        capture.path.write_text(json.dumps(manifest))
        rounds.append({'round_index': number, 'scheduler_round': 100 + number,
                       'groups': [{'group': 10, 'manifest': f'{number}/manifest.json'}]})
    (dp / 'series.json').write_text(json.dumps(dict(
        dp_rank=2, dp_ranks=[2], schema_version=1, kind='kvaf_decode_snapshot_series',
        status='complete', content='scores', batch=1, queries=4,
        completed_rounds=2, expected_rounds=2,
        request_owner_signature=[{'request_id': 'private-request', 'owner': 0}], rounds=rounds)))
    output = tmp_path / 'analysis'
    from diagnostics import calibrate_captures
    calibrate_captures(tmp_path, 10, (1, 2), output)
    targets = json.loads((output / 'fit_targets.json').read_text())
    assert targets['fitted_generator'] is False
    assert targets['request_rounds'] == 2
    assert targets['temporal_jaccard']['mean'] == pytest.approx(2047 / 2049)
    assert targets['totals']['initial_cold'] == 1
    rows = json.loads((output / 'request_rounds.json').read_text())
    assert rows[0]['source_round'] == 1 and rows[1]['scheduler_round'] == 102
    assert rows[0]['temporal'] is None
    assert 'private-request' not in (output / 'request_rounds.json').read_text()
    # Artifact corruption must fail before any fitting claims are produced.
    with (dp / '2/scores.npz').open('ab') as file:
        file.write(b'bad')
    with pytest.raises(ValueError, match='checksum'):
        calibrate_captures(tmp_path, 10, (1, 2), output)


def core_timeline_fixture():
    cores = [[100, 1000] + [0]*30 for _ in range(6)]
    p = [0]*64
    p[1:4] = [110, 120, 130]
    for q in range(4):
        p[10+5*q:15+5*q] = list(range(140+q*20, 145+q*20))
    u = [0]*32
    u[10] = 110
    for q in range(5):
        u[12+2*q:14+2*q] = [230+q*20, 240+q*20]
    u[22:25] = [340, 350, 360]
    p[32:] = u
    f = [0]*128
    for q in range(4):
        cores[2+q][2:7] = [120, 400, 420, 430, 500]
        f[q*24:q*24+3] = [120, 300, 310]
    f[96:101] = [400, 800, 850, 900, 450]
    for i in range(6):
        f[104+2*i:106+2*i] = [460+i*20, 470+i*20]
    return dict(batch=1, workers_per_query=1, clock_mhz=1000,
                cores=cores, producer_update=[p], fetch=[f], flags=[[[1, 0, 0]]*12])


def test_same_replay_latest_core_and_stages():
    raw = core_timeline_fixture()
    raw['cores'][3][1] = 1100
    result = module.decode_timeline(raw)
    assert result['last_block'] == 3
    assert result['observed_span_us'] == 1
    assert result['cores'][3]['role'] == 'fetch Q2 lane0'
    assert len([s for s in result['cores'][0]['stages'] if not s['name'].startswith('control_gap')]) == 27
    assert any(s['name'] == 'wait.Q4_fetch' for s in result['cores'][0]['stages'])


@pytest.mark.parametrize('mutation', ['missing_exit', 'missing_wait', 'publication_order'])
def test_incomplete_or_inconsistent_timeline_fails(mutation):
    raw = copy.deepcopy(core_timeline_fixture())
    if mutation == 'missing_exit':
        raw['cores'][0][1] = 0
    if mutation == 'missing_wait':
        raw['fetch'][0][105] = 0
    if mutation == 'publication_order':
        raw['fetch'][0][1] = 130
    with pytest.raises(ValueError):
        module.decode_timeline(raw)


def test_publication_core_preserves_fetch_roles_and_all_query_stages():
    raw = core_timeline_fixture()
    raw['resolver_cores'] = 3
    publisher = [100, 1000] + [0] * 30
    publisher[2:4] = [115, 125]
    for q in range(4):
        publisher[4+q*6:10+q*6] = list(range(140+q*30, 146+q*30))
    raw['cores'].insert(2, publisher)
    raw['flags'][0].extend([[1, 0, 0]] * 4)
    result = module.decode_timeline(raw)
    assert result['cores'][2]['role'] == 'output_pack b0'
    assert result['cores'][3]['role'] == 'fetch Q1 lane0'
    assert result['cores'][6]['role'] == 'fetch Q4 lane0'
    stages = [s for s in result['cores'][2]['stages']
              if not s['name'].startswith('control_gap')]
    assert len(stages) == 22
    assert stages[-1]['name'] == 'Q4.publish'
    raw['cores'][2][20] = 0
    with pytest.raises(ValueError):
        module.decode_timeline(raw)


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
        assert 'start_us=0.00 end_us=10.00 duration_us=10.00' in log
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
    import diagnostics as diagnostics
    import numpy as np

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
    assert '0.11' in output  # 656 bytes / 6 us, decimal GB/s.
    for value in ('samples=1', 'fetch Q1 lane0', 'Window prefetch / unused',
                  'Per-Q demand / repair', 'Per-Q resolver / update', '1.00', '2.00'):
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
    assert 'Per-Q resolver / update' not in level_two_tables(result)
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
    assert 'submit/copy' in timing and '3.00' in timing and 'Cycles' in timing
    result['level'] = 2
    assert 'Prefetch service' not in level_two_tables(result)


def test_level_two_repeats_are_aggregated_without_mutating_raw_samples():
    from copy import deepcopy

    from diagnostics import aggregate_diagnostic_rows, level_two_tables
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
    assert '6.00 [4.00, 8.00]' in output
    assert 'repeat=' not in output
    assert result == original
    result['level'] = 3
    assert level_two_tables(result).count('Demand and actual reads') == 1
    assert aggregate_diagnostic_rows([[3, 'fetch', 2., '*'], [3, 'fetch', 4., '']], 2) == [
        [3, 'fetch', '3.00 [2.00, 4.00]', '1/2']]


def test_initial_reset_clears_stateful_logical_snapshot():
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    # Execute the production reset with lightweight buffers; no NPU imports.
    source = ast.parse((Path(__file__).parent / 'lifetch/support/execution.py').read_text())
    reset = next(node for node in ast.walk(source) if isinstance(node, ast.FunctionDef)
                 and node.name == 'reset' and 'update.logical' in ast.unparse(node))
    class Buffer:
        def copy_(self, value):
            self.value = value
        def fill_(self, value):
            self.value = value
    state = {'before_logical': [3, 2, 1], 'keep': 42}
    logical = Buffer()
    scope = dict(owner=SimpleNamespace(in_flight=False, reset=lambda: None, mark_launched=lambda: None),
        saved=[], arguments=[], update=SimpleNamespace(logical=logical, identity=[0, 1, 2]),
        state=state, rotating=SimpleNamespace(hot_valid=Buffer(), lru_slots=Buffer(), reserved_hot=Buffer()),
        bench_only=True, order_snapshot=([], []))
    exec(compile(ast.Module(body=[reset], type_ignores=[]), '<production reset>', 'exec'), scope)
    scope['reset']()
    assert logical.value == [0, 1, 2]
    assert state == {'keep': 42}


def test_display_precision_preserves_integer_counts():
    from diagnostics import format_metric
    assert format_metric(2.539174847615422) == '2.54'
    assert format_metric(23.755) == '23.75'
    assert format_metric(0.0) == '0.00'
    assert format_metric(59696) == '59696'
    assert format_metric(None) == 'N/A'


def events():
    return [dict(name='replay', ph='X', ts=100, dur=100),
            dict(name='QuantLightningIndexer_1', ph='X', cat='kernel', ts=110, dur=20),
            dict(name='a5_lidu_fused_batch_get_prepare_mtp_2', ph='X', cat='kernel', ts=135, dur=10),
            dict(name='KvafExecuteBatchGetAivKernelV2_3', ph='X', cat='kernel', ts=150, dur=30)]


def test_native_names_and_gaps_are_accounted_without_ordinal_guessing():
    result = account_native(events(), 'replay', 80)
    assert 'error' not in result
    assert result['stages_us'] == dict(li=20, lidu_prepare=10, fetch=30)
    assert result['kernel_span_us'] == 70
    assert result['inter_kernel_gap_us'] == 10
    assert result['boundary_residual_us'] == 10


def test_unknown_native_kernel_remains_visible_and_incomplete():
    trace = events()
    trace[-1]['name'] = 'DifferentVendorGet'
    result = account_native(trace, 'replay', 80)
    assert 'error' in result
    assert result['missing_stages'] == ['fetch']
    assert result['unclassified'][0]['name'] == 'DifferentVendorGet'


def test_cpu_launch_names_are_not_device_timings():
    trace = events()
    trace[-1]['cat'] = 'cpu_op'
    assert 'error' in account_native(trace, 'replay', 80)


def test_native_known_device_names_without_category():
    trace = events()
    for event in trace:
        event.pop('cat', None)
    result = account_native(trace, 'replay', 100)
    assert 'error' not in result
    assert set(result['stages_us']) == {'li', 'lidu_prepare', 'fetch'}


class PressureTest(unittest.TestCase):
    def test_read_counts_and_cycle_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'current-b16-m15/case0'
            path.mkdir(parents=True)
            report = dict(status='pass', native_misses_per_request=[2]*16,
                          timing={k: {'p50_us': 100} for k in ('native_chain','prototype_with_update')},
                          read_samples={'prototype_with_update': [dict(prefetch_hot=4,prefetch_cold=20,
                              unused_hot=4,unused_cold=0,final=12,total=36)]})
            (path/'report.json').write_text(json.dumps(report))
            sample = dict(round=0, prefetch=24, final=12, accounting={'stages_us': {'resolve_update': 50}},
                          streaming_cycles=[{'core_timeline_raw': {'batch':16}}])
            (path/'full_attribution.json').write_text(json.dumps({'samples':[sample]}))
            cores = [dict(block=0,role='resolver+finalize b0',start_us=0,end_us=50,stages=[]),
                     dict(block=1,role='index_update b0',start_us=0,end_us=20,stages=[]),
                     dict(block=2,role='fetch Q1 lane0',start_us=0,end_us=40,stages=[],published_misses=[12],
                          KVAF_counters={'safe_post_cycles':2000,'safe_CQ_cycles':10000})]
            timeline = dict(cores=cores,last_block=0,clock_mhz=1000,observed_span_us=50)
            with patch('diagnostics.decode_timeline', return_value=timeline):
                text, _, records = analyze_prefetch_pressure(root,16,[15])
                self.assertIn('native_reads=32',text)
                self.assertIn('tail_after_fetch=10.000',text)
                self.assertIn('| 2.00 | 10.00',text)
                self.assertEqual(len(records),1)
                report['read_samples']['prototype_with_update'][0]['total'] = 99
                (path/'report.json').write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError,'Read accounting'):
                    analyze_prefetch_pressure(root,16,[15])


def test_profile_sequence_resets_only_at_sequence_boundaries(tmp_path, monkeypatch):
    log, samples = [], []
    inputs = [SimpleNamespace(source_round=n, scheduler_round=n + 100,
                              identities=((2, 'req', 1),)) for n in (11, 12)]
    class Sequence:
        rounds = ((11,), (12,))
        def __iter__(self):
            return iter(inputs)
    class Profile:
        def __init__(self, **kwargs):
            self.callback = kwargs['on_trace_ready']
            self.steps = 0
        def __enter__(self):
            return self
        def __exit__(self, *args):
            assert self.steps == 4
            self.callback(self)
        def step(self):
            self.steps += 1
        def export_chrome_trace(self, path):
            from pathlib import Path
            Path(path).write_text('{"traceEvents": []}')
    def schedule(**kwargs):
        assert kwargs == dict(wait=0, warmup=0, active=4, repeat=1)
        return None
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace(profiler=SimpleNamespace(
        profile=Profile, schedule=schedule, ProfilerActivity=SimpleNamespace(CPU=0, NPU=1))))
    monkeypatch.setitem(sys.modules, 'torch.profiler', SimpleNamespace(record_function=lambda name: nullcontext()))
    stream = SimpleNamespace(synchronize=lambda: None)
    def event(**kwargs):
        return SimpleNamespace(record=lambda stream: None, synchronize=lambda: None,
                               elapsed_time=lambda other: 0.01)
    torch = SimpleNamespace(npu=SimpleNamespace(current_stream=lambda device: stream, Event=event))
    layout = np.zeros((2, 4), dtype=np.int64)
    class Tensor:
        def __getitem__(self, index):
            return SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: layout[index]))
    state = SimpleNamespace(device=0, rotating=object(), storage=SimpleNamespace(logical_ids=Tensor()))
    def reset():
        log.append('reset')
        layout.fill(0)
    def prepare(item):
        log.append(('prepare', item.source_round, int(layout[1, 0])))
    def replay():
        layout[1] += 1
    def validate(item):
        log.append(('validate', item.source_round, int(layout[1, 0])))
        # A numerical failure is retained; Q/Index state still advances.
        if item.source_round == 11:
            raise AssertionError('numerical fixture failure')
    state.reset, state.prepare = reset, prepare
    state.forward = SimpleNamespace(calibrate_counter_mhz=lambda: 1000,
        set_diagnostics=lambda enabled: log.append(('diagnostics', enabled)), diagnostic_cycles=lambda: [])
    monkeypatch.setattr('diagnostics.algorithm_before', lambda rotating: {})
    def algorithm_after(rotating, before):
        old = int(before['logical_ids'][0, 0])
        assert int(layout[1, 0]) == old + 1
        samples.append(old)
        return dict(totals=dict(prefetch=old, final=1), requests=[])
    monkeypatch.setattr('diagnostics.algorithm_after', algorithm_after)
    monkeypatch.setattr('diagnostics.account', lambda *args, **kwargs: dict(stages_us=dict(li=2)))
    config = SimpleNamespace(warmup=1, iterations=2, transport='urma')
    result = collect_replay(torch, state, SimpleNamespace(replay=replay), Sequence(), validate, config, tmp_path)
    assert log.count('reset') == 3  # One warmup sequence, two profiled sequences.
    assert [item for item in log if isinstance(item, tuple) and item[0] == 'prepare'] == [
        ('prepare', 11, 0), ('prepare', 12, 1)] * 3
    assert samples == [0, 1, 0, 1]
    assert result['correctness'] == 'FAIL' and result['samples'] == 4
    for source in (11, 12):
        data = json.loads((tmp_path / f'round-{source}/full_attribution.json').read_text())['samples']
        assert [sample['round'] for sample in data] == [0, 1]
        assert all(sample['source_round'] == source for sample in data)
        assert all(sample['scheduler_round'] == source + 100 for sample in data)
    assert log[-1] == ('diagnostics', False)


def test_reports_actual_payload_and_both_writers():
    payload = np.arange(64, dtype=np.uint8).reshape(8, 8)
    output = payload.copy()
    output[2] = payload[5]
    selected = np.full((4, 2048), 1, dtype=np.int32)
    ids = selected.copy()
    slots = selected.copy()
    selected[0, 3] = ids[0, 3] = slots[0, 3] = 2
    destinations = np.array([2], dtype=np.int32)
    addresses = np.array([123], dtype=np.int64)
    posts = np.zeros((4, 32), dtype=np.int32)
    posts[0, 0] = 1
    lru = np.array([2, 4], dtype=np.int32)
    mapping = np.full(8, -1, dtype=np.int32)
    mapping[5] = 8
    report = summarize_mismatches(0, selected, ids, slots, output, payload,
                                  destinations, addresses, posts, lru, mapping, 8, 8, np.array([0, 123, 0, 0, 0, 0, 0, 0], dtype=np.uint64))
    assert report['mismatch_positions'] == 1
    item = report['examples'][0]
    assert item['actual_matching_tokens'] == [5]
    assert item['prefetch_writers'] == [5]
    assert item['final_writers'] == [{'q': 1, 'descriptor': 0, 'remote_address': '0x7b', 'source_tokens': [1]}]


def test_native_jetty_audit_rotation_and_phases():
    phases = [list(range(2500)), list(range(5000, 5200))]
    a, b = split_native_jetty_descriptors(phases)
    assert len(a) + len(b) == 2700
    assert a[:128] == list(range(128)) and b[:128] == list(range(128, 256))
    assert a[-128:] == list(range(5000, 5128))
    assert b[-72:] == list(range(5128, 5200))
    assert sorted(a + b) == list(range(2500)) + list(range(5000, 5200))


def test_snapshot_compares_kv_and_reports_prefetch_differences(tmp_path):
    import numpy as np
    import pytest
    from diagnostics import compare_transport_snapshots, save_transport_snapshot

    ids = np.tile(np.arange(2048, dtype=np.int32), (1, 4, 1))
    slots = ids.copy()
    output = np.arange(4096 * 4, dtype=np.uint8).reshape(1, 4096, 4)
    mapping = np.full((1, 4096), -1, dtype=np.int32)
    mapping[0, 8] = 4096
    lru = np.arange(2048, dtype=np.int32)[None, :]
    rows = np.zeros((1, 256), dtype=np.uint32)
    rows[0, 16] = 1
    rows[0, 214] = 1
    addresses = np.arange(4096, dtype=np.int64)[None, :]
    a, b = tmp_path / 'a.npz', tmp_path / 'b.npz'
    save_transport_snapshot(a, ids, slots, output, mapping, lru, rows, addresses)
    mapping[0, 9] = 4097
    rows[0, 17] = 1
    rows[0, 215] = 2
    lru[0, 1] = 17
    save_transport_snapshot(b, ids, slots, output, mapping, lru, rows, addresses)
    result = compare_transport_snapshots(a, b, 4096)
    assert 'logical_topk=identical selected_kv_bytes=identical' in result[0]
    assert 'lru_candidate_order_different_rows=1/1 differing_positions=1' in result
    assert any('ubmem_only=1' in line for line in result)
    output[0, 0, 0] ^= 1
    save_transport_snapshot(b, ids, slots, output, mapping, lru, rows, addresses)
    with pytest.raises(AssertionError, match='KV bytes differ'):
        compare_transport_snapshots(a, b, 4096)
