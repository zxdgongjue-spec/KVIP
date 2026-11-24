import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / 'support'))
import cap_pair_report as report
from reporting import atomic_json


def test_individual_graph_event_timings_are_separate_from_kernel_span(tmp_path):
    events = [150 + i for i in range(20)]
    samples = []
    for i, event in enumerate(events):
        acc = dict(invocation=i, event_us=event, stages_us=dict(li=30, topk_resolve_update=50),
                   intervals_us=dict(li=[[0, 10], [20, 30], [40, 50]], topk_resolve_update=[[60, 110]]))
        samples.append(dict(us=110, accounting=acc))
    path = tmp_path / 'diagnose/round-1/full_attribution.json'
    atomic_json(path, dict(samples=samples, graph_calls=20))
    timing = dict(timing_protocol='one_state_individual_graphs', measured_graphs=20, graph_calls=20,
                  samples_us=events, mean_us=159.5, event_mean_us=159.5, p50_us=159.5)
    assert report.cap_times(tmp_path, timing) == {
        'graph': (159.5, 159.5), 'LI1_to_LI3': (30, 30), 'final_topk_through_tail': (50, 50)}
    timing['p50_us'] = 110
    with pytest.raises(ValueError, match='per-call events'):
        report.cap_times(tmp_path, timing)
    timing['p50_us'] = 159.5
    samples[6]['accounting']['intervals_us']['li'][2] = [40, 65]
    atomic_json(path, dict(samples=samples, graph_calls=20))
    with pytest.raises(ValueError, match='invalid kernel intervals'):
        report.cap_times(tmp_path, timing)


def make_cap_case(tmp_path, monkeypatch):
    owner = tmp_path / 'oracle'
    group = 2
    files = {'main': report.NATIVE, 'kvaf': report.SDK, 'kvaf_tree': report.SDK_TREE, 'physical_npu': '4', 'host': 'B'}
    for file, value in files.items():
        (tmp_path / (file + '.txt')).write_text(value)
    capture = dict(geometry=dict(batch=16, history=131328), page_count=16384, source_paths=['frozen-input'])
    ends = [[131069, 131070, 131071, 131072]]*16
    src = dict(effective_native_ends=ends, identity={'inputs': {'query_ends': 'digest'}})
    monkeypatch.setattr(report, 'archived_oracle', lambda *args: (None, {'native_reference': src}))
    folder = owner / 'g2/logical-ties/results/r/cases/diagnose/fused'
    atomic_json(folder / 'path_result.json', {})
    atomic_json(folder.parent / 'replay_input.json', capture)
    thp = {name: dict(bytes=123456, coverage_min_pct=100., ub_device='hca4', numa_pages_overlapping_vmas={'3': 10})
           for name in report.CAPS}
    caps = {}
    for arm, history in report.CAPS.items():
        folder = tmp_path / 'g2' / arm / 'results/r/cases/diagnose/fused'
        atomic_json(folder / 'path_result.json', {})
        atomic_json(folder.parent / 'capture_input.json', capture)
        atomic_json(folder.parent / 'replay_input.json', {**capture, 'geometry': dict(batch=16, history=history)})
        caps[arm] = dict(capture_history=131328, bound_history=history, effective_ends=ends,
                         query_ends_sha256='digest', captured_page_count=16384, bound_page_count=16384,
                         capture_page_table_sha256='table', capture_state_sha256=dict.fromkeys(
                             ('captured_hot_valid', 'captured_lru', 'tail_bounds', 'logical_address_valid',
                              'logical_score_keys', 'logical_score_valid'), 'hash'),
                         measured_requests=dict(prefetch=[100]*20, final=[5]*20))
    (tmp_path/'g2/order.txt').write_text('cap512k cap128k')
    def measured(root, group, arm, *args, **kw):
        from numpy import ones
        val = 100. if arm == 'cap128k' else 120.
        return (dict(graph=(val, val), LI1_to_LI3=(.6*val, .6*val), final_topk_through_tail=(.4*val, .4*val)),
                ones((20, 16, 4)), thp[arm], dict(runtime=dict(host='B', visible='4'), history_cap=caps[arm],
                    library_sha256='hash', capture_operand_hashes=src['identity']['inputs']), dict(kvaf_root='/sdk', history=report.CAPS[arm]))
    monkeypatch.setattr(report, 'measured_parallel', measured)
    return owner, group, caps, thp


def test_same_input_pair_and_signed_overhead(tmp_path, monkeypatch, capsys):
    owner, group, _, _ = make_cap_case(tmp_path, monkeypatch)
    report.report(tmp_path, group, owner)
    output = capsys.readouterr().out
    assert 'cap512k_minus_cap128k graph_mean/P50_us=+20.00/+20.00' in output
    assert 'graph_mean_change_pct=+20.00' in output
    assert 'order=cap512k,cap128k' in output


def test_changed_active_request_or_unmatched_hca_has_no_comparison(tmp_path, monkeypatch):
    owner, group, caps, thp = make_cap_case(tmp_path, monkeypatch)
    original = caps['cap512k']['effective_ends']
    caps['cap512k']['effective_ends'] = [[200000]*4]*16
    with pytest.raises(ValueError, match='effective bounds'):
        report.report(tmp_path, group, owner)
    caps['cap512k']['effective_ends'] = original
    thp['cap512k']['ub_device'] = 'another-hca'
    with pytest.raises(ValueError, match='different_ub_device'):
        report.report(tmp_path, group, owner)
    assert not (tmp_path/'g2/capacity_comparison.json').exists()


def test_summary_requires_all_five_groups(tmp_path, capsys):
    for g in report.GROUPS[:-1]:
        obj = dict(group=g, host='B', physical_npu=4, runtime={}, library_sha256='hash', kvaf_root='/kvaf',
                   delta={'graph': [5., 4.]}, timing={'cap128k': {'graph':[100., 90.]},
                                                      'cap512k': {'graph':[105.,94.]}})
        atomic_json(tmp_path/f'g{g}/capacity_comparison.json', obj)
    with pytest.raises(FileNotFoundError):
        report.summary(tmp_path)
    g = report.GROUPS[-1]
    atomic_json(tmp_path/f'g{g}/capacity_comparison.json', {**obj, 'group': g})
    report.summary(tmp_path)
    assert 'cap512k_minus_cap128k_us=+5.000' in capsys.readouterr().out
