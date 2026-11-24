import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent/'support'))
import pr30_archive_compare as audit
from reporting import atomic_json


def test_alone_event_and_amortized_event_are_preserved(tmp_path):
    for protocol in (audit.INDIVIDUAL, audit.PACKED):
        individual = protocol == audit.INDIVIDUAL
        samples = []
        for i in range(20):
            acc = dict(invocation=i, stages_us=dict(li=30, topk_resolve_update=50),
                       intervals_us=dict(li=[[0, 10], [12, 22], [25, 35]], topk_resolve_update=[[40, 90]]))
            if individual:
                acc['event_us'] = 250
            samples.append(dict(us=90, accounting=acc))
        event = 250 if individual else 105
        timing = dict(timing_protocol=protocol, measured_graphs=20 if individual else 1, graph_calls=20,
                      samples_us=[250 if individual else 90]*20, event_mean_us=event, mean_us=event,
                      p50_us=250 if individual else 90)
        profile = dict(samples=samples, graph_calls=20, graph_accounting=dict(event_us=event*20))
        atomic_json(tmp_path/'diagnose/round-1/full_attribution.json', profile)
        checked = (audit.cap_times if individual else audit.main_times)(tmp_path, timing)
        metrics = audit.decomposition(profile, timing, checked)
        assert metrics['kernel_span'] == (90, 90)
        assert metrics['inter_kernel_gap'] == (10, 10)
        assert metrics['event_mean'] == event
        assert metrics['event_outside_span_mean'] == event-90


def markers(root, revision):
    root.mkdir(exist_ok=True)
    for k, value in dict(main=revision, kvaf=audit.SDK, kvaf_tree=audit.SDK_TREE,
                         host='node24', physical_npu=4, machine_id='m').items():
        (root/f'{k}.txt').write_text(str(value))


def archive(root, revision, arm, group, stats):
    markers(root, revision)
    folder = root/f'g{group}'/arm/'results/r/cases/diagnose'
    atomic_json(folder/'config.json', dict(production_revision=revision))
    atomic_json(folder/'row.fixture.json', stats)


def row(protocol, history, outside, numa='3'):
    return dict(protocol=protocol, binary='digest', runtime=dict(host='node24', visible='4'), config=dict(test='real'),
                geometry=dict(history=history, batch=16), oracle='oracle', capture='input', cap_state=['pages', 'state'],
                stats=dict(event_mean=90+outside, event_outside_span_mean=outside,
                           LI1_to_LI3=[30,30], final_topk_through_tail=[50,50],
                           inter_kernel_gap=[10,10], kernel_span=[90,90]),
                thp=dict(coverage_min_pct=100, ub_device='u0', numa_pages_overlapping_vmas={numa: 300},
                         bytes=123456))


def read_fixture(root, arm, revision, g):
    folder, = (root/f'g{g}'/arm/'results').glob('*/cases/diagnose')
    assert audit.read(folder/'config.json')['production_revision'] == revision
    return audit.read(folder/'row.fixture.json')


def test_lookup_only_reuses_valid_packed_group(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'measure_arm', read_fixture)
    candidate = tmp_path/'candidate'
    archive(candidate, '1'*40, 'cap512k', 7, row(audit.INDIVIDUAL, 524544, 170))
    for name, revision, old in (
            ('a-one-call', audit.PR30, row(audit.INDIVIDUAL, 131328, 180)),
            ('b-different-numa', audit.PR30, row(audit.PACKED, 131328, 4, numa='2')),
            ('c-different-capture', audit.PR30, dict(row(audit.PACKED, 131328, 4), capture='different')),
            ('d-experiment', audit.FEATURE, row(audit.PACKED, 131328, 4)),
            ('e-same', audit.PR30, row(audit.PACKED, 131328, 9))):
        archive(tmp_path/name, revision, 'cap128k', 7, old)
    assert audit.find_exact_group(candidate, 7, None) == tmp_path/'e-same'
    (tmp_path/'e-same'/'kvaf_tree.txt').write_text('other')
    assert audit.find_exact_group(candidate, 7, None) is None
    with pytest.raises(ValueError, match='Missing candidate'):
        audit.find_exact_group(candidate, 10, None)


@pytest.mark.parametrize('protocol,comparable', [(audit.PACKED, 1),(audit.INDIVIDUAL, 0)])
def test_per_group_reference_map_and_protocol_gate(tmp_path, monkeypatch, capsys, protocol, comparable):
    monkeypatch.setattr(audit, 'measure_arm', read_fixture)
    candidate = tmp_path/'candidate'
    path_map = {}
    for group in audit.GROUPS:
        archive(candidate, 'a'*40, 'cap512k', group, row(protocol, 524544, 180))
        old = tmp_path/f'old-{group}'
        archive(old, audit.PR30, 'cap128k', group, row(audit.PACKED, 131328, 9))
        path_map[str(group)] = str(old)
    manifest = tmp_path/'refs.json'
    atomic_json(manifest, path_map)
    args = SimpleNamespace(reference=None, repo=None, reference_map=manifest, feature_reference=None)
    audit.run(candidate, args)
    output = capsys.readouterr().out
    assert f'FIVE_GROUP_EQUAL_WEIGHT vs=exact_PR30 comparable_operator_event={comparable}' in output
    assert 'candidate_minus_reference_event_mean_us=+171.000 four_kernel_span_mean_us=+0.000' in output
