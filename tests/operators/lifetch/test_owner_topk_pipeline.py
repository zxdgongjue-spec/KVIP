import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / 'support'))
from diagnostics import account
from owner_topk_report import times
from performance import split_graph_invocations


def trace(*, joined, li_count, mixed=False):
    name = 'lifetch_measured_graph'
    events = [dict(name=name, ph='X', ts=0, dur=2050)]
    for i in range(20):
        t = 100*i
        # Kernel names follow the measured CANN trace: symbol plus numeric dispatch suffix.
        for j, kernel in enumerate(('v9_li_kernel_0', 'v9_li_kernel_1_1', 'v9_li_kernel_2_2',
                                    'v9_li_kernel_3_3')[:li_count]):
            events.append(dict(name=kernel, ph='X', ts=t + 4*j, dur=2))
        tail = 'v9_owner_topk_resolve_update_6' if joined else 'v9_streaming_resolve_update_5'
        events.append(dict(name=tail, ph='X', ts=t+18, dur=10 if i < 15 else 28))
        if mixed and i == 10:
            events.append(dict(name='v9_streaming_resolve_update_5', ph='X', ts=t+49, dur=2))
    return account(events, name, 2000., full=True, fused_final=True, owner_topk=joined)


def test_three_li_and_named_joined_kernel_per_call(tmp_path):
    for joined, count in ((False, 4), (True, 3)):
        summary = split_graph_invocations(trace(joined=joined, li_count=count), 20)
        folder = tmp_path / str(joined) / 'diagnose/round-1'
        folder.mkdir(parents=True)
        (folder/'full_attribution.json').write_text(json.dumps(
            dict(samples=[dict(accounting=s) for s in summary['samples']])))
        timing = times(folder.parents[1], (100, 99), joined=joined)
        assert timing['graph'] == (100, 99)
        assert timing['LI1_to_LI3'] == (6, 6)
        assert timing['final_topk_through_tail'] == ((14.5, 10) if joined else (20.5, 16))
    with pytest.raises(ValueError, match='three LI phases'):
        split_graph_invocations(trace(joined=True, li_count=4), 20)
    with pytest.raises(ValueError, match='separate legacy resolver'):
        split_graph_invocations(trace(joined=True, li_count=3, mixed=True), 20)
    with pytest.raises(ValueError, match='expected 3 LI phases'):
        times(tmp_path/'False', (100, 99), joined=True)


def test_legacy_tail_cannot_pass_owner_topk_trace():
    events = [dict(name='measured', ph='X', ts=0, dur=100),
              dict(name='v9_li_kernel_0', ph='X', ts=10, dur=3),
              dict(name='v9_streaming_resolve_update_5', ph='X', ts=30, dur=4)]
    rejected = account(events, 'measured', 60., full=True, fused_final=True, owner_topk=True)
    assert 'error' in rejected
    assert 'error' not in account(events, 'measured', 60., full=True, fused_final=True)


def test_report_pairs_one_oracle_and_two_archived_score_receipts(tmp_path, monkeypatch, capsys):
    import numpy as np
    import owner_topk_report as report
    from correctness import topk_fingerprint
    from score_equivalence import save_checked
    from test_score_equivalence import example

    oracle, parent, new = (tmp_path / name for name in ('exact', 'free-ties', 'new'))
    case = Path('g4/logical-ties/results/job/cases/diagnose/fused')
    original = oracle / case / 'native_li_compare.json'
    original.parent.mkdir(parents=True)
    serial_inputs, serial = example(16)
    identity = dict(inputs=dict(capture='same'), device='4')
    source = dict(identity=identity, replay_identity=identity, archive_path=str(original), origin='completed-archive')
    placements = dict(coverage_min_pct=100, ub_device='urma0', numa_pages_overlapping_vmas={'3':1000})
    receipts = [[('a'*64, 'b'*64)]*16 for _ in range(20)]
    data = {}
    for root, revision in ((oracle, report.HANDOFF_REVISION), (parent, report.FREE_TIES), (new, 'test-sha')):
        root.mkdir(exist_ok=True)
        (root / 'logical-ties.txt').write_text(revision)
        (root / 'physical_npu.txt').write_text('4')
    (parent / 'exact-handoff-reference.txt').write_text(str(oracle))
    for root, joined in ((parent, False), (new, True)):
        folder = root / case
        profile = folder / 'diagnose/round-1'
        profile.mkdir(parents=True)
        result = split_graph_invocations(trace(joined=joined, li_count=3 if joined else 4), 20)
        (profile/'full_attribution.json').write_text(json.dumps(
            dict(samples=[dict(accounting=s) for s in result['samples']])))
        outputs = np.broadcast_to(serial, (20, *serial.shape)).copy()
        outputs[::2, 0, 0, -1] = 2048 if joined else 2049
        save_checked(folder, 4, serial_inputs, serial, source, outputs)
        data[root] = (serial_inputs.geometry, [topk_fingerprint(x) for x in outputs],
                      placements, 99 if joined else 106, 96 if joined else 104, root.name,
                      None, None, 0, None, 0, receipts, None, 0, '/kvaf')
    monkeypatch.setattr(report, 'measurement', lambda *_: (data[parent], {}, source, ({},64,True,'PASS',serial),
                                                            original))
    monkeypatch.setattr(report, 'load', lambda root, *args, **kwargs: data[root])
    monkeypatch.setattr(report, 'serial_graph_line', lambda *_: 'serial fixture')
    monkeypatch.setattr(report, 'dp_capture_id', lambda group, meta: report.INPUT_AND_RESULTS[group][0])
    report.report(new, 4, parent)
    output = capsys.readouterr().out
    assert 'graph_archive_minus_candidate_mean/P50_us=+7.00/+8.00' in output
    assert 'final_topk_through_tail_archive_minus_candidate_mean/P50_us=+6.00/+6.00' in output
    assert 'standalone_LI_per_call=3 fused_owner_topk_tail_per_call=1' in output
    assert 'cutoff_tie_only_sets=10/1280' in output
    (parent / 'logical-ties.txt').write_text('wrong')
    with pytest.raises(ValueError, match='score-equivalent parent'):
        report.report(new, 4, parent)


def test_g2_report_checks_local_native_reference_without_parent(tmp_path, monkeypatch, capsys):
    import hashlib
    import numpy as np
    import li_restore_report
    import owner_topk_report as report
    from correctness import topk_fingerprint
    from score_equivalence import save_checked
    from test_score_equivalence import example

    folder = tmp_path / 'g2/logical-ties/results/job/cases/diagnose/fused'
    profile = folder / 'diagnose/round-1'
    profile.mkdir(parents=True)
    (tmp_path/'physical_npu.txt').write_text('4')
    result = split_graph_invocations(trace(joined=True, li_count=3), 20)
    (profile/'full_attribution.json').write_text(json.dumps(
        dict(samples=[dict(accounting=s) for s in result['samples']])))
    inputs, serial = example(16)
    identity = dict(operator='torch_npu.npu_quant_lightning_indexer', visible='4')
    source = dict(origin='fresh', cache_key=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                  identity=identity, layout_query='BSND', layout_key='PA_BSND', sparse_count=2048,
                  effective_native_ends=inputs.query_ends.tolist(), input_paths=['same-input']*8)
    outputs = np.broadcast_to(serial, (20, *serial.shape)).copy()
    outputs[7:, 1, 3, -1] = 2048
    save_checked(folder, 2, inputs, serial, source, outputs)
    rows = [[('a'*64, 'b'*64)]*16 for _ in range(20)]
    record = (dict(source_paths=source['input_paths']), [topk_fingerprint(call) for call in outputs], None,
              301, 300, None, None, None, None, None, None, rows)
    monkeypatch.setattr(report, 'load', lambda *args, **kwargs: record)
    monkeypatch.setattr(report, 'dp_capture_id', lambda group, _: report.INPUT_AND_RESULTS[group][0])
    monkeypatch.setattr(li_restore_report, 'dp_capture_id', report.dp_capture_id)
    report.report(tmp_path, 2, tmp_path/'no-parent-measurement')
    output = capsys.readouterr().out
    assert 'cutoff_tie_only_sets=13/1280' in output and 'native_LI_reference=fresh' in output
    assert 'vs_q1_handoff_free_ties UNAVAILABLE' in output
    assert 'graph_saved_mean/P50_us=-2.28/-7.30' in output
    assert 'graph_saved_mean/P50_us=+21.88/+15.59' in output
    monkeypatch.setattr(report, 'dp_capture_id', lambda *_: 'wrong')
    with pytest.raises(ValueError, match='archived G2'):
        report.report(tmp_path, 2, tmp_path/'no-parent-measurement')
    monkeypatch.setattr(report, 'dp_capture_id', lambda *_: None)
    monkeypatch.setattr(li_restore_report, 'dp_capture_id', report.dp_capture_id)
    report.report(tmp_path, 2, tmp_path/'no-parent-measurement')
    output = capsys.readouterr().out
    assert 'G2 CHECK TopK_score_vs_strong_serial=PASS' in output
    assert 'vs_serial2_archive UNVERIFIED' in output and 'vs_original_fusion UNVERIFIED' in output
