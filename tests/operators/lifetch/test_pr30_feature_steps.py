from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent/'support'))
import pr30_feature_steps as report
from reporting import atomic_json


def receipts(policy):
    # Row zero triggers batching even though two individual Qs are tiny.
    # Row one sits immediately below the row-level cutoff.
    final = [[1, 2, 253, 0], [255, 0, 0, 0]] + [[0]*4 for _ in range(14)]
    cqe = [[1, 2, 4, 0] if policy == 'paced' else [1, 1, 1, 0], [1, 0, 0, 0]]
    return [dict(prefetch=[0]*16, final=deepcopy(final), cqe=deepcopy(cqe)+[[0]*4 for _ in range(14)],
                 first_post_peak_4us_tokens=32, peak_pending_queries=3) for _ in range(10)]


def test_measured_per_query_completions():
    values = receipts('paced')
    r = report.receipt_summary(values, True)
    assert r == dict(prefetch=0, final=511, eligible_q=20, split_q=20, split_rows=10,
                     cqe_hist=[600, 20, 10, 0, 10])
    with pytest.raises(ValueError, match='CQ receipts'):
        report.receipt_summary(values, False)
    values[0]['cqe'][0][1] = 4
    with pytest.raises(ValueError, match='CQ receipts'):
        report.receipt_summary(values, True)
    with pytest.raises(ValueError, match='Incomplete'):
        report.receipt_summary(values[:9], True)
    with pytest.raises(ValueError, match='Incomplete'):
        report.receipt_summary(values * 2, True)


def test_separate_diagnostic_uses_actual_first_batch_cq_for_paced(tmp_path, monkeypatch):
    phases = {k: [1, 2] for k in ('WQE_and_publish','index_update_and_join','final_CQ_join','prefetch_CQ_join')}
    monkeypatch.setattr(report, 'terminal_detail', lambda folder, expected_calls: dict(status='OK', phases_us=phases))
    def single_post(sample, policy):
        assert policy == 'single'
        return dict(first_post_peak_4us_tokens=250, peak_pending_queries=4)
    monkeypatch.setattr(report, 'diagnostic_pressure', single_post)
    samples = [dict(round=i, source_round=1, algorithm=dict(totals=dict(prefetch=0, final=511)))
               for i in range(20)]
    atomic_json(tmp_path/'diagnose/round-1/detail_samples.json', samples)
    calls = receipts('paced')
    assert report.diagnostic(tmp_path, 'single', 20)['first_post_peak_4us_tokens'] == 250
    with pytest.raises(ValueError, match='incomplete diagnostic calls'):
        report.diagnostic(tmp_path, 'paced', 10, calls)
    atomic_json(tmp_path/'diagnose/round-1/detail_samples.json', samples[:10])
    assert report.diagnostic(tmp_path, 'paced', 10, calls)['first_post_peak_4us_tokens'] == 32
    with pytest.raises(ValueError, match='Missing paced'):
        report.diagnostic(tmp_path, 'paced', 10)
    with pytest.raises(ValueError, match='Live doorbells'):
        report.diagnostic(tmp_path, 'single', 10, calls)
    calls[0]['final'][0][1] += 1
    with pytest.raises(ValueError, match='traffic disagrees'):
        report.diagnostic(tmp_path, 'paced', 10, calls)


def test_archive_compiler_drift_and_fresh_comparisons(tmp_path, monkeypatch, capsys):
    out, archive = tmp_path/'block1', tmp_path/'cached-main'
    common = dict(physical_npu='4', host='b24', kvaf=report.SDK, kvaf_tree=report.SDK_TREE)
    for root, extra in [(out, dict(**report.REVISIONS, reference=str(archive))),
                        (archive, dict(main=report.REVISIONS['mainline']))]:
        root.mkdir()
        for k, val in {**common, **extra}.items():
            (root/f'{k}.txt').write_text(val)
    monkeypatch.setattr(report, 'oracle_for', lambda owner, group: ())
    controls = dict(runtime={'host':'b24','visible':'4'})
    thp = dict(coverage_min_pct=100, ub_device='u0', numa_pages_overlapping_vmas={'3':1})
    def collect(root, group, arm, owner, oracle, host):
        assert root == (archive if arm == 'mainline' else out)
        base = {'mainline':100., 'dualcut':90., 'paced':84.}[arm]
        metrics = {k: (base + i, base - 2 + i) for i, k in enumerate(report.METRICS)}
        sha = dict(report.EXPECTED_TEXT)
        if arm in ('dualcut','paced'):
            sha[report.LI_TEXT[0]] = 'changed LI'
        if arm == 'dualcut':
            sha[report.TERMINAL] = 'compiler variant'
        d = dict(prefetch=0, final=0, first_post_peak_4us_tokens=0, peak_pending_queries=0,
                 last_worker_mean_us={'WQE_and_publish':1, 'index_update_and_join':2,
                                      'final_CQ_join':3, 'prefetch_CQ_join':4})
        m = report.receipt_summary(receipts('paced' if arm == 'paced' else 'single'), arm == 'paced')
        return metrics, m, d, sha, controls.copy(), thp, 1280 if arm == 'mainline' else 640
    monkeypatch.setattr(report, 'collect_arm', collect)
    for group in report.GROUPS:
        (out/f'g{group}').mkdir()
        (out/f'g{group}/order.txt').write_text('paced dualcut')
        report.report(out, group, tmp_path/'owner')
    report.summary(out)
    row = report.read(out/'g7/feature_steps.json')
    assert row['device_text'] == dict(main_tail_equals_dualcut=False, dualcut_li_equals_paced=True)
    assert row['order'] == ['paced','dualcut']
    assert row['calls_per_graph'] == {'mainline': 20, 'dualcut': 10, 'paced': 10}
    assert set(row['measured']) == {'dualcut','paced'}
    text = capsys.readouterr().out
    assert 'G7 dualcut_vs_mainline saved graph_mean/P50_us=+10.00/+10.00' in text
    assert 'historical_main calls_per_graph=20/10 unmodified_stage_binary=DIFF' in text
    assert 'fresh_same_block calls_per_graph=10/10 unmodified_stage_binary=MATCH' in text
    assert 'G7 CHECK arm=mainline TopK_score_vs_strong_serial=PASS calls=20/20 queries=1280/1280' in text
    assert 'G7 CHECK arm=paced TopK_score_vs_strong_serial=PASS calls=10/10 queries=640/640' in text
    assert 'eligible_Q=20/640 split_Q=20/640 split_rows=10/160' in text
    assert 'FIVE_GROUP_EQUAL_WEIGHT paced_vs_dualcut graph_saved_mean_us=+6.000' in text
    (out/'g7/reference.txt').write_text('unavailable')
    report.report(out, 7, tmp_path/'owner')
    report.summary(out)
    missing = report.read(out/'g7/feature_steps.json')
    assert missing['reference'] is None and missing['calls_per_graph'] == {'dualcut': 10, 'paced': 10}
    assert set(missing['timing']) == set(missing['measured']) == {'dualcut', 'paced'}
    text = capsys.readouterr().out
    assert 'G7 dualcut_vs_mainline UNAVAILABLE' in text and 'G7 CHECK arm=mainline' not in text
    assert 'FIVE_GROUP_EQUAL_WEIGHT dualcut_vs_mainline UNAVAILABLE missing_groups=7' in text
    assert 'FIVE_GROUP_EQUAL_WEIGHT paced_vs_dualcut graph_saved_mean_us=+6.000' in text
    def moved(root, group, arm, owner, oracle, host):
        values = list(collect(root, group, arm, owner, oracle, host))
        if arm == 'paced':
            values[5] = dict(thp, ub_device='another_HCA')
        return values
    monkeypatch.setattr(report, 'collect_arm', moved)
    with pytest.raises(ValueError, match='placement differs'):
        report.report(out, 7, tmp_path/'owner')
    (out/'g7/reference.txt').write_text(str(archive))
    (archive/'host.txt').write_text('different-node')
    with pytest.raises(ValueError, match='different host'):
        report.report(out, 7, tmp_path/'owner')
