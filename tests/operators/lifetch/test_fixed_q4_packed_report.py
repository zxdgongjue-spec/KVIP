# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / 'support'))
from fixed_q4_packed_report import cell_summary, main, report_cell


def write_block(root, cohort, block, *, wrong_input=False):
    for arm, base in [('serial', 400), ('fused', 300)]:
        directory = root / f'g17/bs8/cohort{cohort}/block{block}' / arm
        directory.mkdir(parents=True)
        us = base + 12 * cohort + 4 * block
        identity = dict(group=17, batch=8, host='test', native='reader', physical_npu=3,
                        donor_rows=list(range(cohort, 16, 2)), sequence_sha256='other' if wrong_input and arm == 'fused' else 'same')
        data = dict(arm=arm, calls=10, identity=identity, checked_queries=320, checked_rows=80,
                    correctness='PASS', get_workers=2 if arm == 'serial' else 40,
                    graph_event_us=10 * us, graph_mean_us=us, mean_us=dict(LI=210, tail=77))
        if arm == 'fused':
            data['traffic'] = [dict(prefetch=20, final=5) for _ in range(10)]
        (directory / 'packed_result.json').write_text(json.dumps(data))
        (directory / 'placement_verified').touch()
        thp = dict(status='measured', page_mode='thp', coverage_min_pct=100, ub_device='nic',
                   numa_pages_overlapping_vmas={'3': 44})
        (directory / 'thp.json').write_text(json.dumps(thp))


def test_both_blocks_and_all_donor_cohorts(tmp_path):
    for key, value in [('native', 'reader'), ('host', 'test'), ('physical_npu', '3')]:
        (tmp_path / (key + '.txt')).write_text(value)
    for cohort in (0, 1):
        for block in (1, 2):
            write_block(tmp_path, cohort, block, wrong_input=cohort == 1 and block == 2)
    with pytest.raises(ValueError, match='inputs, runtime or memory placement'):
        cell_summary(tmp_path, 17, 8)
    last = tmp_path / 'g17/bs8/cohort1/block2/fused/packed_result.json'
    data = json.loads(last.read_text())
    data['identity']['sequence_sha256'] = 'same'
    last.write_text(json.dumps(data))
    row = cell_summary(tmp_path, 17, 8)
    assert row['mean_us']['serial']['graph'] == 412
    assert row['mean_us']['fused']['graph'] == 312
    assert row['saved_us']['graph'] == 100
    assert row['graph_gain_pct'] == pytest.approx(10000 / 412)
    assert row['reached_20pct']
    data['graph_mean_us'] = data['mean_us']['LI'] + data['mean_us']['tail']
    last.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='ten-call device event'):
        cell_summary(tmp_path, 17, 8)


def test_new_kernel_reuses_only_archived_times_and_checks_each_block(tmp_path):
    control, fresh = tmp_path / 'control', tmp_path / 'fresh'
    for root in (control, fresh):
        root.mkdir()
        for key, value in [('native', root.name), ('host', 'test'), ('physical_npu', '3')]:
            (root / (key + '.txt')).write_text(value)
    (fresh / 'reference.txt').write_text(str(control))
    for cohort in (0, 1):
        for block in (1, 2):
            for root in (control, fresh):
                write_block(root, cohort, block)
                directory = root / f'g17/bs8/cohort{cohort}/block{block}'
                for arm in ('serial', 'fused'):
                    p = directory / arm / 'packed_result.json'
                    if arm == 'serial' and root == fresh:
                        p.unlink()
                        continue
                    data = json.loads(p.read_text())
                    data['identity']['native'] = root.name
                    if root == fresh:
                        data['graph_mean_us'] -= 5
                        data['graph_event_us'] -= 50
                    p.write_text(json.dumps(data))
    result = cell_summary(fresh, 17, 8)
    assert result['no_slowdown'] and result['reached_15pct']
    assert result['saved_vs_previous_us']['graph'] == 5
    assert result['saved_us']['graph'] == 105
    assert result['gap_to_15_us'] == pytest.approx(307 - 0.85 * 412)
    assert result['gap_to_20_us'] == pytest.approx(307 - 0.80 * 412)
    assert result['tokens_per_call']['previous_fused'] == {'prefetch': 20, 'final': 5}
    assert result['graph_saved_vs_previous_by_block_us'] == [5, 5]
    p = fresh / 'g17/bs8/cohort1/block2/fused/packed_result.json'
    data = json.loads(p.read_text())
    data['graph_mean_us'] += 40
    data['graph_event_us'] += 400
    p.write_text(json.dumps(data))
    result = cell_summary(fresh, 17, 8)
    assert result['reached_15pct'] and not result['no_slowdown']
    assert result['graph_saved_vs_previous_by_block_us'] == [5, -15]


def test_one_candidate_block_uses_both_measured_control_blocks(tmp_path, capsys, monkeypatch):
    control, fresh = tmp_path / 'control', tmp_path / 'fresh'
    for root in (control, fresh):
        root.mkdir()
        for key, value in [('native', 'reader'), ('host', 'test'), ('physical_npu', '3')]:
            (root / (key + '.txt')).write_text(value)
    (fresh / 'reference.txt').write_text(str(control))
    (fresh / 'candidate_blocks.txt').write_text('1')
    for cohort in (0, 1):
        for block in (1, 2):
            write_block(control, cohort, block)
        write_block(fresh, cohort, 1)
        serial = fresh / f'g17/bs8/cohort{cohort}/block1/serial/packed_result.json'
        serial.unlink()

    row = cell_summary(fresh, 17, 8)
    assert row['mean_us']['serial']['graph'] == 412
    assert row['mean_us']['previous_fused']['graph'] == 312
    assert row['mean_us']['fused']['graph'] == 310
    assert row['saved_vs_previous_us']['graph'] == 2
    assert 'graph_saved_vs_previous_by_block_us' not in row
    assert row['previous_graph_by_block_us'] == [310, 314]
    assert row['blocks'] == 1
    assert row['calls_by_arm'] == {'serial': 40, 'previous_fused': 40, 'fused': 20}
    report_cell(row)
    output = capsys.readouterr().out
    assert 'candidate_calls=20 archived_calls_per_arm=40 candidate_queries=640 archived_queries_per_arm=1280' in output
    assert 'previous_minus_new_block1/block2' not in output
    monkeypatch.setattr(sys, 'argv', ['report', 'summary', str(fresh), '--groups', '17', '--batches', '8'])
    main()
    summary = json.loads((fresh / 'packed_summary.json').read_text())
    assert (summary['blocks'], summary['archived_blocks_per_arm']) == (1, 2)

    # A mismatched archived second block cannot silently enter the comparison.
    path = control / 'g17/bs8/cohort0/block2/fused/packed_result.json'
    data = json.loads(path.read_text())
    data['identity']['sequence_sha256'] = 'other'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='inputs, runtime or memory placement'):
        cell_summary(fresh, 17, 8)
