"""The cutoff optimization compares independent archived runs without launching controls."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / 'support'))
import li_restore_report as report
from li_restore_archive import published


def test_published_li_and_strong_serial_signs(monkeypatch):
    line, = published(17, 'no_local_copy', (270, 270), (25, 25))
    assert 'LI_saved_mean/P50_us=-1.09/-1.23' in line and 'LI4_saved_mean/P50_us=+1.01/+0.98' in line
    monkeypatch.setattr(report, 'dp_capture_id', lambda group, inputs: report.INPUT_AND_RESULTS[group][0])
    result = report.serial_graph_line(7, {}, (400, 400))
    assert 'graph_saved_mean/P50_us=+189.60/+175.60' in result
    assert 'gap_to_35_mean/P50_us=+16.76/+25.86' in result
    assert 'UNAVAILABLE' in report.serial_graph_line(10, {}, (400, 400))
    monkeypatch.setattr(report, 'dp_capture_id', lambda *args: None)
    assert 'UNVERIFIED' in report.serial_graph_line(7, {}, (400, 400))


@pytest.mark.parametrize('fault', (None, 'boot', 'placement', 'query', 'revision'))
def test_previous_exact_run_requires_same_host_inputs_and_set(tmp_path, monkeypatch, fault):
    old, new = (tmp_path / name for name in ('fused.original', 'fused.current'))
    new.mkdir()
    folder = old / 'g17/logical-ties/results/test/cases/diagnose/fused'
    details = folder / 'diagnose/round-1'
    details.mkdir(parents=True)
    for root in (old, new):
        (root / 'physical_npu.txt').write_text('4')
    (old / 'logical-ties.txt').write_text(report.FIRST_EXACT if fault != 'revision' else '0'*40)
    (folder / 'path_result.json').write_text('{}')
    archived = dict(native_reference=dict(identity=dict(boot='boot-uuid', inputs=dict(query='query-hash'))),
                    group=17, measured_calls=20, queries_per_call=64)
    (folder / 'native_li_compare.json').write_text(json.dumps(archived))
    if fault == 'boot':
        archived['native_reference']['identity']['boot'] = 'different-boot-uuid'
    samples = [dict(accounting=dict(stages_us=dict(li=16), intervals_us=dict(li=[[0, 2], [2, 5], [5, 9], [9, 16]])))]
    (details / 'full_attribution.json').write_text(json.dumps(dict(samples=samples * 20)))
    thp = dict(coverage_min_pct=100, ub_device='test', numa_pages_overlapping_vmas={'1': 100})
    prior = [{}, ['digest']*20, thp, 50, 48, None, None, None, 21, None, 20]
    data = prior.copy()
    data[3:5] = 48, 45
    if fault == 'placement':
        data[2] = {**thp, 'numa_pages_overlapping_vmas': {'2': 100}}
    monkeypatch.setattr(report, 'load', lambda *args, **kw: prior)
    monkeypatch.setattr(report, 'read_sets', lambda *args: (0, 0, 0, 0, fault != 'query'))
    result = report.exact_archive_line(new, 17, data, ((12, 11), (4, 3)), archived)
    if fault:
        assert 'UNAVAILABLE' in result and 'saved_mean' not in result
    else:
        assert 'LI_saved_mean/P50_us=+4.00/+5.00 LI4_saved_mean/P50_us=+3.00/+4.00' in result
        assert 'graph_saved_mean/P50_us=+2.00/+3.00' in result
