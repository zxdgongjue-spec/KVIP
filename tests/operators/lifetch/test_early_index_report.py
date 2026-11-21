"""Candidate-only replay consumes the archived oracle and rejects incomparable runs."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / 'support'))
import early_index_report as report
import owner_pack_tail as tail
from correctness import topk_fingerprint


def test_archived_oracle_and_level1_savings(tmp_path, monkeypatch, capsys):
    archive = tmp_path / 'fused.m3IizR'
    new = tmp_path / 'candidate'
    for root in (archive, new):
        root.mkdir()
        (root / 'physical_npu.txt').write_text('4')
    (archive / 'logical-ties.txt').write_text('24c619b9581aad43fd53c815928def26a1e19560')
    (new / 'early-handoff-reference.txt').write_text(str(archive))
    identity = dict.fromkeys(('inputs','host','npu','visible','operator','sparse_mode','torch','torch_npu','cann'), 'same')
    ids = np.broadcast_to(np.arange(2048, dtype=np.int32), (16, 4, 2048))
    refs = dict(strong_serial=ids, strong_serial_repeat=ids[..., ::-1], same_layout_native=ids)
    digest = topk_fingerprint(ids)
    receipts = [[('a' * 64, 'b' * 64)] * 16] * 20
    place = dict(coverage_min_pct=100, ub_device='urma0', numa_pages_overlapping_vmas={'3':1000})
    runs = {}
    def write(root, refs, metadata, event_mean, span, li, tail_us):
        folder = root / 'g10/logical-ties/results/run/cases/diagnose/fused'
        profile = folder / 'diagnose/round-1'
        profile.mkdir(parents=True)
        counts = dict(matched_calls=20, matched_queries=1280, total_queries=1280)
        comparison = dict(group=10, phase='measured', measured_calls=20, queries_per_call=64,
                          domains={}, serial_reference_stability=dict(stable_queries=64),
                          comparisons={k:counts for k in refs}, raw_ids='ids.npz', native_reference=metadata)
        (folder / 'native_li_compare.json').write_text(json.dumps(comparison))
        np.savez(folder / 'ids.npz', fused=np.broadcast_to(ids, (20, *ids.shape)), **refs)
        accounting = dict(stages_us=dict(li=li), intervals_us=dict(li=[(i*li, i*li+li/4) for i in range(4)]))
        (profile / 'full_attribution.json').write_text(json.dumps(dict(samples=[dict(accounting=accounting)]*20)))
        runs[root] = ('inputs', [digest]*20, place.copy(), event_mean, span, root.name, None, [span]*20,
                      tail_us, 1, tail_us, receipts, None, li, '/runtime')
    write(archive, refs, dict(identity=identity), 401, 400, 275, 119)
    def load(root, group, label, *, row_dependencies):
        assert (group, label, row_dependencies) == (10, 'logical-ties', True)
        return runs[root]
    monkeypatch.setattr(report, 'load', load)
    monkeypatch.setattr(tail, 'load', load)
    reused, metadata = tail.archived_references(archive, 10, identity.copy())
    for k in refs:
        np.testing.assert_array_equal(reused[k], refs[k])
    with pytest.raises(ValueError, match='inputs'):
        tail.archived_references(archive, 10, dict(identity, inputs='different'))
    write(new, reused, metadata, 398, 399, 276, 116)
    assert report.report(new, 10) == 0
    output = capsys.readouterr().out
    assert 'queries=1280/1280' in output and 'same_event_calls=20/20' in output
    assert 'graph_saved_mean/P50_us=+3.00/+1.00 tail_saved_mean/P50_us=+3.00/+3.00 LI_saved_mean/P50_us=-1.00/-1.00' in output
    runs[new][2]['ub_device'] = 'other'
    assert report.report(new, 10) == 0
    output = capsys.readouterr().out
    assert 'UNCOMPARABLE placement=different_ub_device' in output and 'graph_saved' not in output
