import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent/'support'))
from correctness import topk_fingerprint
import li_window_report as report
from main_five_report import MAIN
from test_main_five_report import owner_fixture, profile, put


def test_five_case_target_is_the_reduction_in_equal_weight_graph_means():
    rows = [dict(group=g, mainline={'graph': (value, 1)}, candidate={'graph': (70, 1)})
            for g, value in zip(report.GROUPS, (40, 60, 100, 150, 150))]
    assert report.aggregate(rows) == (100, 70, 30)
    with pytest.raises(ValueError, match='five capture groups'):
        report.aggregate(rows[1:])


def test_real_report_reads_archived_main_not_owner_and_refuses_placement_mismatch(tmp_path, capsys):
    group = 10
    root, ref, owner = (tmp_path/name for name in ('new', 'baseline', 'owner'))
    src, oracle = owner_fixture(owner, group)
    receipt = json.loads((src/'score_topk_check.json').read_text())
    identity = dict(receipt['native_reference']['identity'], inputs={'key': 'hash'})
    receipt['native_reference']['identity'] = identity
    put(src/'score_topk_check.json', receipt)
    path = Path(receipt['native_reference']['archive_path'])
    oracle_doc = json.loads(path.read_text())
    oracle_doc['native_reference']['identity'] = identity
    put(path, oracle_doc)
    inputs = dict(source_paths=['captured'], geometry={'token_bytes': 656})
    put(src.parent/'replay_input.json', inputs)
    candidate = 'a' * 40

    for r, label, rev, total in ((ref, 'mainline', MAIN, 2000), (root, 'li-window', candidate, 1700)):
        r.mkdir()
        (r/'main.txt').write_text(MAIN)
        (r/'physical_npu.txt').write_text('4')
        (r/'kvaf.txt').write_text(report.KVAF)
        arm = r/f'g{group}'/label
        fused = arm/'results/test/cases/diagnose/fused'
        trace, timing = profile(fused)
        data = json.loads(trace.read_text())
        data['graph_accounting']['event_us'] = total
        for sample in data['samples']:
            sample['algorithm'] = dict(totals={'prefetch': 5, 'final': 3})
        put(trace, data)
        timing['mean_us'] = timing['event_mean_us'] = total/20
        sha = hashlib.sha256(rev.encode()).hexdigest()
        binary = fused/'kernel.so'
        if r == root:
            binary.write_bytes(rev.encode())
        put(fused/'mainline_checks.json', dict(production_revision=rev, production_terminal=True,
            profiled_terminal_kernel='v9_parallel_terminal', measured_calls=20, measured_kv_index_rows=320,
            ordered_diagnostic_calls=20, ordered_diagnostic_rows=320, library_sha256=sha,
            loaded_library=str(binary), runtime={'host':'B24'}, capture_operand_hashes=identity['inputs']))
        put(fused.parent/'config.json', dict(group=group, transport='urma', input='real', device=0, physical_npu=4,
            warmup=4, iterations=20, intermediate_fetches=4, prefetch_tokens=1024, prefetch_rank_start=1024,
            fused_only=True, production_revision=rev, kvaf_root='/test/kvaf'))
        put(fused.parent/'replay_input.json', inputs)
        put(fused/'path_result.json', dict(input_paths=inputs['source_paths'], geometry=inputs['geometry'],
            path='fused', transport='urma', runtime_workers=40, correctness='PASS', failures=[],
            all_rounds=timing, topk_set_digests=[topk_fingerprint(oracle)]*20))
        strong = dict(receipt['native_reference'], origin='completed-owner-archive', archive_path=str(owner))
        put(fused/'score_topk_check.json', dict(receipt, native_reference=strong, exact_queries=1280, exact_calls=20))
        np.savez_compressed(fused/'score_topk_ids.npz', fused=np.broadcast_to(oracle, (20,16,4,2048)))
        put(arm/'thp.json', dict(status='measured', page_mode='thp', bytes=1, coverage_min_pct=100,
            ub_device='same', numa_pages_overlapping_vmas={'0':1}))
    (root/'candidate.txt').write_text(candidate)
    (root/'reference.txt').write_text(str(ref))
    report.compare(root, group, owner, ref)
    assert 'graph_saved_mean/P50_us=+15.00/+0.00' in capsys.readouterr().out
    artifact = root/f'g{group}/comparison.json'
    assert json.loads(artifact.read_text())['mainline']['graph'][0] == 100
    artifact.unlink()
    thp_file = root/'g10/li-window/thp.json'
    thp = json.loads(thp_file.read_text())
    put(thp_file, dict(thp, numa_pages_overlapping_vmas={'1':1}))
    with pytest.raises(ValueError, match='different_numa'):
        report.compare(root, group, owner, ref)
    assert not artifact.exists()


def test_explicit_main_cannot_accidentally_rerun_the_archive():
    launcher = Path(__file__).parent/'main_five_npu4.sh'
    bad = subprocess.run(['bash', launcher, '/unavailable', 'reader', MAIN], capture_output=True, text=True)
    assert bad.returncode == 2 and 'independent candidate' in bad.stderr


def test_final_window_cost_uses_last_finishing_row_then_last_query(tmp_path):
    assert report.critical_window_workers(tmp_path) is None
    events = [dict(epoch=3, lane=row, end_cycles=5000+row) for row in range(16)]
    detail = dict(streaming_cycles=[dict(terminal_timeline=dict(clock_mhz=100), prefetch_services=events)])
    early = [900, 1000, 1100, 1500, 0, 0, 0, 0]
    late = [1150, 1000, 1200, 2000, 0, 0, 0, 0]
    rows = [[early]*4 for _ in range(16)]
    rows[15] = [early, late, early, early]
    put(tmp_path/'diagnose/round-1/detail_samples.json', [detail]*20)
    put(tmp_path/'window_workers.json', dict(calls=[rows]*20))
    workers = report.critical_window_workers(tmp_path)
    assert workers['timings'] == ([1.5]*20, [0.5]*20, [8.0]*20)
    assert workers['last_rows'] == [0]*15 + [20]
    assert workers['last_queries'] == [0, 20, 0, 0]
    assert workers['tied_rows'] == workers['tied_queries'] == 0
    events[3]['end_cycles'] = events[15]['end_cycles']
    put(tmp_path/'diagnose/round-1/detail_samples.json', [detail]*20)
    tied = report.critical_window_workers(tmp_path)
    assert tied['last_rows'][3] == 20 and tied['last_queries'] == [20, 0, 0, 0]
    assert tied['tied_rows'] == tied['tied_queries'] == 20


def test_counts_completed_work_and_ignores_late_offers():
    unused = [1, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    active = [1, 1, 12, 24, 50, 1, 256, 7, 1, 32, 0, 20]
    rows = [[unused, unused]] * 14 + [[active, unused], [active, active]]
    assert report.penultimate_worker_counts([rows]*20) == [280, 20, 20]
    incomplete = list(active)
    incomplete[1] = 0
    with pytest.raises(ValueError, match='completion'):
        report.penultimate_worker_counts([[*rows[:-1], [active, incomplete]]]*20)
