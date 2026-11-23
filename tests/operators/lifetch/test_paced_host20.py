import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent/'support'))
import paced_host20 as report
from correctness import topk_fingerprint
from test_main_five_report import profile, put


def test_b20_fresh_reference_runs_without_b24_archives(tmp_path, capsys):
    root = tmp_path/'local-run'
    folder = root/'g7/paced-tail'/'results/test/cases/diagnose/fused'
    _, timing = profile(folder)
    binary = folder/'test.so'
    binary.write_bytes(b'local')
    (root/'candidate.txt').write_text(report.PACED)
    serial = np.broadcast_to(np.arange(2048, dtype=np.int32), (16,4,2048))
    outputs = np.broadcast_to(serial, (20,16,4,2048))
    fingerprint = topk_fingerprint(serial)
    np.savez_compressed(folder/'score_topk_ids.npz', fused=outputs, strong_serial=serial)
    identity = dict(visible='0', inputs={'query':'q'})
    put(folder/'score_topk_check.json', dict(group=7, calls=20, queries=1280, exact_queries=1280,
        exact_calls=20,serial_set_digest=fingerprint, native_reference=dict(origin='fresh',identity=identity)))
    put(folder/'path_result.json', dict(correctness='PASS',failures=[],topk_set_digests=[fingerprint]*20,
                                         all_rounds=timing))
    put(folder.parent/'config.json', dict(production_revision=report.PACED,physical_npu=0,device=0))
    put(folder.parent/'replay_input.json', {'source_paths':[]})
    proof = dict(production_revision=report.PACED,runtime={'visible':'0'},
                 capture_operand_hashes=identity['inputs'],production_terminal=True,
                 profiled_terminal_kernel='v9_parallel_terminal', measured_calls=20,
                 measured_kv_index_rows=320,ordered_diagnostic_calls=20,ordered_diagnostic_rows=320,
                 loaded_library=str(binary),library_sha256=hashlib.sha256(b'local').hexdigest())
    put(folder/'mainline_checks.json', proof)
    put(root/'g7/paced-tail/thp.json',dict(status='measured',page_mode='thp',bytes=656))
    report.report(root,7)
    out = capsys.readouterr().out
    assert 'TopK_score_vs_strong_serial=PASS' in out
    assert 'vs_B20_NPU0_mainline UNAVAILABLE' in out
    assert 'vs_archived_NPU4_serial2 historical_other_device' in out
    proof['runtime']['visible']='4'; put(folder/'mainline_checks.json', proof)
    with pytest.raises(ValueError,match='device'):
        report.report(root,7)


def test_gate_requires_local_device_and_fails_on_a_single_regression(tmp_path, capsys):
    for group in report.GROUPS:
        put(tmp_path/f'g{group}/paced_comparison.json',dict(
            graph=[80,79],local={'serial2':[100,100], 'mainline':None if group==2 else [100,100]}))
    report.summary(tmp_path)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].endswith('NOT_MET/PASS')
    assert lines[1].endswith('UNAVAILABLE/UNAVAILABLE')
    put(tmp_path/'g2/paced_comparison.json',dict(graph=[101,99],local={'serial2':None,'mainline':[100,100]}))
    report.summary(tmp_path)
    assert capsys.readouterr().out.splitlines()[1].endswith('NOT_MET/PASS')
