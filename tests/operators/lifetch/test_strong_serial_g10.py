import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / 'support'))
import strong_serial_g10 as serial


def fixture(area, monkeypatch):
    root = area / 'fused.serial'
    case = root / 'g10/serial2/results/job/cases/diagnose'
    raw = case / 'serial/path_result.json'
    inputs = dict(
        source_paths=[], rounds=1, page_count=12, kv_payload=16,
        geometry=dict(batch=16, resident=16384, token_bytes=656, heads=32, history=131328),
        identities=[[rank, 0] for rank in range(8)]*2)
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    for rank in range(8):
        manifest = area / f'dp{rank}/manifest.json'
        write(manifest, dict(archive=dict(sha256=f'{rank:064x}')))
        inputs['source_paths'].append(str(manifest))
    ident = serial.dp_capture_id(10, inputs)
    monkeypatch.setitem(serial.INPUT_AND_RESULTS, 10, (ident, {}))
    monkeypatch.setattr(serial, 'from_this_host', lambda r: r == root)
    write(case/'config.json', dict(group=10, warmup=4, iterations=20, device=0, round_range=[1,1],
                                    mode='diagnose', input='real', intermediate_fetches=4,
                                    prefetch_tokens=1024, prefetch_rank_start=1024))
    write(case/'replay_input.json', inputs)
    samples = [dict(invocation=i, us=325., stages_us=dict(li=220., lidu_prepare=30., fetch=70.),
                    intervals_us=dict(li=[[0.,220.]], lidu_prepare=[[222.,252.]], fetch=[[255.,325.]]))
               for i in range(20)]
    timing = dict(samples=20, graph_calls=20, measured_graphs=1, timing_protocol='one_profiled_graph',
                  event_mean_us=329., p50_us=325., samples_us=[325.]*20,
                  stage_mean_us=samples[0]['stages_us'], inter_kernel_gap_mean_us=5., overlap_mean_us=0.,
                  outside_invocations_mean_us=4.)
    write(raw, dict(correctness='PASS', failures=[], path='serial', transport='urma', runtime_workers=2,
                    input_paths=inputs['source_paths'], all_rounds=timing))
    write(raw.parent/'diagnose/round-1/graph_profile.json', dict(samples=samples, accounting={}))
    place = dict(status='measured', page_mode='thp', bytes=(12+2)*656*128, ub_device='ub0',
                 coverage_min_pct=100, numa_pages_overlapping_vmas={'3':128})
    write(root/'g10/serial2/thp.json', place)
    (root/'physical_npu.txt').write_text('4')
    candidate = area / 'fused.owner'; candidate.mkdir()
    (candidate/'logical-ties.txt').write_text(serial.OWNER_TOPK)
    (candidate/'physical_npu.txt').write_text('4')
    monkeypatch.setattr(serial, 'load_candidate', lambda *args, **kw: (inputs, None, place, 310., 301.))
    return root, candidate, inputs, raw


def test_reusable_source_validated_serial_vs_archived_candidate(tmp_path, monkeypatch, capsys):
    root, candidate, inputs, raw = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, 'argv', ['serial', 'cached', str(tmp_path), str(candidate)])
    assert serial.main() == 0  # discovers this already-complete local archive
    assert (tmp_path/serial.MARKER).resolve() == raw
    assert 'graph_mean/P50_us=329.00/325.00' in capsys.readouterr().out
    line = serial.comparison_line(tmp_path, inputs, (310.,301.))
    assert 'graph_saved_mean/P50_us=+19.00/+24.00' in line
    assert 'source=fused.serial' in line
    assert serial.main() == 0  # reuse the published reference without measuring

    result = json.loads(raw.read_text())
    result['runtime_workers'] = 32
    raw.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        serial.main()


def test_missing_or_wrong_capture_never_registers_reference(tmp_path, monkeypatch):
    root, candidate, inputs, raw = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, 'argv', ['serial', 'cached', str(tmp_path), str(candidate)])
    (root/'physical_npu.txt').write_text('0')
    assert serial.main() == 3
    assert not (tmp_path/serial.MARKER).exists()
    (root/'physical_npu.txt').write_text('4')
    with pytest.raises(ValueError):
        serial.same_capture(inputs, {**inputs, 'identities':list(reversed(inputs['identities']))})
