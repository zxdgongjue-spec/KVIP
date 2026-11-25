import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / 'support'))
import main_five_report as main
from correctness import topk_fingerprint


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def owner_fixture(root, group):
    root.mkdir()
    (root/'logical-ties.txt').write_text(main.OWNER)
    (root/'physical_npu.txt').write_text('4')
    folder = root/f'g{group}/logical-ties/results/123/cases/diagnose/fused'
    put(folder/'path_result.json', {})
    ids = np.broadcast_to(np.arange(2048, dtype=np.int32), (16,4,2048))
    source = dict(identity=dict(visible='4', operator='torch_npu.npu_quant_lightning_indexer'),
                  effective_native_ends=np.full((16,4), 2049).tolist())
    if group == 2:
        source['origin'] = 'validated-cache'
        path = folder/'score_topk_ids.npz'
    else:
        oracle = root/'other/native_li_compare.json'
        put(oracle, dict(group=group, phase='measured', measured_calls=20,
                         raw_ids='native_li_ids.npz', native_reference=source))
        source.update(origin='completed-archive', archive_path=str(oracle))
        path = oracle.parent/'native_li_ids.npz'
    np.savez_compressed(path, strong_serial=ids)
    put(folder/'score_topk_check.json', dict(group=group, calls=20, queries=1280,
        serial_set_digest=topk_fingerprint(ids), native_reference=source))
    return folder, ids


@pytest.mark.parametrize('group', [2, 10])
def test_oracle_reads_the_native_ids_not_the_candidate_ids(tmp_path, group):
    root = tmp_path/'owner'
    folder, ids = owner_fixture(root, group)
    actual, receipt = main.archived_oracle(root, group)
    assert np.array_equal(actual, ids)
    receipt['serial_set_digest'] = '0'*64
    put(folder/'score_topk_check.json', receipt)
    with pytest.raises(ValueError, match='native TopK IDs'):
        main.archived_oracle(root, group)


def profile(folder, calls=20):
    parts, durations = [], []
    for i in range(calls):
        start = i*60
        li = [[start+j*5, start+j*5+3] for j in range(3)]
        end = start+33+(i%4)
        tail = [[start+18, end]]
        durations.append(end-start)
        parts.append(dict(us=end-start, accounting=dict(invocation=i,
            intervals_us=dict(li=li, topk_resolve_update=tail),
            stages_us=dict(li=9, topk_resolve_update=end-start-18))))
    path = folder/'diagnose/round-1/full_attribution.json'
    put(path, dict(samples=parts, graph_calls=calls, graph_accounting=dict(event_us=60.1*calls)))
    return path, dict(graph_calls=calls, measured_graphs=1, timing_protocol='one_profiled_graph', event_mean_us=60.1,
                      mean_us=60.1, p50_us=float(np.median(durations)), samples_us=durations)


def test_measured_boundary_uses_graph_event_mean_and_joined_kernel_span(tmp_path):
    path, timing = profile(tmp_path)
    data = main.main_times(tmp_path, timing)
    assert data == dict(graph=(60.1,34.5), LI1_to_LI3=(9,9), final_topk_through_tail=(16.5,16.5))
    obj = main.read(path)
    obj['samples'][0]['accounting']['intervals_us']['resolve_update'] = [[33, 38]]
    put(path, obj)
    with pytest.raises(ValueError, match='mainline parallel terminal'):
        main.main_times(tmp_path, timing)


def test_mismatched_event_and_profiler_time_is_rejected(tmp_path):
    _, timing = profile(tmp_path)
    timing['event_mean_us'] = timing['mean_us'] = 34.5
    with pytest.raises(ValueError, match='Graph event mean'):
        main.main_times(tmp_path, timing)


def test_ten_call_graph_preserves_event_and_rejects_other_denominators(tmp_path):
    _, timing = profile(tmp_path, 10)
    assert main.main_times(tmp_path, timing, expected_calls=10)['graph'] == (60.1, 34.0)
    with pytest.raises(ValueError, match='20 independent'):
        main.main_times(tmp_path, timing)
    timing['mean_us'] = timing['event_mean_us'] = 30.05
    with pytest.raises(ValueError, match='Graph event mean'):
        main.main_times(tmp_path, timing, expected_calls=10)
