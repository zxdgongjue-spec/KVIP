# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / 'support'))
import pr_head_parity_report as report


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize('source_id', [dict(reference_kind='native-only', ids_sha256='serial'),
                                       dict(serial_set_digest='serial')])
def test_block_checks_capacity_and_uses_device_event_separately_from_span(tmp_path, monkeypatch, source_id):
    cap = 131328
    group = 7
    order = list(report.REVISIONS)
    measured = {}
    monkeypatch.setattr(report, 'identity', lambda root: {'host': 'b20'})
    ends = [[130000 + q for q in range(4)]] * 16
    source = dict(effective_native_ends=ends, identity={'inputs': {'query_ends': 'ends'}})
    receipt = dict(native_reference=source, **source_id)
    monkeypatch.setattr(report, 'oracle_for', lambda owner, g: (None, receipt, {}))
    monkeypatch.setattr(report, 'case_folder', lambda root, g, arm: root/f'g{g}'/arm/'fused')
    monkeypatch.setattr(report, 'measured_parallel', lambda root, g, arm, *a, **k: measured[arm])
    monkeypatch.setattr(report.subprocess, 'check_output', lambda args, text: report.REVISIONS[Path(args[2]).name])
    def row(block, name, index):
        folder = block/f'g7/{name}/fused'
        capture = dict(geometry=dict(batch=16, history=cap, token_bytes=656, resident=16384), page_count=128)
        put(folder.parent / 'replay_input.json', capture)
        put(folder.parent / 'capture_input.json', capture)
        call = dict(prefetch=[0]*16, final=[[0]*4]*16, cqe=[[0]*4]*16)
        put(folder/'ablation_receipts.json', dict(policy='paced', independent_status_allocations=10,
                                                 measured=[call]*10, detail=[call]*10))
        sample = lambda i: dict(us=100+i, accounting=dict(
            intervals_us={'li': [(200*i, 200*i+80)], 'topk_resolve_update': [(200*i+80, 200*i+100+i)]},
            stages_us={'li': 80, 'topk_resolve_update': 20+i}, inter_kernel_gap_us=0))
        event = 205 + index
        put(folder/'diagnose/round-1/full_attribution.json', dict(samples=[sample(i) for i in range(10)],
            graph_calls=10, graph_accounting={'event_us': 10*event}))
        history = dict(capture_history=cap, bound_history=cap, effective_ends=ends, query_ends_sha256='ends',
                       captured_page_count=128, bound_page_count=128, capture_page_table_sha256='pages',
                       capture_state_sha256=dict.fromkeys(('captured_hot_valid', 'captured_lru', 'tail_bounds',
                           'logical_address_valid', 'logical_score_keys', 'logical_score_valid'), 'digest'),
                       measured_requests={'prefetch': [0]*10, 'final': [0]*10})
        proof = dict(runtime={'host': 'b20','visible': '4'}, independent_buffers=dict.fromkeys(
            ('kv', 'index', 'winners', 'window'), 10), history_cap=history, production_python=str(Path(name).resolve()/'python/kvip/__init__.py'),
            capture_operand_hashes={'query_ends': 'ends'}, library_sha256=name)
        thp = dict(bytes=1200, coverage_min_pct=100, ub_device='udma0', numa_pages_overlapping_vmas={'3': 10})
        measured[name]=(dict(graph=(event,104.5), LI1_to_LI3=(80,80), final_topk_through_tail=(24.5,24.5)),
                        np.zeros(640, dtype=bool), thp, proof, dict(history=cap, production_root=name))
    for block_id, names in ((1, order), (2, list(reversed(order)))):
        block = tmp_path/f'block{block_id}'
        for i, name in enumerate(order):
            row(block, name, i)
        (block/'g7/order.txt').write_text(' '.join(names))
        report.report(block, group, tmp_path/'owner')
    rows=[json.loads((tmp_path/f'block{i}/g7/pr_head_parity.json').read_text()) for i in (1,2)]
    combined=report.summarize_pair(group,*rows)
    assert combined['means']['pr31']['event_per_call'] == 207
    assert combined['means']['pr31']['kernel_span'] == 104.5
    assert combined['kernel_span_p50_us']['pr31'] == 104.5
    assert combined['deltas']['pr31_minus_pr30']['event_per_call'] == 2
    assert rows[0]['arms']['pr31']['spans_us'] == list(range(100,110))
    rows[1]['order'] = order
    with pytest.raises(ValueError,match='reverse-order'):
        report.summarize_pair(group,*rows)
    measured['pr31'][3]['history_cap']['bound_history']=524544
    with pytest.raises(ValueError,match='capacity mismatch'):
        report.report(tmp_path/'block1', group, tmp_path/'owner')
