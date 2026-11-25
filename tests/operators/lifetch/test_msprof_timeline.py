# SPDX-License-Identifier: Apache-2.0
"""Regression tests for rejecting incomplete A5 fused msprof exports."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'experiments/lifetch_mr23_service'))
from lifetch_msprof_timeline import (  # noqa: E402
    PHASES, PHASES_COLD_P4, PHASES_DEDICATED, PHASES_UNIFIED, device_business_view, publish_business_views,
    main, require_fused_replay,
)


def hardware(name, tick, task, duration=0, *, task_type=None, stream=17):
    args = {'Physic Stream Id': stream, 'Task Id': task}
    if task_type:
        args['Task Type'] = task_type
    return dict(name=name, ph='X', ts=str(tick), dur=duration, pid=1, tid=17, args=args)


def fused_replay(phases=PHASES):
    events = []
    for index, (stage, kernel, _) in enumerate(phases):
        start, first = 1_000 + 40 * index, 3 * index
        for suffix, tick, task in (('BEGIN', start, first), ('END', start + 30, first + 2)):
            marker = f'KVIP_LIFETCH_{stage}_{suffix}'
            events.append(hardware(marker, tick, task))
            events.append(dict(name=marker, ph='X', ts=str(tick - 9), dur=0, args={'mark_id': task}))
        events.append(hardware(kernel, start + 4, first + 1, 20, task_type='MIX_AIC'))
    return events


def test_scoped_cann_export_adds_four_business_phases_without_changing_raw_events():
    raw = fused_replay()
    original = deepcopy(raw)
    stages = require_fused_replay(raw)
    assert [stage['kernel'] for stage in stages] == [kernel for _, kernel, _ in PHASES]
    assert {stage['duration_us'] for stage in stages} == {20.0}
    assert {stage['task_identity'] for stage in stages} == {'named_device_task'}
    view = device_business_view(raw, stages)
    assert raw == original
    assert [event['name'] for event in view['traceEvents'] if event.get('cat') == 'KVIP fused kernel'] == [
        name for _, _, name in PHASES
    ]


def test_handoffs_split_at_device_marks_and_include_only_owner_stream_tasks():
    raw = fused_replay()
    raw.extend([
        hardware('same-stream event', 1_032, 90, 2, task_type='EVENT'),
        hardware('other stream event', 1_032, 91, 2, task_type='EVENT', stream=19),
    ])
    view = device_business_view(raw, require_fused_replay(raw))
    handoffs = view['kvip_inter_kernel_handoffs']
    assert len(handoffs) == 3
    first = handoffs[0]
    assert (first['begin_us'], first['end_us'], first['duration_us']) == (1_024, 1_044, 20)
    assert [segment['duration_us'] for segment in first['segments']] == [6, 10, 4]
    assert [task['name'] for task in first['observed_stream_tasks']] == [
        'KVIP_LIFETCH_HOT_END', 'same-stream event', 'KVIP_LIFETCH_THRESHOLD_BEGIN',
    ]
    assert sum(event.get('cat') == 'KVIP kernel handoff' for event in view['traceEvents']) == 3


def test_labels_existing_scoped_trace_without_overwriting_it(tmp_path):
    raw_path = tmp_path / 'msprof_old.json'
    output = tmp_path / 'annotated.json'
    raw = fused_replay(PHASES_DEDICATED)
    raw_path.write_text(json.dumps(raw))
    main([str(raw_path), '--route', 'dedicated', '--out', str(output)])
    assert json.loads(raw_path.read_text()) == raw
    assert len(json.loads(output.read_text())['kvip_inter_kernel_handoffs']) == 3


def test_batch_put_only_and_leaked_initialization_are_rejected():
    batch_put = hardware('kvaf_execute_batch_put_kernel', 300, 1, 8, task_type='AI_VECTOR_CORE')
    with pytest.raises(ValueError, match='BatchPut'):
        require_fused_replay([batch_put])
    with pytest.raises(ValueError, match='BatchPut'):
        require_fused_replay([batch_put, *fused_replay()])


def test_raw_host_marks_cannot_mask_a_missing_device_marker():
    raw = fused_replay()[1:]
    with pytest.raises(ValueError, match='missing device markers'):
        require_fused_replay(raw)


def test_phase_requires_a_kernel_task_inside_its_own_marked_stream():
    raw = fused_replay()
    raw[-1]['args']['Physic Stream Id'] = 5
    with pytest.raises(ValueError, match='outside its device marks'):
        require_fused_replay(raw)


def test_cold_p4_aliases_qualify_only_on_the_requested_route(tmp_path):
    raw = fused_replay(PHASES_COLD_P4)
    stages = require_fused_replay(raw, cold_route='cold_p4')
    assert [stage['kernel'] for stage in stages] == [kernel for _, kernel, _ in PHASES_COLD_P4]
    assert {stage['task_identity'] for stage in stages} == {'named_device_task'}
    with pytest.raises(ValueError, match='different named kernel'):
        require_fused_replay(raw)
    with pytest.raises(ValueError, match='different named kernel'):
        require_fused_replay(fused_replay(), cold_route='cold_p4')

    source = {'traceEvents': [dict(ph='X', args=dict(marked_lanes=4, kernel_entry=kernel))
                              for _, kernel, _ in PHASES_COLD_P4]}
    source_path, _ = publish_business_views(
        tmp_path, raw, source, stages, lambda path, payload: path.write_text(json.dumps(payload)),
        cold_route='cold_p4',
    )
    assert json.loads(source_path.read_text())['traceEvents'] == source['traceEvents']


def test_generic_a5_task_is_bound_by_unique_device_id_and_marker_range():
    raw = fused_replay()
    raw[-1]['name'] = 'FUSION'
    raw[-1]['args']['Task Type'] = 'FUSION'
    assert require_fused_replay(raw)[-1]['task_identity'] == 'unique_compute_task_between_marks'
    other = hardware('AIV_SQE', 1130, 42, 6, task_type='AIV_SQE')
    with pytest.raises(ValueError, match='no unique compute task'):
        require_fused_replay([*raw, other])


def test_unified_score_and_cold_t2_must_match_their_route():
    raw = fused_replay(PHASES_UNIFIED)
    for route in ('full_p2', 'full_uniform'):
        stages = require_fused_replay(raw, cold_route=route)
        assert [stage['kernel'] for stage in stages] == [kernel for _, kernel, _ in PHASES_UNIFIED]
    with pytest.raises(ValueError, match='different named kernel'):
        require_fused_replay(raw)


def test_dedicated_vendor_kernels_require_the_dedicated_route():
    raw = fused_replay(PHASES_DEDICATED)
    stages = require_fused_replay(raw, cold_route='dedicated')
    assert stages[2]['kernel'] == 'v9_li_kernel_cold_dedicated_service'
    with pytest.raises(ValueError, match='different named kernel'):
        require_fused_replay(raw)


def test_published_source_and_vendor_views_keep_their_clocks(tmp_path):
    raw = fused_replay()
    source = {'traceEvents': [dict(name=name, ph='X', ts=index, dur=0.8,
                                   args=dict(marked_lanes=4, kernel_entry=kernel))
                              for index, (_, kernel, name) in enumerate(PHASES)]}
    stages = require_fused_replay(raw)
    source_path, device_path = publish_business_views(
        tmp_path, raw, source, stages, lambda path, payload: path.write_text(json.dumps(payload)),
    )
    assert source_path.name == 'lifetch_business_trace.json'
    assert json.loads(source_path.read_text())['traceEvents'] == source['traceEvents']
    device = json.loads(device_path.read_text())['traceEvents']
    assert device[:len(raw)] == raw
    kernel_track = [event for event in device if event.get('cat') == 'KVIP fused kernel']
    assert kernel_track[-1]['ts'] == stages[-1]['begin_us']
    handoffs = json.loads((tmp_path / 'lifetch_handoff_report.json').read_text())
    assert [row['duration_us'] for row in handoffs['native_device_handoffs']] == [20, 20, 20]
    assert handoffs['source_business_handoffs'] == []
