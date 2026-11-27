# SPDX-License-Identifier: Apache-2.0
"""Check the per-core ring decoder and preservation of the paired trace tracks."""

import json
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / 'support'))
from fixed_q4_pressure_msprof import PHASES, _label_capture, _source_comparison
from fixed_q4_staged_trace import CAPACITY, EVENT, HEADER_BYTES, LANE_BYTES, LANES, MAGIC, VERSION, decode, render, verify


def test_staged_ring_preserves_core_roles_and_detects_overflow():
    stride = LANE_BYTES + CAPACITY * EVENT.size
    data = bytearray(HEADER_BYTES + LANES * stride)
    struct.pack_into('<Q4I', data, 0, MAGIC, VERSION, LANES, CAPACITY, EVENT.size)
    for lane, phase in ((0, 191), (64, 184)):
        offset = HEADER_BYTES + lane * stride
        struct.pack_into('<2I', data, offset, 1, 0)
        EVENT.pack_into(data, offset + LANE_BYTES, 100, 105, 78, 22, phase, 0, 0, 1, 0, 1, 0, 0)
    result = decode(data)
    assert [(item['core_type'], item['phase'], item['core']) for item in result] == [
        ('AIV', 191, 78), ('AIC', 184, 78),
    ]
    struct.pack_into('<2I', data, HEADER_BYTES + 64 * stride, 1, 1)
    with pytest.raises(ValueError, match='overflow at lane 64'):
        decode(data)


@pytest.mark.parametrize('marked', (False, True))
def test_pair_preserves_vendor_payload_and_all_source_lane_events(tmp_path, marked):
    kernels = ('v9_li_fixed_hot', 'v9_li_kernel_1', 'v9_li_fixed_cold', 'v9_parallel_terminal')
    timeline = tmp_path / 'msprof_1.json'
    sentinel = dict(ph='X', ts=1001, dur=0.1, pid=9, tid=2, name='ACL wait', cat='ACL', args={})
    vendor = [sentinel, *(dict(name=kernel, ph='X', cat='Kernel', pid=10, tid=2,
                              ts=1000 + n * 20, dur=8, args={}) for n, kernel in enumerate(kernels))]
    if marked:
        for index, (stage, _, _) in enumerate(PHASES):
            for side, delta in (('BEGIN', -1), ('END', 9)):
                vendor.append(dict(ph='i', s='t', ts=1000 + index * 20 + delta, pid=10, tid=2,
                                   name=f'KVIP_LIFETCH_{stage}_{side}',
                                   args={'Physic Stream Id': 2, 'Task Id': index * 2 + (side == 'END')}))
        for task in vendor[1:5]:
            task['args']['Physic Stream Id'] = 2
    timeline.write_text(json.dumps(dict(traceEvents=vendor)))
    source = [dict(lane=0, block=0, subblock=0, core=78, phase=1,
                   kernel=k, begin=100 + 50 * k, end=120 + 50 * k, detail=0,
                   core_type='AIV', iteration=0) for k in range(20, 24)]
    source.append(dict(source[2], phase=192, iteration=2, begin=1201, end=1209))
    captures = {}
    for label in ('baseline', 'optimized'):
        path = tmp_path / label
        path.mkdir()
        capture = _label_capture([timeline], path, label, {}, lambda p, x: p.write_text(json.dumps(x)))
        assert capture['attribution'] == 'VERIFIED'
        rendered = json.loads(Path(capture['trace']).read_text())
        assert rendered['traceEvents'][:len(vendor)] == vendor
        assert len(capture['device_handoffs']) == 3
        assert capture['attribution_source'] == ('device_marks' if marked else 'kernel_names')
        if marked:
            handoff = capture['device_handoffs'][0]
            assert (handoff['kernel_to_end_mark_us'], handoff['between_marks_us'],
                    handoff['begin_mark_to_kernel_us'], handoff['stream_id']) == (1, 10, 1, '2')
        document, _ = render(source, 50)
        assert any(event.get('args', {}).get('phase_id') == 192 for event in document['traceEvents'])
        business = path / 'lifetch_business_trace.json'
        business.write_text(json.dumps(document))
        captures[label] = dict(capture, arm=label, business_trace=str(business))
    compared = _source_comparison(captures)
    spans = [event for event in compared['traceEvents'] if event.get('args', {}).get('phase_id') == 192]
    assert len(spans) == 2 and len({event['pid'] for event in spans}) == 2
    assert spans[0]['ts'] == spans[1]['ts']


def test_four_phase_only_capture_is_rejected():
    with pytest.raises(ValueError, match='Hot Score missing staged scopes'):
        verify([], batch=4, windows=4, service_lanes=4, hot_groups=32, cold_groups=32, prefetch=1, final_get=1)
