# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the MR12-style P2 source-trace buffer and timeline."""

import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent / 'support'))
import baseline_stage_trace as source_trace  # noqa: E402
from baseline_stage_trace import (  # noqa: E402
    decode, render, scoped_msprof, verify_route, CAPACITY, EVENT, HEADER_BYTES, LANE_BYTES, LANES, MAGIC,
    NO_QUERY, PIPE_SCOPES, REQUIRED, REQUIRED_PIPE, VERSION,
)


def sample_buffer(*, pipe_scopes=False):
    lane_stride = LANE_BYTES + CAPACITY * EVENT.size
    raw = bytearray(HEADER_BYTES + LANES * lane_stride)
    struct.pack_into('<Q4I', raw, 0, MAGIC, VERSION, LANES, CAPACITY, EVENT.size)
    for lane in (0, 1, 2, 3, 64, 65):
        base = HEADER_BYTES + lane * lane_stride
        count = 0
        for kernel, phases in REQUIRED.items():
            if lane >= 64 and kernel in (21, 23):
                continue
            begin = 3_000 + (kernel - 20) * 1_000 + lane
            selected = phases if lane == 0 else {1}
            if pipe_scopes:
                core_type = 'AIC' if lane >= 64 else 'AIV'
                selected = selected | {phase for phase in REQUIRED_PIPE[kernel] if PIPE_SCOPES[phase][0] == core_type}
            for phase in sorted(selected):
                phase_begin = begin + count * 3 if phase != 1 else begin
                detail = 0
                if phase in PIPE_SCOPES:
                    query_start = NO_QUERY if lane >= 64 else lane % 2 * 2
                    detail = (3 << 24) | (query_start << 20) | 127
                event = (phase_begin, begin + (800 if phase == 1 else 500), lane, kernel, phase, lane % 64,
                         int(lane in (1, 3)), 0, detail, 1, 0, 0)
                EVENT.pack_into(raw, base + LANE_BYTES + count * EVENT.size, *event)
                count += 1
        struct.pack_into('<2I', raw, base, count, 0)
    return raw


def test_trace_has_measured_lane_and_stage_envelopes():
    events = decode(sample_buffer())
    verify_route(events, batch=1)
    document, summaries = render(events, counter_mhz=1000)
    assert document['metadata']['counter_mhz'] == 1000
    spans = [entry for entry in document['traceEvents'] if entry['ph'] == 'X']
    hot_envelope = next(entry for entry in spans if entry['name'] == 'Hot Score')
    terminal_envelope = next(entry for entry in spans if entry['name'] == 'Parallel Terminal')
    assert hot_envelope['ts'] == 0
    assert hot_envelope['dur'] == 0.865
    assert hot_envelope['args']['marked_lanes'] == 6
    assert terminal_envelope['ts'] == 3
    assert terminal_envelope['args']['marked_lanes'] == 4
    assert {record['logical_lanes'] for record in summaries if record['phase'] == 'Kernel body'} == {4, 6}
    assert any(entry['pid'] == 1 and entry['tid'] == 64 for entry in spans)


def test_overflow_and_incomplete_events_are_rejected():
    raw = sample_buffer()
    struct.pack_into('<I', raw, HEADER_BYTES + 4, 1)
    with pytest.raises(ValueError, match='overflow'):
        decode(raw)
    struct.pack_into('<I', raw, HEADER_BYTES + 4, 0)
    struct.pack_into('<I', raw, HEADER_BYTES + LANE_BYTES + 3 * 8 + 6 * 4, 0)
    with pytest.raises(ValueError, match='Incomplete'):
        decode(raw)


def test_zero_tick_substage_is_preserved():
    raw = sample_buffer()
    offset = HEADER_BYTES + LANE_BYTES + EVENT.size
    begin, *_ = EVENT.unpack_from(raw, offset)
    struct.pack_into('<Q', raw, offset + 8, begin)
    events = decode(raw)
    verify_route(events, batch=1)
    document, _ = render(events, counter_mhz=1000)
    assert any(event.get('name') == 'Kernel setup' and event.get('dur') == 0.0
               for event in document['traceEvents'])


def test_missing_stage_duplicate_capture_or_underfilled_lane_fails():
    events = decode(sample_buffer())
    with pytest.raises(ValueError, match='Missing required source stages'):
        verify_route([event for event in events if event['phase'] != 160], batch=1)
    with pytest.raises(ValueError, match='Multiple marked invocations'):
        verify_route([*events, events[0]], batch=1)
    with pytest.raises(ValueError, match='marked lanes'):
        verify_route([event for event in events if event['lane'] != 64], batch=1)


def test_pipe_scopes_identify_sampled_tiles_and_core_lanes():
    events = decode(sample_buffer(pipe_scopes=True))
    verify_route(events, batch=1, require_pipe_scopes=True)
    with pytest.raises(ValueError, match='Missing required pipe scopes'):
        verify_route([event for event in events if event['phase'] != 184], batch=1, require_pipe_scopes=True)

    document, summaries = render(events, counter_mhz=1000)
    scopes = [event for event in document['traceEvents'] if event.get('cat') == 'KVIP source MTE1']
    assert all(event['args']['request_row'] == 3 and event['args']['tile_index'] == 127 for event in scopes)
    assert all('query_start' not in event['args'] and event['args']['core_type'] == 'AIC' for event in scopes)
    assert {event['args']['raw_core_id_hex'] for event in scopes} == {'0x40', '0x41'}
    for row in summaries:
        if row['pipe'] == 'MTE1':
            assert row['logical_aic_lanes'] == 2 and row['logical_aiv_lanes'] == 0
            assert row['observed_aic_lanes'] == [
                dict(block=0, subblock=0, raw_core_ids=[64]),
                dict(block=1, subblock=0, raw_core_ids=[65]),
            ]
    counts = [event['args']['lanes'] for event in document['traceEvents']
              if event['ph'] == 'C' and event['name'] == 'AIC MTE1 source lanes']
    assert max(counts) == 2 and counts[-1] == 0


def test_source_lane_count_deduplicates_nested_scopes():
    spans = [dict(begin=10, end=30, lane=0), dict(begin=20, end=40, lane=0), dict(begin=25, end=35, lane=1)]
    assert source_trace.source_lane_counts(spans) == [(10, 1), (25, 2), (35, 1), (40, 0)]


def test_external_instruction_report_records_stock_launch_plan_only(tmp_path, monkeypatch):
    monkeypatch.setattr(source_trace, 'replay_evidence', lambda *_: dict(correctness='PASS', total_prefetch=9))
    shape = SimpleNamespace(liCores=32)
    state = SimpleNamespace(batch=16, rotating=SimpleNamespace(runtime_shape=shape))
    bridge = tmp_path / 'libv9_bridge.so'
    result = source_trace.write_instruction_replay(tmp_path, state, bridge, None, 'Ascend950')
    assert json.loads((tmp_path / 'instruction_replay_report.json').read_text()) == result
    assert result['invocations'] == 1 and result['total_prefetch'] == 9
    assert result['source_trace_enabled'] is False and result['inner_profiler_enabled'] is False
    assert result['kernel_launch_plan'] == [
        dict(kernel_entry='v9_li_kernel', launches=1, blocks=32),
        dict(kernel_entry='v9_li_kernel_1', launches=1, blocks=32),
        dict(kernel_entry='v9_li_kernel_2', launches=1, blocks=32),
        dict(kernel_entry='v9_parallel_terminal', launches=1, blocks=64),
    ]


def fake_profiler(calls, *, failed=None):
    def operation(name, result=0):
        def call(*arguments):
            calls.append((name, *arguments))
            return 99 if failed == name else result
        return call

    return SimpleNamespace(
        init=operation('init'), create_config=operation('create_config', 17),
        start=operation('start'), stop=operation('stop'),
        destroy_config=operation('destroy_config'), finalize=operation('finalize'),
    )


def test_msprof_wraps_only_the_requested_device_replay(tmp_path):
    calls = [('KVAF BatchPut',), ('setup stream fence',)]
    output = tmp_path / 'raw'
    with scoped_msprof(fake_profiler(calls), output, device_id=5) as report:
        calls.extend([('marked P2 forward',), ('terminal stream fence',)])
    calls.append(('source trace readback',))
    assert calls == [
        ('KVAF BatchPut',), ('setup stream fence',), ('init', str(output)),
        ('create_config', [5], 0, 0, 0x0183), ('start', 17),
        ('marked P2 forward',), ('terminal stream fence',),
        ('stop', 17), ('destroy_config', 17), ('finalize',), ('source trace readback',),
    ]
    assert report['status'] == dict(init=0, start=0, stop=0, destroy_config=0, finalize=0)


@pytest.mark.parametrize('failed, expected', [
    ('start', ['init', 'create_config', 'start', 'destroy_config', 'finalize']),
    ('stop', ['init', 'create_config', 'start', 'replay', 'stop', 'destroy_config', 'finalize']),
])
def test_msprof_cleans_up_acquired_resources_before_returning_errors(tmp_path, failed, expected):
    calls = []
    with pytest.raises(RuntimeError, match=rf'acl\.prof\.{failed} failed with status 99'):
        with scoped_msprof(fake_profiler(calls, failed=failed), tmp_path, device_id=5):
            calls.append(('replay',))
    assert [call[0] for call in calls] == expected
