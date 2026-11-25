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
    NO_QUERY, PIPE_SCOPES, REQUIRED, REQUIRED_PIPE, VERSION, WAIT_PARTITIONED_HALF,
)


def sample_buffer(*, pipe_scopes=False, batch=1, cold_parts=2, score_parts=2, unified=False):
    lane_stride = LANE_BYTES + CAPACITY * EVENT.size
    raw = bytearray(HEADER_BYTES + LANES * lane_stride)
    struct.pack_into('<Q4I', raw, 0, MAGIC, VERSION, LANES, CAPACITY, EVENT.size)
    cold_blocks = batch * cold_parts
    score_blocks = batch * score_parts
    total_blocks = max(score_blocks, cold_blocks)
    for lane in (*range(2 * total_blocks), *range(64, 64 + total_blocks)):
        base = HEADER_BYTES + lane * lane_stride
        count = 0
        for kernel, phases in REQUIRED.items():
            active_blocks = {20: score_blocks, 22: cold_blocks}.get(kernel, 2 * batch)
            if lane not in range(2 * active_blocks) and lane not in range(64, 64 + active_blocks):
                continue
            if lane >= 64 and (kernel in (21, 23) or (unified and kernel == 22)):
                continue
            begin = 3_000 + (kernel - 20) * 1_000 + lane
            selected = phases if lane == 0 else {1}
            if unified and kernel == 22:
                selected = selected - {21}
            if pipe_scopes:
                core_type = 'AIC' if lane >= 64 else 'AIV'
                pipe_phases = source_trace.T2_PIPE_PHASES if unified and kernel == 22 else REQUIRED_PIPE[kernel]
                selected = selected | {phase for phase in pipe_phases if PIPE_SCOPES[phase][0] == core_type}
            for phase in sorted(selected):
                phase_begin = begin + count * 3 if phase != 1 else begin
                detail = 0
                if phase in PIPE_SCOPES:
                    query_start = NO_QUERY if lane >= 64 else lane % 2 * 2
                    detail = (3 << 24) | (query_start << 20) | 127
                windows = (1, 2, 3) if phase in source_trace.T2_PIPE_PHASES else (0,)
                for window in windows:
                    event = (phase_begin, begin + (800 if phase == 1 else 500), lane, kernel, phase, lane % 64,
                             int(lane in (1, 3)), window, detail, 1, 0, 0)
                    EVENT.pack_into(raw, base + LANE_BYTES + count * EVENT.size, *event)
                    count += 1
            if cold_parts == 4 and kernel == 22 and lane == 0:
                for epoch in (1, 2, 3):
                    begin_wait = begin + epoch * 100
                    detail = (3 << 24) | (2 << 20) | 2
                    event = (begin_wait, begin_wait + 10, lane, kernel, WAIT_PARTITIONED_HALF, lane,
                             0, epoch, detail, 1, 0, 0)
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


def test_handoff_counts_cold_cursor_reset_on_the_threshold_source_clock():
    events = decode(sample_buffer())
    template = next(e for e in events if e['kernel'] == 21 and e['lane'] == 0)
    events.extend([
        dict(template, phase=source_trace.COLD_PROGRESS_RESET, begin=3_700, end=3_810),
        dict(template, phase=source_trace.COLD_PROGRESS_RESET, lane=1, block=1, begin=3_730, end=3_840),
    ])
    document, _ = render(events, counter_mhz=1000, cold_route='dedicated')
    threshold = next(e for e in document['traceEvents']
                     if e.get('cat') == 'Kernel source envelope' and e['name'] == 'Resident threshold')
    assert threshold['args']['begin_tick'] == 3_700
    handoffs = document['metadata']['source_handoffs']
    assert len(handoffs) == 3
    reset = next(e for e in handoffs[0]['observed_scopes'] if e['phase'].startswith('Clear Cold progress'))
    assert reset['duration_on_any_lane_us'] == 0.14
    assert handoffs[0]['uncovered_source_us'] == pytest.approx(0.518 - 0.140 - 0.003)
    assert [x['tick'] for x in handoffs[0]['boundaries'].values()] == [3500, 3865, 4000, 4018]
    assert [x['duration_us'] for x in handoffs[0]['body_segments']] == pytest.approx([0.365, 0.135, 0.018])
    assert sum(x['uncovered_source_us'] for x in handoffs[0]['body_segments']) == pytest.approx(
        handoffs[0]['uncovered_source_us'])
    assert any(e.get('cat') == 'KVIP business handoff' for e in document['traceEvents'])
    assert any(e.get('cat') == 'KVIP source handoff breakdown' for e in document['traceEvents'])


def test_overlapping_recorded_kernel_bodies_are_counted_once():
    pieces = source_trace.split_recorded_bodies(10, 90, 60, 40, 10, [(10, 50), (45, 60)])
    assert [(x['name'], x['begin_tick'], x['end_tick'], x['uncovered_source_us']) for x in pieces] == [
        ('Previous kernel tail', 10, 40, 0),
        ('Overlapping recorded kernel bodies', 40, 60, 0),
        ('Following kernel entry', 60, 90, 3),
    ]
    assert sum(x['duration_us'] for x in pieces) == 8
    assert not source_trace.split_recorded_bodies(10, 90, None, 40, 10, [])
    assert not source_trace.split_recorded_bodies(90, 10, 60, 40, 10, [])


def test_handoff_gaps_are_drawn_on_each_logical_lane_without_covering_work():
    # Raw group IDs can be shared. The per-lane comparison uses logical IDs.
    events = [dict(event, core=0) for event in decode(sample_buffer())]
    resident = next(e for e in events if e['kernel'] == 21 and e['lane'] == 0)
    events.append(dict(resident, phase=source_trace.COLD_PROGRESS_RESET, begin=3900, end=3910))
    document, _ = render(events, counter_mhz=1000)
    gaps = [e for e in document['traceEvents'] if e.get('cat') == 'KVIP source lane handoff']
    assert gaps and all(e['dur'] > 0 and e['args']['derived'] for e in gaps)
    for gap in gaps:
        args = gap['args']
        assert gap['pid'] == (1 if gap['tid'] >= 64 else 2)
        assert not any(e['lane'] == gap['tid'] and
                       max(e['begin'], args['begin_tick']) < min(e['end'], args['end_tick'])
                       for e in events)

    def intervals(handoff, lane):
        return [(e['args']['begin_tick'], e['args']['end_tick']) for e in gaps
                if e['args']['handoff'].startswith(handoff) and e['tid'] == lane]

    assert intervals('Resident Top2048', 64) == [(4500, 4803), (4803, 5000), (5000, 5027)]
    assert intervals('Resident Top2048', 0) == [(4800, 4803), (4803, 5000)]
    assert intervals('Hot score', 0) == [(3800, 3865), (3865, 3900), (3910, 4000)]


def test_handoff_window_state_reports_epoch_lane_and_resident_query():
    events = decode(sample_buffer())
    template = next(e for e in events if e['kernel'] == 22 and e['lane'] == 0)
    events.append(dict(template, phase=243, iteration=2, detail=7))
    resident = next(e for e in events if e['kernel'] == 21 and e['lane'] == 0)
    events.append(dict(resident, phase=242, detail=(3 << 24) | (2 << 20) | source_trace.NO_TILE))
    document, _ = render(events, counter_mhz=1000)
    scope = next(e for e in document['traceEvents'] if e.get('name') == 'Stage window state and ready cursors')
    assert scope['args']['window_epoch'] == 2
    assert scope['args']['service_lane'] == 7
    seed = next(e for e in document['traceEvents'] if e.get('name') == 'Publish initial window Top2048 state')
    assert (seed['args']['request_row'], seed['args']['query']) == (3, 2)


def test_dynamic_service_job_and_handoff_detail_share_source_timeline():
    events = decode(sample_buffer())
    template = next(e for e in events if e['kernel'] == 22 and e['lane'] == 0)
    events.extend([
        dict(template, phase=source_trace.DYNAMIC_SERVICE_JOB, detail=(3 << 4) | (2 << 2) | (9 << 8)),
        dict(template, phase=243, iteration=2, detail=3),
    ])
    document, _ = render(events, counter_mhz=1000, cold_route='dynamic')
    job = next(e for e in document['traceEvents'] if e.get('name') == 'Dynamically assigned Service window')
    assert (job['args']['request_row'], job['args']['window_epoch'], job['args']['job_ticket']) == (3, 2, 9)
    assert any(e.get('cat') == 'KVIP source handoff breakdown' for e in document['traceEvents'])


def test_fresh_capture_requires_handoff_scopes_but_old_capture_still_decodes():
    events = decode(sample_buffer())
    verify_route(events, batch=1)
    with pytest.raises(ValueError, match='Missing required handoff scopes'):
        verify_route(events, batch=1, require_handoff_scopes=True)
    for kernel, phases in source_trace.REQUIRED_HANDOFF.items():
        template = next(e for e in events if e['kernel'] == kernel and e['lane'] == 0)
        events.extend(dict(template, phase=phase) for phase in phases)
    verify_route(events, batch=1, require_handoff_scopes=True)
    with pytest.raises(ValueError, match=r'245'):
        verify_route([e for e in events if e['phase'] != 245], batch=1, require_handoff_scopes=True)

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


def test_cold_p4_uses_native_alias_and_separate_part_waits():
    events = decode(sample_buffer(pipe_scopes=True, batch=8, cold_parts=4))
    verify_route(events, batch=8, require_pipe_scopes=True, cold_route='cold_p4')
    with pytest.raises(ValueError, match='Unexpected WaitPartitionedHalf'):
        verify_route(events, batch=8)
    missing_window = [event for event in events
                      if (event['phase'], event['iteration']) != (WAIT_PARTITIONED_HALF, 2)]
    with pytest.raises(ValueError, match='all three Cold windows'):
        verify_route(missing_window, batch=8, cold_route='cold_p4')

    document, stages = render(events, counter_mhz=1000, cold_route='cold_p4')
    trace = document['traceEvents']
    waits = [event for event in trace if event['name'] == 'WaitPartitionedHalf']
    assert len(waits) == 3
    for epoch, event in enumerate(waits, 1):
        args = event['args']
        assert args['phase_id'] == WAIT_PARTITIONED_HALF
        assert args['kernel_entry'] == 'v9_li_kernel_cold_partitioned'
        assert (args['request_row'], args['query_start'], args['cold_part'], args['window_epoch']) == (3, 2, 2, epoch)
        assert 'tile_index' not in args and 'sampled_scan_tile' not in args
    bodies = [event for event in trace if event.get('cat') == 'Kernel source envelope']
    assert {event['args']['kernel_entry']: event['args']['marked_lanes'] for event in bodies} == {
        'v9_li_kernel': 48, 'v9_li_kernel_threshold_cold_partitioned': 32,
        'v9_li_kernel_cold_partitioned': 96, 'v9_parallel_terminal': 32,
    }
    assert {stage['window_epoch'] for stage in stages if stage['phase_id'] == WAIT_PARTITIONED_HALF} == {1, 2, 3}


def test_full_score_records_partitioned_compute_and_aiv_only_cold_t2():
    events = decode(sample_buffer(pipe_scopes=True, batch=8, score_parts=4, unified=True))
    verify_route(events, batch=8, require_pipe_scopes=True, cold_route='full_uniform', score_parts=4)
    with pytest.raises(ValueError, match='Missing required pipe scopes'):
        verify_route([event for event in events if event['phase'] != 220], batch=8,
                     require_pipe_scopes=True, cold_route='full_uniform', score_parts=4)
    with pytest.raises(ValueError, match='expected window/lane coverage'):
        verify_route([event for event in events
                      if (event['phase'], event['lane'], event['iteration']) != (220, 0, 1)],
                     batch=8, require_pipe_scopes=True, cold_route='full_uniform', score_parts=4)
    document, stages = render(events, counter_mhz=1000, cold_route='full_uniform')
    bodies = {stage['kernel_entry']: stage['logical_lanes'] for stage in stages if stage['phase_id'] == 1}
    assert bodies['v9_li_kernel_score_t2'] == 96
    assert bodies['v9_li_kernel_cold_t2'] == 32
    sample = next(event for event in document['traceEvents'] if event['name'] == 'Cold T2 Score GM to UB with fence')
    assert sample['args']['query_start'] == sample['args']['block'] % 2 * 2
    assert sample['args']['sampled_scan_tile'] is True


def test_dedicated_wait_and_ready_lanes_are_rendered_separately():
    events = [event for event in decode(sample_buffer(pipe_scopes=True, batch=16))
              if not (event['kernel'] == 22 and event['lane'] >= 88)]
    for request in range(16):
        for window in range(4):
            scope = dict(lane=48 + request, core=request, kernel=22, block=48 + request,
                         subblock=request % 2, begin=5100, end=5110, iteration=window)
            if window:
                events.append(dict(scope, phase=source_trace.DEDICATED_DRIVER_WAIT, detail=request))
            for q in range(4):
                events.append(dict(scope, phase=source_trace.DEDICATED_READY_SNAPSHOT, detail=4 * request + q))
    verify_route(events, 16, require_pipe_scopes=True, cold_route='dedicated')
    verified, _ = render(events, 1000, cold_route='dedicated')
    wait = next(event for event in verified['traceEvents']
                if event['name'] == 'Dedicated driver wait' and event['tid'] == 63)
    assert wait['args']['service_lane'] == 15
    assert wait['args']['window_epoch'] == 1
    ready = next(event for event in verified['traceEvents']
                 if event['name'] == 'Dedicated ready cursor snapshot' and event['tid'] == 63)
    assert ready['args']['request_row'] == 15 and ready['args']['query'] == 0
    with pytest.raises(ValueError, match='incomplete lane/window coverage'):
        verify_route(events[:-1], 16, cold_route='dedicated')


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
