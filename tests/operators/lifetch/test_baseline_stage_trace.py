# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the MR12-style P2 source-trace buffer and timeline."""

from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent / 'support'))
from baseline_stage_trace import (  # noqa: E402
    decode, render, verify_route, CAPACITY, EVENT, HEADER_BYTES, LANE_BYTES, LANES, MAGIC, REQUIRED, VERSION,
)


def sample_buffer():
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
            for phase in sorted(phases if lane == 0 else {1}):
                phase_begin = begin + count * 3 if phase != 1 else begin
                event = (phase_begin, begin + (800 if phase == 1 else 500), lane, kernel, phase, lane % 64,
                         int(lane in (1, 3)), 0, 0, 1, 0, 0)
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
