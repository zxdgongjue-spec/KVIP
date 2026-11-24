# SPDX-License-Identifier: Apache-2.0
"""Capture and decode the opt-in P2/Terminal source trace on an A5 device."""

from __future__ import annotations

import argparse
from collections import defaultdict
import ctypes
import hashlib
import math
from pathlib import Path
import struct

from reporting import atomic_json
from terminal_handoff import fixture_hash, request_counts

MAGIC = 0x4B56495054524331
VERSION = 1
LANES = 96
CAPACITY = 256
HEADER_BYTES = LANE_BYTES = 128
EVENT = struct.Struct('<3Q8IQ')
BINDING_OFFSET = 192

KERNELS = {
    20: ('Hot Score', 'v9_li_kernel'),
    21: ('Resident threshold', 'v9_li_kernel_1'),
    22: ('Cold Score + rotating Service', 'v9_li_kernel_2'),
    23: ('Parallel Terminal', 'v9_parallel_terminal'),
    24: ('Fallback Final', 'v9_li_kernel_3'),
}
PHASES = {
    1: 'Kernel body',
    2: 'Kernel setup',
    21: 'AIC QK matmul scan',
    22: 'AIV Score and classify scan',
    23: 'Resident Top2048 and threshold',
    25: 'Window advance and TopK publication',
    26: 'Fallback Final Top2048',
    27: 'AIC tail dependency wait',
    150: 'Rotating Service',
    151: 'Service predecessor wait',
    152: 'Service candidate union and dedup',
    153: 'Prefetch descriptor preparation',
    154: 'Prefetch WQE submission',
    155: 'Window per-Q TopK and classify',
    160: 'Terminal Top2048 and classify',
    161: 'Resolve and dedup',
    162: 'Resolve capacity pass',
    163: 'Publish output slots and logical IDs',
    164: 'Final Get WQE write',
    165: 'Final Get doorbell',
    166: 'Persistent Index update',
    167: 'Wait final Get CQ',
    168: 'Join prefetch CQ',
    169: 'Request join and completion status',
}
# Stages that must appear on this P2/W4 route. Get submission may be
# skipped without eligible requests. Extra capacity passes are conditional.
REQUIRED = {
    20: {1, 2, 21, 22},
    21: {1, 2, 23},
    22: {1, 2, 21, 22, 25, 150, 152},
    23: {1, 2, 160, 161, 163, 166, 169},
}
CLOCK = ('Device GetSystemCycle relative to the first recorded scope, scaled by an independent '
         'NPU Event calibration. msprof host and device task timestamps are separate.')


class StageBuffer:
    """Attach a private trace buffer only for one already-bound native call."""

    def __init__(self, state):
        library = state.forward.rotating.library
        probe = getattr(library, 'kvip_stage_trace_supported', None)
        if probe is None:
            raise RuntimeError('LIFetch library has no KVIP_STAGE_TRACE entry point')
        probe.argtypes, probe.restype = [], ctypes.c_uint32
        if probe() != VERSION:
            raise RuntimeError('LIFetch stage trace ABI version mismatch')
        self.workspace = state.forward.workspace
        if self.workspace.numel() != 256:
            raise RuntimeError('LIFetch streaming binding must be exactly 256 bytes')
        self.buffer = state.torch.zeros(HEADER_BYTES + LANES * (LANE_BYTES + CAPACITY * EVENT.size),
                                        dtype=state.torch.uint8, device=state.device)
        header = struct.pack('<Q4I', MAGIC, VERSION, LANES, CAPACITY, EVENT.size)
        self.buffer[:len(header)] = state.torch.tensor(list(header), dtype=state.torch.uint8, device=state.device)
        address = struct.pack('<Q', self.buffer.data_ptr())
        self.pointer = state.torch.tensor(list(address), dtype=state.torch.uint8, device=state.device)

    def attach(self):
        self.workspace[BINDING_OFFSET:BINDING_OFFSET + 8].copy_(self.pointer)

    def detach(self):
        self.workspace[BINDING_OFFSET:BINDING_OFFSET + 8].zero_()

    def read(self):
        return self.buffer.cpu().numpy().tobytes()


def decode(raw):
    if len(raw) < HEADER_BYTES:
        raise ValueError('Missing LIFetch stage trace header')
    magic, version, lanes, capacity, stride = struct.unpack_from('<Q4I', raw)
    if (magic, version, lanes, stride) != (MAGIC, VERSION, LANES, EVENT.size) or not 0 < capacity <= 65536:
        raise ValueError('Invalid LIFetch stage trace header')
    lane_stride = LANE_BYTES + capacity * EVENT.size
    if len(raw) != HEADER_BYTES + lanes * lane_stride:
        raise ValueError('LIFetch stage trace buffer size differs from header')

    events = []
    for lane in range(lanes):
        base = HEADER_BYTES + lane * lane_stride
        count, overflow = struct.unpack_from('<2I', raw, base)
        if overflow or count > capacity:
            raise ValueError(f'Stage trace overflow on lane {lane}, capacity={capacity}')
        for i in range(count):
            fields = EVENT.unpack_from(raw, base + LANE_BYTES + i * EVENT.size)
            begin, end, core, kernel, phase, block, subblock, iteration, detail, complete, _, _ = fields
            if complete != 1 or begin <= 0 or end < begin or (phase == 1 and end == begin):
                raise ValueError(f'Incomplete or invalid stage trace event on lane {lane}, slot {i}')
            if kernel not in KERNELS or phase not in PHASES or block != lane % 64:
                raise ValueError(f'Unexpected trace routing on lane {lane}, kernel={kernel}, phase={phase}, block={block}')
            events.append(dict(lane=lane, begin=begin, end=end, core=core, kernel=kernel, phase=phase,
                               block=block, subblock=subblock, iteration=iteration, detail=detail))
    if not events:
        raise ValueError('The device did not record any stage trace events')
    return sorted(events, key=lambda e: (e['begin'], -e['end'], e['lane']))


def verify_route(events, batch):
    phases = defaultdict(set)
    bodies = defaultdict(set)
    for event in events:
        kernel, phase = event['kernel'], event['phase']
        phases[kernel].add(phase)
        if phase == 1:
            lane = event['lane']
            if lane in bodies[kernel]:
                raise ValueError(f'Multiple marked invocations on kernel {kernel}, lane {lane}')
            bodies[kernel].add(lane)
    if set(phases) != set(REQUIRED):
        raise ValueError(f'Expected P2 Hot/Threshold/Cold/Terminal trace, got kernel IDs {sorted(phases)}')
    missing = {kernel: sorted(expected - phases[kernel]) for kernel, expected in REQUIRED.items()
               if not expected.issubset(phases[kernel])}
    if missing:
        raise ValueError(f'Missing required source stages: {missing}')
    four_aiv = set(range(4 * batch))
    two_aic = set(range(64, 64 + 2 * batch))
    expected = {20: four_aiv | two_aic, 21: four_aiv, 22: four_aiv | two_aic, 23: four_aiv}
    for kernel, lanes in expected.items():
        if bodies[kernel] != lanes:
            raise ValueError(f'{KERNELS[kernel][1]} expected {len(lanes)} marked lanes, '
                             f'got {len(bodies[kernel])}')


def render(events, counter_mhz):
    if not math.isfinite(counter_mhz) or counter_mhz <= 0:
        raise ValueError('A positive measured device counter frequency is required')
    origin = min(event['begin'] for event in events)
    trace = []

    def metadata(name, pid, tid, label):
        trace.append(dict(name=name, ph='M', pid=pid, tid=tid, args={'name': label}))

    def add_span(name, cat, pid, tid, begin, end, args):
        trace.append(dict(name=name, cat=cat, ph='X', pid=pid, tid=tid,
                          ts=(begin - origin) / counter_mhz, dur=(end - begin) / counter_mhz, args=args))

    for pid, name in [(0, 'Kernel envelopes | source scopes'), (1, 'AIC lanes'), (2, 'AIV lanes')]:
        metadata('process_name', pid, 0, name)
    kernel_bodies = defaultdict(list)
    stage_sets = defaultdict(list)
    seen_lanes = set()
    for event in events:
        lane = event['lane']
        pid = 1 if lane >= 64 else 2
        if lane not in seen_lanes:
            metadata('thread_name', pid, lane, f'{"AIC" if pid == 1 else "AIV"} logical lane {event["block"]}')
            seen_lanes.add(lane)
        begin, end, phase = event['begin'], event['end'], event['phase']
        kernel = event['kernel']
        kernel_stage, entry = KERNELS[kernel]
        args = dict(kernel_entry=entry, phase_id=phase, block=event['block'], subblock=event['subblock'],
                    raw_core_id=event['core'], iteration=event['iteration'], detail=event['detail'],
                    begin_tick=begin, end_tick=end)
        window_epoch = None
        if kernel in (20, 22) and phase in (21, 22, 25, *range(150, 156)):
            window_epoch = event['iteration']
            args['window_epoch'] = window_epoch
        if phase in (23, 155) or kernel == 23:
            args.update(request_row=event['detail'] // 4, query=event['detail'] % 4)
        elif phase in (21, 22, 152, 153):
            args['request_row'] = event['detail']
        elif phase in (25, 150, 151):
            args['service_lane'] = event['detail']
        elif phase == 154:
            args['request_or_service_lane'] = event['detail']
        add_span(PHASES[phase], kernel_stage, pid, lane, begin, end, args)
        if phase == 1:
            kernel_bodies[kernel].append(event)
        stage_sets[(kernel, phase, window_epoch)].append(event)

    def bounds(spans):
        return min(e['begin'] for e in spans), max(e['end'] for e in spans)

    for kernel, spans in sorted(kernel_bodies.items()):
        name, entry = KERNELS[kernel]
        metadata('thread_name', 0, kernel, entry)
        begin, end = bounds(spans)
        add_span(name, 'Kernel source envelope', 0, kernel, begin, end,
                 dict(kernel_entry=entry, marked_lanes=len(spans), begin_tick=begin, end_tick=end))

    summaries = []
    for (kernel, phase, window_epoch), spans in sorted(stage_sets.items()):
        begin, end = bounds(spans)
        summaries.append(dict(kernel_entry=KERNELS[kernel][1], phase=PHASES[phase], window_epoch=window_epoch,
                              events=len(spans), logical_lanes=len({e['lane'] for e in spans}),
                              first_us=(begin - origin) / counter_mhz, last_us=(end - origin) / counter_mhz,
                              envelope_us=(end - begin) / counter_mhz,
                              sum_per_core_scope_us=sum(e['end'] - e['begin'] for e in spans) / counter_mhz))
    document = dict(traceEvents=trace, displayTimeUnit='ms', metadata=dict(clock=CLOCK, counter_mhz=counter_mhz))
    return document, summaries


def write_capture(directory, raw, counter_mhz, state, library, config, soc):
    events = decode(raw)
    verify_route(events, state.batch)
    document, stages = render(events, counter_mhz)
    windows = state.rotating.runtime_shape.rounds
    requests = request_counts(state.rotating.rows.cpu().numpy(), state.rotating.final_posts.cpu().numpy(),
                              windows, config.prefetch_tokens)
    total_prefetch = sum(req['prefetch_requests'] for req in requests)
    total_final_get = sum(req['final_get_requests'] for req in requests)
    phases = {event['phase'] for event in events}
    if ((total_prefetch and not {153, 154, 168}.issubset(phases))
            or (total_final_get and not {164, 165, 167}.issubset(phases))):
        raise ValueError('Observed Get counts have missing descriptor, submission or completion markers')
    fixture = Path(state.initial.source_paths[0])
    report = dict(
        correctness='PASS', execution='COMPLETED', route='Production P2 + four-AIV Terminal', soc=soc,
        aics=32, geometry=state.initial.geometry, miss_percent=config.miss_percent,
        warmup=config.warmup, traced_invocations=1, windows=windows, requested_windows=config.intermediate_fetches,
        prefetch_tokens=config.prefetch_tokens, prefetch_rank_start=config.prefetch_rank_start,
        total_prefetch=total_prefetch, total_final_get=total_final_get, requests=requests,
        fixture_sha256=fixture_hash(fixture), library_sha256=hashlib.sha256(Path(library).read_bytes()).hexdigest(),
        clock=CLOCK, counter_mhz=counter_mhz, events=len(events), stages=stages,
        raw_trace='baseline_stage_trace.bin', visual_trace='baseline_stage_trace.json',
        observer_cost='Measured source scopes include marker overhead and existing synchronization',
    )
    atomic_json(directory / 'baseline_stage_trace.json', document)
    atomic_json(directory / 'baseline_stage_report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw', type=Path, help='captured baseline_stage_trace.bin')
    parser.add_argument('--counter-mhz', type=float, required=True, help='use the measured value from the capture report')
    parser.add_argument('--output', type=Path, required=True, help='output Chrome/Perfetto trace JSON')
    args = parser.parse_args()
    document, _ = render(decode(args.raw.read_bytes()), args.counter_mhz)
    atomic_json(args.output, document)


if __name__ == '__main__':
    main()
