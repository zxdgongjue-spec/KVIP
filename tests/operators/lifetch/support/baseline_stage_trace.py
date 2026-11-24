# SPDX-License-Identifier: Apache-2.0
"""Capture and decode the opt-in P2/Terminal source trace on an A5 device."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
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
CAPACITY = 512
HEADER_BYTES = LANE_BYTES = 128
EVENT = struct.Struct('<3Q8IQ')
BINDING_OFFSET = 192
# AscendCL acl/acl_prof.h. Collect task timing, msproftx stream marks,
# Runtime API calls and ACL API calls for the business replay only.
ACL_PROF_TASK_TIME = 0x0002
ACL_PROF_MSPROFTX = 0x0080
ACL_PROF_RUNTIME_API = 0x0100
ACL_PROF_ACL_API = 0x0001
MSPROF_DATA_TYPES = ACL_PROF_TASK_TIME | ACL_PROF_MSPROFTX | ACL_PROF_RUNTIME_API | ACL_PROF_ACL_API

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
PIPE_SCOPES = {
    180: ('AIC', 'WAIT', 'Wait for AIV credit'),
    181: ('AIC', 'MTE2', 'K GM to L1 with ready wait'),
    182: ('AIC', 'MTE2', 'Q GM to L1 with ready wait'),
    183: ('AIC', 'MTE1', 'L1 to L0A/L0B with ready wait'),
    184: ('AIC', 'CUBE', 'MMAD issue'),
    185: ('AIC', 'FIXP', 'Cube output issue'),
    186: ('AIC', 'WAIT', 'Wait for FIXP output bank'),
    188: ('AIV', 'MTE2', 'Query weights and scales with ready wait'),
    189: ('AIV', 'MTE2', 'K scales with ready wait'),
    190: ('AIV', 'WAIT', 'Wait for Cube result'),
    191: ('AIV', 'VF', 'Score and classify SIMD VF'),
    192: ('AIV', 'MTE3', 'Score, hit and progress writeback issue'),
    193: ('AIV', 'WAIT', 'Wait for output bank'),
    196: ('AIV', 'MTE2', 'Resident threshold input with fence'),
    197: ('AIV', 'VF', 'Resident Top2048 and victim classification'),
    198: ('AIV', 'MTE3', 'Resident threshold publication'),
    200: ('AIV', 'MTE2', 'Service window inputs'),
    201: ('AIV', 'VF', 'Service window TopK'),
    202: ('AIV', 'MTE3', 'Service window publication'),
    203: ('AIV', 'VF', 'Service candidate union issue'),
    204: ('AIV', 'VF', 'Prefetch descriptors with fence'),
    205: ('AIV', 'MTE3', 'Prefetch SQ write'),
    206: ('AIV', 'MTE2', 'Service candidates with fence'),
    210: ('AIV', 'MTE2', 'Terminal TopK inputs'),
    211: ('AIV', 'VF', 'Terminal Top2048 and selection'),
    212: ('AIV', 'MTE3', 'Terminal TopK scratch write'),
    213: ('AIV', 'VF', 'Terminal resolve with fence'),
    214: ('AIV', 'MTE3', 'Terminal output publication'),
    215: ('AIV', 'MTE3', 'Terminal WQE writes with fence'),
    216: ('AIV', 'VF', 'Index update with fence'),
    217: ('AIV', 'VF', 'Final WQE assembly with fence'),
}
PHASES.update({phase: name for phase, (_, _, name) in PIPE_SCOPES.items()})
NO_TILE, NO_QUERY, SERVICE_LANE = (1 << 20) - 1, 15, 14
PIPE_SEMANTICS = ('Instrumented source scopes bracket instruction issue and existing fences. '
                  'Lane counters count overlapping scopes per logical lane, with no PMU sampling. '
                  'Score scan scopes sample the first and last tile of each active half-window. '
                  'Core IDs are the raw device core-group '
                  'values observed alongside logical block and subblock IDs.')
# Stages that must appear on this P2/W4 route. Get submission may be
# skipped without eligible requests. Extra capacity passes are conditional.
REQUIRED = {
    20: {1, 2, 21, 22},
    21: {1, 2, 23},
    22: {1, 2, 21, 22, 25, 150, 152},
    23: {1, 2, 160, 161, 163, 166, 169},
}
REQUIRED_PIPE = {
    20: {183, 184, 191, 192},
    21: {196, 197, 198},
    22: {183, 184, 191, 192},
    23: {211, 213, 214, 216},
}
CLOCK = ('Device GetSystemCycle relative to the first recorded scope, scaled by an independent '
         'NPU Event calibration. msprof host and device task timestamps are separate.')


@contextmanager
def scoped_msprof(prof, output, device_id):
    """Collect a single replay using the existing ACL context and lease stream."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = dict(api='acl.prof', output=str(output), device_id=device_id, data_types=MSPROF_DATA_TYPES,
                  scope='one marked P2/Terminal forward through lease stream completion', status={})

    def checked(operation, *arguments):
        code = getattr(prof, operation)(*arguments)
        report['status'][operation] = code
        if code != 0:
            raise RuntimeError(f'acl.prof.{operation} failed with status {code}')

    with ExitStack() as cleanup:
        checked('init', str(output))
        cleanup.callback(checked, 'finalize')
        handle = prof.create_config([device_id], 0, 0, MSPROF_DATA_TYPES)
        if not handle:
            raise RuntimeError('acl.prof.create_config did not return a configuration handle')
        cleanup.callback(checked, 'destroy_config', handle)
        checked('start', handle)
        cleanup.callback(checked, 'stop', handle)
        yield report


class StageBuffer:
    """Attach a private trace buffer only for one already-bound native call."""

    def __init__(self, state):
        self.forward = state.forward
        library = self.forward.rotating.library
        probe = getattr(library, 'kvip_stage_trace_supported', None)
        if probe is None:
            raise RuntimeError('LIFetch library has no KVIP_STAGE_TRACE entry point')
        probe.argtypes, probe.restype = [], ctypes.c_uint32
        if probe() != VERSION:
            raise RuntimeError('LIFetch stage trace ABI version mismatch')
        self.profile_forward = library.kvip_parallel_tail_profile_forward
        self.profile_forward.argtypes = [ctypes.c_void_p] * 4
        self.profile_forward.restype = ctypes.c_uint32
        self.workspace = self.forward.workspace
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

    def submit_marked(self, stream):
        owner_stream = self.forward.rotating.owner.lease.check_stream()
        if stream.npu_stream != owner_stream.npu_stream:
            raise RuntimeError('LIFetch profiling must execute on the active URMA lease stream')
        code = self.profile_forward(self.forward.bindings.data_ptr(), self.forward.terminal.workspace.data_ptr(),
                                    stream.npu_stream, self.forward.shape_pointer)
        if code:
            raise RuntimeError(f'Marked LIFetch forward failed with status {code}')


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
            core_type = 'AIC' if lane >= 64 else 'AIV'
            if phase in PIPE_SCOPES and PIPE_SCOPES[phase][0] != core_type:
                raise ValueError(f'Unexpected core type for pipe scope {phase} on lane {lane}')
            events.append(dict(lane=lane, begin=begin, end=end, core=core, kernel=kernel, phase=phase,
                               block=block, subblock=subblock, iteration=iteration, detail=detail))
    if not events:
        raise ValueError('The device did not record any stage trace events')
    return sorted(events, key=lambda e: (e['begin'], -e['end'], e['lane']))


def verify_route(events, batch, *, require_pipe_scopes=False):
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
    if require_pipe_scopes:
        missing_pipe = {KERNELS[kernel][1]: sorted(expected - phases[kernel])
                        for kernel, expected in REQUIRED_PIPE.items() if not expected.issubset(phases[kernel])}
        if missing_pipe:
            raise ValueError(f'Missing required pipe scopes: {missing_pipe}')
    four_aiv = set(range(4 * batch))
    two_aic = set(range(64, 64 + 2 * batch))
    expected = {20: four_aiv | two_aic, 21: four_aiv, 22: four_aiv | two_aic, 23: four_aiv}
    for kernel, lanes in expected.items():
        if bodies[kernel] != lanes:
            raise ValueError(f'{KERNELS[kernel][1]} expected {len(lanes)} marked lanes, '
                             f'got {len(bodies[kernel])}')


def source_lane_counts(spans):
    """Count distinct logical lanes, even when their source scopes overlap."""
    changes = defaultdict(Counter)
    for event in spans:
        begin, end, lane = event['begin'], event['end'], event['lane']
        if begin < end:
            changes[begin][lane] += 1
            changes[end][lane] -= 1
    active = Counter()
    result = []
    last_count = 0
    for tick, change in sorted(changes.items()):
        active.update(change)
        # Counter.update retains empty keys. Count only active source lanes.
        count = sum(depth > 0 for depth in active.values())
        if count != last_count:
            result.append((tick, count))
            last_count = count
    return result


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

    for pid, name in [(0, 'Kernel envelopes | source scopes'), (1, 'AIC lanes'), (2, 'AIV lanes'),
                      (3, 'Observed source lane counts')]:
        metadata('process_name', pid, 0, name)
    kernel_bodies = defaultdict(list)
    stage_sets = defaultdict(list)
    lane_scopes = defaultdict(list)
    seen_lanes = set()
    for event in events:
        lane = event['lane']
        pid = 1 if lane >= 64 else 2
        core_type = 'AIC' if pid == 1 else 'AIV'
        if lane not in seen_lanes:
            metadata('thread_name', pid, lane, f'{"AIC" if pid == 1 else "AIV"} logical lane {event["block"]}')
            seen_lanes.add(lane)
        begin, end, phase = event['begin'], event['end'], event['phase']
        kernel = event['kernel']
        kernel_stage, entry = KERNELS[kernel]
        args = dict(kernel_entry=entry, phase_id=phase, block=event['block'], subblock=event['subblock'],
                    core_type=core_type, raw_core_id=event['core'], raw_core_id_hex=hex(event['core']),
                    iteration=event['iteration'], detail=event['detail'], begin_tick=begin, end_tick=end)
        window_epoch = None
        category = kernel_stage
        if kernel in (20, 22) and (phase in (21, 22, 25, *range(150, 156)) or phase in PIPE_SCOPES):
            window_epoch = event['iteration']
            args['window_epoch'] = window_epoch
        if phase in PIPE_SCOPES:
            _, pipe, _ = PIPE_SCOPES[phase]
            category = f'KVIP source {pipe}'
            detail = event['detail']
            query, tile = (detail >> 20) & NO_QUERY, detail & NO_TILE
            args['pipe'] = pipe
            if query == SERVICE_LANE:
                args['service_lane'] = detail >> 24
            else:
                args['request_row'] = detail >> 24
            if query not in (NO_QUERY, SERVICE_LANE):
                args['query_start' if phase < 200 else 'query'] = query
            if tile != NO_TILE:
                args['tile_index'] = tile
                args['sampled_scan_tile'] = True
            lane_scopes[(core_type, pipe)].append(event)
        elif phase in (23, 155) or kernel == 23:
            args.update(request_row=event['detail'] // 4, query=event['detail'] % 4)
        elif phase in (21, 22, 152, 153):
            args['request_row'] = event['detail']
        elif phase in (25, 150, 151):
            args['service_lane'] = event['detail']
        elif phase == 154:
            args['request_or_service_lane'] = event['detail']
        add_span(PHASES[phase], category, pid, lane, begin, end, args)
        if phase == 1:
            kernel_bodies[kernel].append(event)
            lane_scopes[(core_type, 'kernel body')].append(event)
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
    def lane_mapping(mapping):
        return [dict(block=block, subblock=subblock, raw_core_ids=sorted(raw_ids))
                for (block, subblock), raw_ids in sorted(mapping.items())]

    for (kernel, phase, window_epoch), spans in sorted(stage_sets.items()):
        begin, end = bounds(spans)
        aic, aiv = defaultdict(set), defaultdict(set)
        for span in spans:
            mapping = aic if span['lane'] >= 64 else aiv
            mapping[(span['block'], span['subblock'])].add(span['core'])
        summaries.append(dict(kernel_entry=KERNELS[kernel][1], phase=PHASES[phase], window_epoch=window_epoch,
                              pipe=PIPE_SCOPES[phase][1] if phase in PIPE_SCOPES else None,
                              events=len(spans), logical_lanes=len({e['lane'] for e in spans}),
                              logical_aic_lanes=len({block for block, _ in aic}),
                              logical_aiv_lanes=len({block for block, _ in aiv}),
                              observed_aic_lanes=lane_mapping(aic), observed_aiv_lanes=lane_mapping(aiv),
                              first_us=(begin - origin) / counter_mhz, last_us=(end - origin) / counter_mhz,
                              envelope_us=(end - begin) / counter_mhz,
                              sum_per_core_scope_us=sum(e['end'] - e['begin'] for e in spans) / counter_mhz))
    for tid, ((core_type, scope), spans) in enumerate(sorted(lane_scopes.items()), 1):
        title = f'{core_type} {scope} source lanes'
        for tick, count in source_lane_counts(spans):
            trace.append(dict(name=title, cat='KVIP source lane count', ph='C', pid=3, tid=tid,
                              ts=(tick - origin) / counter_mhz, args={'lanes': count}))
    document = dict(traceEvents=trace, displayTimeUnit='ms',
                    metadata=dict(clock=CLOCK, counter_mhz=counter_mhz, pipe_scope_semantics=PIPE_SEMANTICS))
    return document, summaries


def replay_evidence(state, library, config, soc):
    windows = state.rotating.runtime_shape.rounds
    requests = request_counts(state.rotating.rows.cpu().numpy(), state.rotating.final_posts.cpu().numpy(),
                              windows, config.prefetch_tokens)
    fixture = Path(state.initial.source_paths[0])
    with Path(library).open('rb') as binary:
        library_sha256 = hashlib.file_digest(binary, 'sha256').hexdigest()
    return dict(
        correctness='PASS', execution='COMPLETED', route='Production P2 + four-AIV Terminal', soc=soc,
        aics=32, geometry=state.initial.geometry, miss_percent=config.miss_percent,
        warmup=config.warmup, windows=windows, requested_windows=config.intermediate_fetches,
        prefetch_tokens=config.prefetch_tokens, prefetch_rank_start=config.prefetch_rank_start,
        total_prefetch=sum(req['prefetch_requests'] for req in requests),
        total_final_get=sum(req['final_get_requests'] for req in requests),
        requests=requests, fixture_sha256=fixture_hash(fixture), library_sha256=library_sha256,
    )


def write_instruction_replay(directory, state, library, config, soc):
    """Record the checked worker replay independently of msOpProf artifacts."""
    report = replay_evidence(state, library, config, soc)
    shape = state.rotating.runtime_shape
    report.update(
        invocations=1,
        kernel_launch_plan=[dict(kernel_entry=KERNELS[kernel][1], launches=1,
                                 blocks=4 * state.batch if kernel == 23 else shape.liCores)
                            for kernel in REQUIRED],
        bridge=str(Path(library).resolve()),
        source_trace_enabled=False,
        inner_profiler_enabled=False,
    )
    atomic_json(directory / 'instruction_replay_report.json', report)
    return report


def write_capture(directory, raw, counter_mhz, state, library, config, soc, msprof_capture):
    events = decode(raw)
    verify_route(events, state.batch, require_pipe_scopes=True)
    document, stages = render(events, counter_mhz)
    report = replay_evidence(state, library, config, soc)
    phases = {event['phase'] for event in events}
    if ((report['total_prefetch'] and not {153, 154, 168, 204, 205}.issubset(phases))
            or (report['total_final_get'] and not {164, 165, 167, 215, 217}.issubset(phases))):
        raise ValueError('Observed Get counts have missing descriptor, submission or completion markers')
    report.update(
        traced_invocations=1,
        clock=CLOCK, counter_mhz=counter_mhz, events=len(events), stages=stages,
        msprof_capture=msprof_capture,
        raw_trace='baseline_stage_trace.bin', visual_trace='baseline_stage_trace.json',
        observer_cost='Measured source scopes include marker overhead and existing synchronization',
        pipe_scope_semantics=PIPE_SEMANTICS,
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
