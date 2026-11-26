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
COLD_P4_KERNELS = {
    **KERNELS,
    21: (KERNELS[21][0], 'v9_li_kernel_threshold_cold_partitioned'),
    22: (KERNELS[22][0], 'v9_li_kernel_cold_partitioned'),
}
UNIFIED_ROUTES = ('full_p2', 'full_uniform')
UNIFIED_KERNELS = {
    **KERNELS,
    20: ('Full history Score', 'v9_li_kernel_score_t2'),
    22: ('Cold T2 + rotating Service', 'v9_li_kernel_cold_t2'),
}
DEDICATED_KERNELS = {
    **KERNELS,
    21: (KERNELS[21][0], 'v9_li_kernel_threshold_dedicated_service'),
    22: ('Cold Score + dedicated Service', 'v9_li_kernel_cold_dedicated_service'),
}
STATIC_READY_KERNELS = {
    **DEDICATED_KERNELS,
    20: (KERNELS[20][0], 'v9_li_kernel_hot_ready_service'),
    22: ('Cold Score + fixed Service', 'v9_li_kernel_cold_static_ready_service'),
}
DYNAMIC_KERNELS = {
    **STATIC_READY_KERNELS,
    21: (KERNELS[21][0], 'v9_li_kernel_dynamic_service'),
    22: ('Dynamic Cold Score and Service', 'v9_li_kernel_dynamic_service'),
}
SMALL_COLD_KERNELS = {**COLD_P4_KERNELS,
                      22: (KERNELS[22][0], 'v9_li_kernel_cold_ready_partitioned')}
SMALL_RETIRED_KERNELS = {
    **SMALL_COLD_KERNELS,
    21: (KERNELS[21][0], 'v9_li_kernel_threshold_retired_service'),
    22: ('Cold Score + retired AIV Service', 'v9_li_kernel_cold_retired_service'),
}
ROUTE_KERNELS = {'p2': KERNELS, 'cold_p4': COLD_P4_KERNELS, 'small_retired': SMALL_RETIRED_KERNELS,
                 'small_cold': SMALL_COLD_KERNELS, 'small_topk': KERNELS,
                 'dedicated': DEDICATED_KERNELS, 'static_ready': STATIC_READY_KERNELS,
                 'dynamic': DYNAMIC_KERNELS,
                 **{route: UNIFIED_KERNELS for route in UNIFIED_ROUTES}}
WAIT_PARTITIONED_HALF = 218
T2_PIPE_PHASES = (219, 220, 221, 222)
DEDICATED_DRIVER_WAIT = 224
DEDICATED_READY_SNAPSHOT = 225
COLD_PROGRESS_RESET = 229
DYNAMIC_SCORE_JOB = 248
DYNAMIC_SERVICE_JOB = 249
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
    DEDICATED_DRIVER_WAIT: 'Dedicated driver wait',
    DEDICATED_READY_SNAPSHOT: 'Dedicated ready cursor snapshot',
    226: 'Initialize pipeline events and credits',
    227: 'Drain pipeline events and output banks',
    228: 'Destroy LI pipeline',
    COLD_PROGRESS_RESET: 'Clear Cold progress before Resident Top2048',
    230: 'Read resident thresholds into Cold UB',
    231: 'Drain and rebind Dedicated Cold query',
    232: 'Prepare Cold progress staging in UB',
    233: 'Publish final TopK generation and prepare Resolve',
    234: 'Read LI shape and select split',
    235: 'Initialize local LI buffers',
    236: 'Bind LI prefix and progress workspace',
    237: 'Wait for Score output banks to retire',
    242: 'Publish initial window Top2048 state',
    243: 'Stage window state and ready cursors',
    244: 'Drain and publish window state',
    245: 'Prepare LI per-query run info',
    246: 'P8 cursor read and ready-prefix fold',
    247: 'Enqueue next-window Top2048 header read',
    250: 'P8 bulk cursor GM to UB',
    DYNAMIC_SCORE_JOB: 'Dynamically assigned Score fragment',
    DYNAMIC_SERVICE_JOB: 'Dynamically assigned Service window',
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
    197: ('AIV', 'VF', 'Resident Top2048 VF (legacy includes victim classification)'),
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
    WAIT_PARTITIONED_HALF: ('AIV', 'WAIT', 'WaitPartitionedHalf'),
    219: ('AIV', 'WAIT', 'Cold T2 output bank and MTE2 handoff'),
    220: ('AIV', 'MTE2', 'Cold T2 Score GM to UB with fence'),
    221: ('AIV', 'VF', 'Cold T2 resident-threshold predicate'),
    222: ('AIV', 'MTE3', 'Cold T2 predicate and progress publication'),
    223: ('AIV', 'STORE', 'Prefetch doorbell'),
    238: ('AIV', 'VF', 'Reduce Resident Top2048 to cutoff and fence'),
    239: ('AIV', 'MTE2', 'Load Resident victim IDs'),
    240: ('AIV', 'VF', 'Gather and classify Resident victims'),
    241: ('AIV', 'MTE3', 'Publish Resident hit bitmap'),
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
REQUIRED_HANDOFF = {
    20: {226, 227, 228, 234, 235, 236, 237},
    21: {226, 227, 228, 234, 235, 236, 238, 242, 245},
    22: {226, 227, 228, 230, 232, 234, 235, 236, 237, 243, 244},
    23: {233},
}
CLOCK = ('Device GetSystemCycle relative to the first recorded scope, scaled by an independent '
         'NPU Event calibration. msprof host and device task timestamps are separate.')


@contextmanager
def scoped_msprof(prof, output, device_id):
    """Collect a single replay using the existing ACL context and lease stream."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = dict(api='acl.prof', output=str(output), device_id=device_id, data_types=MSPROF_DATA_TYPES,
                  scope='one marked fused forward through lease stream completion', status={})

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

    def __init__(self, state, *, cold_route='p2', cold_parts=4):
        self.forward = state.forward
        library = self.forward.rotating.library
        probe = getattr(library, 'kvip_stage_trace_supported', None)
        if probe is None:
            raise RuntimeError('LIFetch library has no KVIP_STAGE_TRACE entry point')
        probe.argtypes, probe.restype = [], ctypes.c_uint32
        if probe() != VERSION:
            raise RuntimeError('LIFetch stage trace ABI version mismatch')
        profile_symbol = {
            'p2': 'kvip_parallel_tail_profile_forward',
            'cold_p4': 'kvip_parallel_tail_cold_partitioned_profile_forward',
            'small_cold': 'kvip_small_batch_cold_profile_forward',
            'small_retired': 'kvip_small_batch_cold_retired_profile_forward',
            'small_topk': 'kvip_small_batch_topk_profile_forward',
            'dedicated': 'kvip_parallel_tail_dedicated_service_profile_forward',
            'static_ready': 'kvip_parallel_tail_static_ready_service_profile_forward',
            'dynamic': 'kvip_parallel_tail_dynamic_service_profile_forward',
            'full_p2': 'kvip_parallel_tail_unified_p2_profile_forward',
            'full_uniform': 'kvip_parallel_tail_unified_partitioned_profile_forward',
        }[cold_route]
        if cold_route == 'small_cold' and cold_parts == 8:
            profile_symbol = 'kvip_small_batch_cold_p8_profile_forward'
        if cold_route == 'small_cold' and cold_parts in (16, 32):
            profile_symbol = 'kvip_small_batch_cold_full_profile_forward'
        self.profile_forward = getattr(library, profile_symbol)
        self.profile_forward.argtypes = [ctypes.c_void_p] * 4
        self.profile_forward.restype = ctypes.c_uint32
        self.workspace = self.forward.workspace
        if self.workspace.numel() != 256:
            raise RuntimeError('LIFetch streaming binding must be exactly 256 bytes')
        capacity = 2048 if cold_route == 'dynamic' else CAPACITY
        self.buffer = state.torch.zeros(HEADER_BYTES + LANES * (LANE_BYTES + capacity * EVENT.size),
                                        dtype=state.torch.uint8, device=state.device)
        header = struct.pack('<Q4I', MAGIC, VERSION, LANES, capacity, EVENT.size)
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


def verify_route(events, batch, *, require_pipe_scopes=False, require_handoff_scopes=False,
                 cold_route='p2', score_parts=2, cold_parts=4):
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
        raise ValueError(f'Expected Hot/Threshold/Cold/Terminal trace, got kernel IDs {sorted(phases)}')
    unified = cold_route in UNIFIED_ROUTES
    required = {**REQUIRED, 22: REQUIRED[22] - {21}} if unified else REQUIRED
    missing = {kernel: sorted(expected - phases[kernel]) for kernel, expected in required.items()
               if not expected.issubset(phases[kernel])}
    if cold_route == 'dynamic':
        missing_jobs = {DYNAMIC_SCORE_JOB, DYNAMIC_SERVICE_JOB} - phases[22]
        if missing_jobs:
            raise ValueError(f'Missing dynamic source stages: {sorted(missing_jobs)}')
    if missing:
        raise ValueError(f'Missing required source stages: {missing}')
    if require_pipe_scopes:
        required_pipe = {**REQUIRED_PIPE, 22: set(T2_PIPE_PHASES)} if unified else REQUIRED_PIPE
        missing_pipe = {ROUTE_KERNELS[cold_route][kernel][1]: sorted(expected - phases[kernel])
                        for kernel, expected in required_pipe.items() if not expected.issubset(phases[kernel])}
        if missing_pipe:
            raise ValueError(f'Missing required pipe scopes: {missing_pipe}')
    if require_handoff_scopes:
        expected = dict(REQUIRED_HANDOFF)
        if cold_route in ('cold_p4', 'small_cold', 'small_retired', 'dedicated', 'static_ready', 'dynamic'):
            expected[21] = expected[21] | {COLD_PROGRESS_RESET}
        if cold_route in ('dedicated', 'static_ready'):
            expected[22] = expected[22] | {231}
        if cold_route == 'small_retired':
            expected[22] = expected[22] | {DYNAMIC_SERVICE_JOB}
        if cold_route == 'dynamic':
            # The dynamic job scope wraps role-local initialization and drain.
            expected[22] = {228, DYNAMIC_SCORE_JOB, DYNAMIC_SERVICE_JOB}
        missing_handoff = {kernel: sorted(required - phases[kernel]) for kernel, required in expected.items()
                           if not required.issubset(phases[kernel])}
        if missing_handoff:
            raise ValueError(f'Missing required handoff scopes: {missing_handoff}')
    partition_waits = [event for event in events if event['phase'] == WAIT_PARTITIONED_HALF]
    if cold_route == 'cold_p4':
        if (not partition_waits or {event['iteration'] for event in partition_waits} != {1, 2, 3}
                or any(event['kernel'] != 22 or event['lane'] >= 64 for event in partition_waits)):
            raise ValueError('Cold P4 WaitPartitionedHalf trace must cover all three Cold windows on AIV')
    elif partition_waits:
        raise ValueError('Unexpected WaitPartitionedHalf on a two-shard Cold replay')
    if any(event['phase'] in T2_PIPE_PHASES and (not unified or event['kernel'] != 22)
           for event in events):
        raise ValueError('Unexpected Cold T2 pipe scope for this replay')
    if unified and any(event['kernel'] == 22 and (event['lane'] >= 64 or event['phase'] == 21)
                       for event in events):
        raise ValueError('Cold T2 must only trace AIV work')
    four_aiv = set(range(4 * batch))
    two_aic = set(range(64, 64 + 2 * batch))
    p4_lanes = set(range(8 * batch)) | set(range(64, 64 + 4 * batch))
    if cold_route in ('small_cold', 'small_retired') and (cold_parts not in (4, 8, 16, 32) or batch * cold_parts > 32):
        raise ValueError(f'Unsupported small-batch Cold trace part count {cold_parts}')
    small_cold_groups = batch * (cold_parts if cold_route in ('small_cold', 'small_retired') else 4)
    small_cold_lanes = set(range(2 * small_cold_groups)) | set(range(64, 64 + small_cold_groups))
    cold_lanes = {'cold_p4': p4_lanes, 'small_cold': small_cold_lanes,
                  'small_retired': small_cold_lanes, 'small_topk': four_aiv | two_aic,
                  'dedicated': four_aiv | set(range(64, 88)),
                  'static_ready': four_aiv | set(range(64, 88)),
                  'dynamic': four_aiv | two_aic,
                  'p2': four_aiv | two_aic}.get(cold_route, four_aiv)
    score_groups = batch * score_parts
    score_lanes = set(range(2 * score_groups)) | set(range(64, 64 + score_groups))
    expected = {20: score_lanes, 21: four_aiv, 22: cold_lanes, 23: four_aiv}
    kernels = ROUTE_KERNELS[cold_route]
    for kernel, lanes in expected.items():
        if bodies[kernel] != lanes:
            raise ValueError(f'{kernels[kernel][1]} expected {len(lanes)} marked lanes, '
                             f'got {len(bodies[kernel])}')
    if any(event['lane'] not in bodies[event['kernel']] for event in events):
        raise ValueError('A source scope has no corresponding kernel body on its lane')
    dedicated_scopes = {DEDICATED_DRIVER_WAIT, DEDICATED_READY_SNAPSHOT}
    dedicated_events = [event for event in events if event['phase'] in dedicated_scopes]
    if cold_route in ('dedicated', 'static_ready', 'dynamic'):
        if batch != 16:
            raise ValueError('Dedicated Service source trace requires BS16')
        driver_coverage = {(event['lane'], event['iteration']) for event in dedicated_events
                           if event['phase'] == DEDICATED_DRIVER_WAIT}
        expected_waits = {(lane, window) for lane in range(48, 64) for window in (1, 2, 3)}
        snapshot_coverage = {(event['lane'], event['iteration'], event['detail'])
                             for event in dedicated_events if event['phase'] == DEDICATED_READY_SNAPSHOT}
        workers = (verify_dynamic_jobs(events) if cold_route == 'dynamic' else
                   {(request, window): 48 + request for request in range(16) for window in range(4)})
        expected_snapshots = {(lane, window, request * 4 + q)
                              for (request, window), lane in workers.items() for q in range(4)}
        if cold_route == 'dynamic':
            expected_waits.clear()
        if (driver_coverage != expected_waits or snapshot_coverage != expected_snapshots
                or any(event['kernel'] != 22 for event in dedicated_events)):
            raise ValueError('Dedicated Service driver waits or ready snapshots have incomplete lane/window coverage')
    elif dedicated_events:
        raise ValueError('Dedicated Service scopes appeared on a different Cold route')
    if unified and require_pipe_scopes:
        sampled = {(event['iteration'], event['phase'], event['lane']) for event in events
                   if event['phase'] in T2_PIPE_PHASES}
        needed = {(window, phase, lane) for window in (1, 2, 3)
                  for phase in T2_PIPE_PHASES for lane in cold_lanes}
        if sampled != needed:
            raise ValueError(f'Cold T2 samples differ from expected window/lane coverage: '
                             f'{len(needed - sampled)} missing, {len(sampled - needed)} unexpected')



def dynamic_job(detail):
    return (detail >> 4) & 15, (detail >> 2) & 3, detail & 3


def verify_dynamic_jobs(events):
    """Verify the 64 independent Resident producers, three score peers and window handoffs."""
    resident = {}
    tickets = defaultdict(list)
    services = {}
    for event in events:
        phase, lane, detail = event['phase'], event['lane'], event['detail']
        if phase == 23 and event['kernel'] == 21:
            if detail in resident or detail != lane:
                raise ValueError('Dynamic Resident requires a unique AIV per Q')
            resident[detail] = event['end']
        elif phase == DYNAMIC_SCORE_JOB:
            if event['kernel'] != 22 or dynamic_job(detail)[1] != event['iteration']:
                raise ValueError('Score job tag does not match its window or kernel')
            tickets[detail].append(event)
        elif phase == DYNAMIC_SERVICE_JOB:
            row, window, _ = dynamic_job(detail)
            key = row, window
            if (key in services or lane >= 64 or event['kernel'] != 22
                    or window != event['iteration']):
                raise ValueError(f'Duplicate or invalid Service assignment at {key}')
            services[key] = event
    if set(resident) != set(range(64)):
        raise ValueError('Incomplete single-round Resident coverage')
    score = {}
    for ticket, markers in tickets.items():
        key = dynamic_job(ticket)
        aics = [e['lane'] for e in markers if e['lane'] >= 64]
        if key in score or len(markers) != 3 or len(aics) != 1:
            raise ValueError(f'Score job {key} did not execute once as a MIX group')
        group = aics[0] - 64
        if not group or {e['lane'] for e in markers} != {aics[0], 2 * group, 2 * group + 1}:
            raise ValueError(f'Score job {key} has mismatched FIXP peers')
        if min(e['begin'] for e in markers) < max(resident[key[0] * 4 + q] for q in range(4)):
            raise ValueError(f'Score job {key} began before its Resident inputs')
        score[key] = max(e['end'] for e in markers)
    if set(score) != {(b, w, f) for b in range(16) for w in (1, 2, 3) for f in range(3)}:
        raise ValueError('Dynamic Score did not cover every fragment once')
    if set(services) != {(b, w) for b in range(16) for w in range(4)}:
        raise ValueError('Dynamic Service did not cover every window once')
    for (row, window), event in services.items():
        ready = max(resident[row * 4 + q] for q in range(4))
        if window:
            ready = max(ready, services[row, window - 1]['end'],
                        *(score[row, window, f] for f in range(3)))
        if event['begin'] < ready:
            raise ValueError(f'Service {(row, window)} started before its data dependencies')
    return {key: event['lane'] for key, event in services.items()}

def verify_retired_jobs(events, windows, service_lanes):
    """Check unique logical-window pickup after the physical MIX tail drain."""
    tails = {event['lane']: event['end'] for event in events
             if event['kernel'] == 22 and event['phase'] == 27}
    expected = {(epoch, lane) for epoch in range(1, windows) for lane in range(service_lanes)}
    jobs = {}
    for event in events:
        if event['phase'] != DYNAMIC_SERVICE_JOB:
            continue
        logical, epoch, _ = dynamic_job(event['detail'])
        worker = event['lane']
        retired_at = tails.get(64 + worker // 2)
        key = epoch, logical
        if (key in jobs or key not in expected or event['kernel'] != 22
                or worker >= 64 or worker % 2 or epoch != event['iteration']
                or retired_at is None or event['begin'] < retired_at):
            raise ValueError(f'Invalid retired AIV Service assignment at {key} on worker {worker}')
        jobs[key] = worker
    if set(jobs) != expected:
        raise ValueError('Incomplete retired AIV Service assignments')
    return [dict(epoch=epoch, service_lane=lane, worker_aiv=worker)
            for (epoch, lane), worker in sorted(jobs.items())]


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


LI_ENTRY_SCOPES = (2, 226, 234, 235, 236)
LI_EXIT_SCOPES = (27, 227, 228, 237)
SOURCE_HANDOFFS = (
    ('Hot score to Resident Top2048', 20, (21, 22), 21, (23,),
     (*LI_EXIT_SCOPES, COLD_PROGRESS_RESET, *LI_ENTRY_SCOPES, 245),
     'BF16 score rows in GM'),
    ('Resident Top2048 to Cold or W0 Service', 21, (23,), 22, (21, 22, 150),
     (*LI_EXIT_SCOPES, *LI_ENTRY_SCOPES, 230, 231, 232),
     'Resident cutoff, hit bitmap and Top2048 window state'),
    ('Cold and Service to first Resolve', 22, (21, 22, 150), 23, (161,),
     (*LI_EXIT_SCOPES, 2, 160, 233),
     'Cold score, predicates, window state and submitted prefetch mapping'),
)


def boundary(event, edge):
    return dict(tick=event[edge], phase_id=event['phase'], lane=event['lane'],
                block=event['block'], raw_core_id=event['core'], iteration=event['iteration'])


def split_recorded_bodies(begin, end, left_end, right_begin, counter_mhz, observed):
    """Disjoint pieces on the source clock, even if recorded bodies overlap."""
    if end <= begin or left_end is None or right_begin is None:
        return []
    cuts = sorted({begin, end, *(tick for tick in (left_end, right_begin) if begin < tick < end)})
    pieces = []
    for a, b in zip(cuts, cuts[1:]):
        previous, following = a < left_end, b > right_begin
        if previous and following:
            name = 'Overlapping recorded kernel bodies'
        elif previous:
            name = 'Previous kernel tail'
        elif following:
            name = 'Following kernel entry'
        else:
            name = 'Between recorded kernel bodies'
        covered = covered_ticks([(max(a, start), min(b, stop)) for start, stop in observed
                                 if max(a, start) < min(b, stop)])
        pieces.append(dict(name=name, begin_tick=a, end_tick=b, duration_us=(b-a)/counter_mhz,
                           uncovered_source_us=(b-a-covered)/counter_mhz))
    return pieces


def source_handoffs(events, counter_mhz, *, cold_route='p2'):
    """Measure business boundaries on the source clock, with work on any lane."""
    by_phase = defaultdict(list)
    for event in events:
        by_phase[event['kernel'], event['phase']].append(event)
    result = []
    for label, left, previous, right, following, phases, data in SOURCE_HANDOFFS:
        ends = {PHASES[phase]: max(e['end'] for e in by_phase[left, phase])
                for phase in previous if by_phase[left, phase]}
        starts = {PHASES[phase]: min(e['begin'] for e in by_phase[right, phase])
                  for phase in following if by_phase[right, phase]}
        if not ends or not starts:
            continue
        begin, end = max(ends.values()), min(starts.values())
        previous_event = max((e for phase in previous for e in by_phase[left, phase]),
                             key=lambda e: e['end'])
        following_event = min((e for phase in following for e in by_phase[right, phase]),
                              key=lambda e: e['begin'])
        left_body = max(by_phase[left, 1], key=lambda e: e['end'], default=None)
        right_body = min(by_phase[right, 1], key=lambda e: e['begin'], default=None)
        left_end = left_body['end'] if left_body else None
        right_begin = right_body['begin'] if right_body else None
        coverage = []
        all_spans = []
        for phase in phases:
            spans = [(max(begin, e['begin']), min(end, e['end']))
                     for kernel in (left, right) for e in by_phase[kernel, phase]
                     if max(begin, e['begin']) < min(end, e['end'])]
            if not spans:
                continue
            all_spans.extend(spans)
            coverage.append(dict(phase=PHASES[phase],
                                 duration_on_any_lane_us=covered_ticks(spans) / counter_mhz))
        duration = max(0, end - begin)
        gate = 'Request-local readiness' if cold_route == 'dynamic' and left == 21 else 'Same-stream stage gate'
        result.append(dict(name=label, begin_tick=begin, end_tick=end,
                           dependency=dict(gate=gate, data=data, producer_stage=left, consumer_stage=right),
                           boundaries=dict(previous_business_end=boundary(previous_event, 'end'),
                                           previous_body_end=boundary(left_body, 'end') if left_body else None,
                                           following_body_begin=boundary(right_body, 'begin') if right_body else None,
                                           following_business_begin=boundary(following_event, 'begin')),
                           body_segments=split_recorded_bodies(begin, end, left_end, right_begin,
                                                               counter_mhz, all_spans),
                           last_previous_scope_end_ticks=ends, first_following_scope_begin_ticks=starts,
                           duration_us=duration / counter_mhz,
                           uncovered_source_us=(duration - covered_ticks(all_spans)) / counter_mhz,
                           overlapping_source_scopes=end < begin, observed_scopes=coverage))
    return result


def covered_ticks(spans):
    covered, last = 0, 0
    for start, end in sorted(spans):
        covered += max(0, end - max(start, last))
        last = max(last, end)
    return covered


def unrecorded_lane_spans(begin, end, sorted_spans):
    """Complement of recorded source scopes on one logical lane."""
    cursor = begin
    for start, stop in sorted_spans:
        if start >= end:
            break
        if stop <= cursor or stop <= start:
            continue
        if start > cursor:
            yield cursor, start
        cursor = max(cursor, stop)
        if cursor >= end:
            return
    if cursor < end:
        yield cursor, end


def handoff_activity(events, begin, end):
    """Partition the interval at observed source edges without double-counting lanes."""
    if end <= begin:
        return []
    changes = defaultdict(list, {begin: [], end: []})
    for index, event in enumerate(events):
        start, stop = max(begin, event['begin']), min(end, event['end'])
        if start < stop:
            changes[start].append((index, event))
            changes[stop].append((index, None))

    active = {}
    pieces = []
    cuts = sorted(changes)
    for start, stop in zip(cuts, cuts[1:]):
        for index, event in changes[start]:
            if event is None:
                active.pop(index)
            else:
                active[index] = event

        per_lane = {}
        for event in active.values():
            lane = event['lane']
            priority = (event['phase'] == 1, event['end'] - event['begin'], event['phase'])
            if lane not in per_lane or priority < per_lane[lane][0]:
                per_lane[lane] = (priority, event)
        scoped = [event for priority, event in per_lane.values() if not priority[0]]
        ordered = sorted(scoped, key=lambda e: (-e['end'], e['lane']))
        owners = [dict(lane=e['lane'], block=e['block'], raw_core_id=e['core'],
                       core_type='AIC' if e['lane'] >= 64 else 'AIV',
                       kernel_id=e['kernel'], phase_id=e['phase'], phase=PHASES[e['phase']])
                  for e in ordered[:4]]
        status = 'recorded_scope' if scoped else 'kernel_body_only' if active else 'no_recorded_scope'
        details = dict(status=status, scope_owners=owners, scoped_lane_count=len(scoped),
                       aic_blocks=sorted(lane % 64 for lane in per_lane if lane >= 64),
                       aiv_blocks=sorted(lane for lane in per_lane if lane < 64))
        if pieces and pieces[-1]['details'] == details:
            pieces[-1]['end_tick'] = stop
        else:
            pieces.append(dict(begin_tick=start, end_tick=stop, details=details))
    return pieces


def render(events, counter_mhz, *, cold_route='p2'):
    if not math.isfinite(counter_mhz) or counter_mhz <= 0:
        raise ValueError('A positive measured device counter frequency is required')
    origin = min(event['begin'] for event in events)
    trace = []
    kernels = ROUTE_KERNELS[cold_route]
    scope_labels = {(20, 22): 'AIV full-history Score scan',
                    (20, 191): 'Full-history Score reduction SIMD VF',
                    (20, 192): 'Full-history Score writeback issue',
                    (22, 22): 'AIV precomputed Cold tile scan'} if cold_route in UNIFIED_ROUTES else {}
    if cold_route in ('dedicated', 'static_ready'):
        scope_labels[22, 150] = 'Dedicated Service'
    elif cold_route == 'dynamic':
        scope_labels[22, 150] = 'Service on assigned AIV'

    def metadata(name, pid, tid, label):
        trace.append(dict(name=name, ph='M', pid=pid, tid=tid, args={'name': label}))

    def add_span(name, cat, pid, tid, begin, end, args):
        trace.append(dict(name=name, cat=cat, ph='X', pid=pid, tid=tid,
                          ts=(begin - origin) / counter_mhz, dur=(end - begin) / counter_mhz, args=args))

    for pid, name in [(0, 'Kernel envelopes | source scopes'), (1, 'AIC lanes'), (2, 'AIV lanes'),
            (3, 'Observed source lane counts'), (4, 'Business handoffs | source clock'),
            (5, 'Handoff activity and dependencies | source clock')]:
        metadata('process_name', pid, 0, name)
    kernel_bodies = defaultdict(list)
    stage_sets = defaultdict(list)
    lane_scopes = defaultdict(list)
    recorded_lane_spans = defaultdict(list)
    seen_lanes = set()
    for event in events:
        lane = event['lane']
        pid = 1 if lane >= 64 else 2
        core_type = 'AIC' if pid == 1 else 'AIV'
        if lane not in seen_lanes:
            metadata('thread_name', pid, lane, f'{"AIC" if pid == 1 else "AIV"} logical lane {event["block"]}')
            seen_lanes.add(lane)
        begin, end, phase = event['begin'], event['end'], event['phase']
        recorded_lane_spans[lane].append((begin, end))
        kernel = event['kernel']
        kernel_stage, entry = kernels[kernel]
        args = dict(kernel_id=kernel, kernel_entry=entry, phase_id=phase, block=event['block'], subblock=event['subblock'],
                    core_type=core_type, raw_core_id=event['core'], raw_core_id_hex=hex(event['core']),
                    iteration=event['iteration'], detail=event['detail'], begin_tick=begin, end_tick=end)
        window_epoch = None
        category = kernel_stage
        if kernel in (20, 22) and (phase in (21, 22, 25, *range(150, 156), 243, 244, 246, 247, 250,
                                             DEDICATED_DRIVER_WAIT, DEDICATED_READY_SNAPSHOT, 230, 231,
                                             DYNAMIC_SCORE_JOB, DYNAMIC_SERVICE_JOB)
                                   or phase in PIPE_SCOPES):
            window_epoch = event['iteration']
            args['window_epoch'] = window_epoch
        if phase in PIPE_SCOPES:
            _, pipe, _ = PIPE_SCOPES[phase]
            category = f'KVIP source {pipe}'
            detail = event['detail']
            query, slot = (detail >> 20) & NO_QUERY, detail & NO_TILE
            args['pipe'] = pipe
            if query == SERVICE_LANE:
                args['service_lane'] = detail >> 24
            else:
                args['request_row'] = detail >> 24
            if phase == WAIT_PARTITIONED_HALF:
                args.update(query_start=query, cold_part=slot)
            elif query not in (NO_QUERY, SERVICE_LANE):
                args['query_start' if phase < 200 or phase in T2_PIPE_PHASES else 'query'] = query
            if phase != WAIT_PARTITIONED_HALF and slot != NO_TILE:
                args['tile_index'] = slot
                args['sampled_scan_tile'] = True
            lane_scopes[(core_type, pipe)].append(event)
        elif phase in (DYNAMIC_SCORE_JOB, DYNAMIC_SERVICE_JOB):
            row, window, fragment = dynamic_job(event['detail'])
            args.update(request_row=row, window_epoch=window, job_ticket=event['detail'] >> 8)
            if phase == DYNAMIC_SCORE_JOB:
                args['fragment'] = fragment
        elif phase in (23, 155, 245, DEDICATED_READY_SNAPSHOT, COLD_PROGRESS_RESET) or kernel == 23:
            args.update(request_row=event['detail'] // 4, query=event['detail'] % 4)
        elif phase == 242:
            args.update(request_row=event['detail'] >> 24, query=(event['detail'] >> 20) & NO_QUERY)
        elif phase in (21, 22, 152, 153, 230, 231):
            args['request_row'] = event['detail']
        elif phase in (25, 150, 151, 243, 244, 246, 247, 250, DEDICATED_DRIVER_WAIT):
            args['service_lane'] = event['detail']
        elif phase == 154:
            args['request_or_service_lane'] = event['detail']
        add_span(scope_labels.get((kernel, phase), PHASES[phase]), category, pid, lane, begin, end, args)
        if phase == 1:
            kernel_bodies[kernel].append(event)
            lane_scopes[(core_type, 'kernel body')].append(event)
        stage_sets[(kernel, phase, window_epoch)].append(event)

    def bounds(spans):
        return min(e['begin'] for e in spans), max(e['end'] for e in spans)

    for kernel, spans in sorted(kernel_bodies.items()):
        name, entry = kernels[kernel]
        metadata('thread_name', 0, kernel, entry)
        # The partitioned routes reset the GM Cold cursors before RunLi.
        # The reset belongs to the same Threshold dispatch.
        prologue = stage_sets.get((kernel, COLD_PROGRESS_RESET, None), [])
        begin, end = bounds([*spans, *prologue])
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
        summaries.append(dict(kernel_entry=kernels[kernel][1],
                              phase=scope_labels.get((kernel, phase), PHASES[phase]), window_epoch=window_epoch,
                              kernel_id=kernel, phase_id=phase,
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
    handoffs = source_handoffs(events, counter_mhz, cold_route=cold_route)
    for spans in recorded_lane_spans.values():
        spans.sort()
    for tid, handoff in enumerate(handoffs, 1):
        metadata('thread_name', 4, tid, handoff['name'])
        activity = handoff_activity(events, handoff['begin_tick'], handoff['end_tick'])
        totals = dict.fromkeys(('recorded_scope', 'kernel_body_only', 'no_recorded_scope'), 0)
        for piece in activity:
            totals[piece['details']['status']] += piece['end_tick'] - piece['begin_tick']
        longest = sorted((piece for piece in activity if piece['details']['scope_owners']),
                         key=lambda x: x['begin_tick'] - x['end_tick'])[:5]
        handoff['activity'] = dict(
            **{status + '_us': ticks / counter_mhz for status, ticks in totals.items()},
            longest_observed_spans=[
                dict(**piece, duration_us=(piece['end_tick'] - piece['begin_tick']) / counter_mhz)
                for piece in longest
            ])
        if handoff['duration_us']:
            metadata('thread_name', 5, tid, handoff['name'] + ' · activity')
            dependency_tid = tid + len(handoffs)
            metadata('thread_name', 5, dependency_tid, handoff['name'] + ' · dependency')
            dependency = handoff['dependency']
            add_span(dependency['gate'] + ' · ' + dependency['data'], 'KVIP handoff dependency',
                     5, dependency_tid, handoff['begin_tick'], handoff['end_tick'],
                     dict(**dependency, boundaries=handoff['boundaries']))
            for piece in activity:
                details = piece['details']
                if details['scope_owners']:
                    owner = details['scope_owners'][0]
                    text = f"{owner['core_type']}{owner['block']} · {owner['phase']}"
                else:
                    text = ('Kernel body only' if details['status'] == 'kernel_body_only'
                            else 'No source scope recorded')
                add_span(text, 'KVIP handoff observed activity', 5, tid,
                         piece['begin_tick'], piece['end_tick'], details)
            add_span(handoff['name'], 'KVIP business handoff', 4, tid, handoff['begin_tick'],
                     handoff['end_tick'], handoff)
        if handoff['body_segments']:
            detail_tid = tid + len(handoffs)
            metadata('thread_name', 4, detail_tid, handoff['name'] + ' breakdown')
            for segment in handoff['body_segments']:
                add_span(segment['name'], 'KVIP source handoff breakdown', 4, detail_tid,
                         segment['begin_tick'], segment['end_tick'], segment)
                for lane, spans in recorded_lane_spans.items():
                    gaps = unrecorded_lane_spans(segment['begin_tick'], segment['end_tick'], spans)
                    for begin, end in gaps:
                        core_type = 'AIC' if lane >= 64 else 'AIV'
                        add_span('Handoff gap · ' + segment['name'], 'KVIP source lane handoff',
                                 1 if lane >= 64 else 2, lane, begin, end,
                                 dict(handoff=handoff['name'], body_segment=segment['name'],
                                      core_type=core_type, block=lane % 64, derived=True,
                                      begin_tick=begin, end_tick=end,
                                      coverage='No LIFetch source scope recorded on this logical lane'))
    document = dict(traceEvents=trace, displayTimeUnit='ms',
                    metadata=dict(clock=CLOCK, counter_mhz=counter_mhz, cold_route=cold_route,
                                  pipe_scope_semantics=PIPE_SEMANTICS, source_handoffs=handoffs,
                                  handoff_semantics='Body segments partition the source handoff at recorded '
                                  'KernelBody boundaries. Between bodies may still contain source work. '
                                  'Observed scopes can overlap and their durations are not additive.',
                                  lane_handoff_semantics='Derived handoff gaps on observed logical lanes '
                                  'exclude all recorded source scopes. Hardware idle time requires a '
                                  'device-side measurement.',
                                  handoff_activity_semantics='One shortest recorded scope per logical lane. '
                                  'The label shows the sampled scope with the latest end in that slice. '
                                  'Stage gates describe code dependencies, not a point-to-point transfer.'))
    return document, summaries


def replay_evidence(state, library, config, soc):
    from kvip.operators.lifetch.geometry import prefetch_budget

    windows = state.rotating.runtime_shape.rounds
    _, window_limit = prefetch_budget(config.prefetch_tokens, getattr(config, 'prefetch_window_limit', None),
                                      config.intermediate_fetches)
    requests = request_counts(state.rotating.rows.cpu().numpy(), state.rotating.final_posts.cpu().numpy(),
                              windows, config.prefetch_tokens)
    if getattr(config, 'prefetch_window_limit', None) is not None:
        if any(count > window_limit for row in requests for count in row['prefetch_by_window']):
            raise ValueError('Observed prefetch exceeds the configured window cap')
    fixture = Path(state.initial.source_paths[0])
    with Path(library).open('rb') as binary:
        library_sha256 = hashlib.file_digest(binary, 'sha256').hexdigest()
    return dict(
        correctness='PASS', execution='COMPLETED', route='Production P2 + four-AIV Terminal', soc=soc,
        aics=32, geometry=state.initial.geometry, miss_percent=config.miss_percent,
        warmup=config.warmup, windows=windows, requested_windows=config.intermediate_fetches,
        prefetch_tokens=config.prefetch_tokens, prefetch_rank_start=config.prefetch_rank_start,
        prefetch_window_limit=window_limit,
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


def write_capture(directory, raw, counter_mhz, state, library, config, soc, msprof_capture, *,
                  cold_parts=2, score_parts=2, trace_route=None, resident_seed_sha256=None):
    comparison_route = getattr(config, 'msprof_cold_route', None)
    cold_route = trace_route or comparison_route or 'p2'
    events = decode(raw)
    verify_route(events, state.batch, require_pipe_scopes=True, require_handoff_scopes=True,
                 cold_route=cold_route, score_parts=score_parts, cold_parts=cold_parts)
    if comparison_route is None and getattr(config, 'prefetch_window_limit', None) is not None:
        shape = state.rotating.runtime_shape
        expected_service = {(epoch, lane) for epoch in range(shape.rounds) for lane in range(shape.serviceLanes)}
        observed = [(event['iteration'], event['detail']) for event in events
                    if event['kernel'] == 22 and event['phase'] == 150]
        if set(observed) != expected_service or len(observed) != len(expected_service):
            raise ValueError('Service trace does not cover each effective window and lane exactly once')
    if comparison_route:
        cold_phases = (22, 151) if cold_route in UNIFIED_ROUTES else (21, 22, 151, 192)
        for phase in cold_phases:
            windows = {event['iteration'] for event in events if event['kernel'] == 22 and event['phase'] == phase}
            if windows != {1, 2, 3}:
                raise ValueError(f'Cold comparison phase {PHASES[phase]} must cover all three Cold windows')
    document, stages = render(events, counter_mhz, cold_route=cold_route)
    report = replay_evidence(state, library, config, soc)
    if cold_route == 'small_retired':
        shape = state.rotating.runtime_shape
        report['retired_service_jobs'] = verify_retired_jobs(events, shape.rounds, shape.serviceLanes)
    if cold_route == 'cold_p4':
        report['route'] = 'Hot P2 + Cold P4 + four-AIV Terminal'
    elif cold_route in ('dedicated', 'static_ready'):
        report.update(route='Hot P2 + Cold Score with Dedicated Service + four-AIV Terminal',
                      cold_score_aic=24, cold_score_aiv=48, cold_service_aiv=16)
    elif cold_route == 'dynamic':
        report.update(route='Hot P2 + Resident and dynamically assigned Cold/Service + four-AIV Terminal',
                      cold_score_worker_pool=31, service_worker_pool=64)
    elif cold_route in UNIFIED_ROUTES:
        report['route'] = f'Full Score P{score_parts} + Cold T2 + four-AIV Terminal'
    small_mode = getattr(config, 'small_batch_mode', None)
    if small_mode is not None:
        from small_batch import trace_evidence

        report['small_batch_mode'] = small_mode
        report['resident_seed_sha256'] = resident_seed_sha256
        trace_evidence(state, report)
    phases = {event['phase'] for event in events}
    if ((report['total_prefetch'] and not {153, 154, 168, 204, 205}.issubset(phases))
            or (report['total_final_get'] and not {164, 165, 167, 215, 217}.issubset(phases))):
        raise ValueError('Observed Get counts have missing descriptor, submission or completion markers')
    report.update(
        traced_invocations=1,
        cold_route=cold_route, hot_parts=score_parts, cold_parts=None if cold_route in ('dedicated', 'static_ready', 'dynamic') else cold_parts,
        service_lanes=int(state.rotating.runtime_shape.serviceLanes),
        local_counts=getattr(config, 'msprof_local_counts', False),
        clock=CLOCK, counter_mhz=counter_mhz, events=len(events), stages=stages,
        msprof_capture=msprof_capture,
        raw_trace='baseline_stage_trace.bin', visual_trace='baseline_stage_trace.json',
        observer_cost='Measured source scopes include marker overhead and existing synchronization',
        pipe_scope_semantics=PIPE_SEMANTICS,
        source_handoffs=document['metadata']['source_handoffs'],
    )
    atomic_json(directory / 'baseline_stage_trace.json', document)
    atomic_json(directory / 'baseline_stage_report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw', type=Path, help='captured baseline_stage_trace.bin')
    parser.add_argument('--counter-mhz', type=float, required=True, help='use the measured value from the capture report')
    parser.add_argument('--output', type=Path, required=True, help='output Chrome/Perfetto trace JSON')
    parser.add_argument('--handoff-report', type=Path, help='optional compact handoff report on the source clock')
    parser.add_argument('--cold-route', choices=ROUTE_KERNELS, default='p2',
                        help='kernel aliases used by the recorded replay')
    args = parser.parse_args()
    if args.handoff_report and args.handoff_report.resolve() == args.output.resolve():
        parser.error('Use different paths for --output and --handoff-report')
    document, _ = render(decode(args.raw.read_bytes()), args.counter_mhz, cold_route=args.cold_route)
    atomic_json(args.output, document)
    if args.handoff_report:
        atomic_json(args.handoff_report, dict(clock=CLOCK, counter_mhz=args.counter_mhz,
                                             cold_route=args.cold_route,
                                             source_business_handoffs=document['metadata']['source_handoffs']))


if __name__ == '__main__':
    main()
