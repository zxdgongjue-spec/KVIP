# SPDX-License-Identifier: Apache-2.0
"""Read the fixed-Q4 device recorder using the base staged workspace ABI."""

from collections import Counter, defaultdict
import math
import struct

MAGIC = 0x4B56495054524331
VERSION = 1
LANES = 96
CAPACITY = 1024
HEADER_BYTES = LANE_BYTES = 128
WORKSPACE_OFFSET = 192
EVENT = struct.Struct('<3Q8IQ')
KERNELS = {
    20: ('Hot Score', 'v9_li_fixed_hot'),
    21: ('Resident Threshold', 'v9_li_kernel_1'),
    22: ('Cold Score + Service', 'v9_li_fixed_cold'),
    23: ('Parallel Terminal', 'v9_parallel_terminal'),
}
PHASES = {1: 'Kernel body',
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
 224: 'Dedicated driver wait',
 225: 'Dedicated ready cursor snapshot',
 226: 'Initialize pipeline events and credits',
 227: 'Drain pipeline events and output banks',
 228: 'Destroy LI pipeline',
 229: 'Clear Cold progress before Resident Top2048',
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
 248: 'Dynamically assigned Score fragment',
 249: 'Dynamically assigned Service window',
 250: 'P8 bulk cursor GM to UB',
 251: 'Snapshot W1 TopK ready bounds',
 252: 'Join W1 TopK workers',
 253: 'W1 TopK worker query slice'}
PIPE_SCOPES = {180: ('AIC', 'WAIT', 'Wait for AIV credit'),
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
 218: ('AIV', 'WAIT', 'WaitPartitionedHalf'),
 219: ('AIV', 'WAIT', 'Cold T2 output bank and MTE2 handoff'),
 220: ('AIV', 'MTE2', 'Cold T2 Score GM to UB with fence'),
 221: ('AIV', 'VF', 'Cold T2 resident-threshold predicate'),
 222: ('AIV', 'MTE3', 'Cold T2 predicate and progress publication'),
 223: ('AIV', 'STORE', 'Prefetch doorbell'),
 238: ('AIV', 'VF', 'Reduce Resident Top2048 to cutoff and fence'),
 239: ('AIV', 'MTE2', 'Load Resident victim IDs'),
 240: ('AIV', 'VF', 'Gather and classify Resident victims'),
 241: ('AIV', 'MTE3', 'Publish Resident hit bitmap')}
PHASES.update({phase: name for phase, (_, _, name) in PIPE_SCOPES.items()})
PHASES.update({251: 'Last-window TopK snapshot', 252: 'Join last-window TopK workers',
               253: 'Last-window TopK worker'})
CLOCK = 'Device GetSystemCycle scaled by NPU Event calibration'
SCOPE_SEMANTICS = ('Sampled source issue and synchronization scopes. Actual durations may overlap. '
                   'CANN device task timestamps use a separate clock domain.')


class StageBuffer:
    """Attach a private ring only for the synchronized marked replay."""

    def __init__(self, state):
        self.workspace = state.forward.workspace
        if self.workspace.numel() != 256:
            raise ValueError('The staged trace requires the 256-byte LIFetch workspace')
        torch = state.torch
        self.buffer = torch.zeros(HEADER_BYTES + LANES * (LANE_BYTES + CAPACITY * EVENT.size),
                                  dtype=torch.uint8, device=state.device)
        header = struct.pack('<Q4I', MAGIC, VERSION, LANES, CAPACITY, EVENT.size)
        self.buffer[:len(header)].copy_(torch.tensor(list(header), dtype=torch.uint8))
        address = self.buffer.data_ptr()
        if not address or address % HEADER_BYTES:
            raise ValueError('The staged device ring must be 128-byte aligned')
        self.pointer = torch.tensor(list(struct.pack('<Q', address)), dtype=torch.uint8)
        self.binding = self.workspace[WORKSPACE_OFFSET:WORKSPACE_OFFSET + 8]

    def attach(self):
        self.binding.copy_(self.pointer)

    def detach(self):
        self.binding.zero_()

    def read(self):
        return self.buffer.cpu().numpy().tobytes()


def decode(raw):
    if len(raw) < HEADER_BYTES:
        raise ValueError('Missing LIFetch source trace header')
    magic, version, lanes, capacity, stride = struct.unpack_from('<Q4I', raw)
    if (magic, version, lanes, stride) != (MAGIC, VERSION, LANES, EVENT.size) or not 0 < capacity <= 65536:
        raise ValueError('Invalid LIFetch source trace header')
    lane_stride = LANE_BYTES + capacity * EVENT.size
    if len(raw) != HEADER_BYTES + lanes * lane_stride:
        raise ValueError('Source trace buffer size differs from header')
    events = []
    for lane in range(lanes):
        base = HEADER_BYTES + lane * lane_stride
        count, overflow = struct.unpack_from('<2I', raw, base)
        if overflow or count > capacity:
            raise ValueError(f'Source trace overflow at lane {lane}, capacity {capacity}')
        core_type = 'AIV' if lane < 64 else 'AIC'
        for slot in range(count):
            fields = EVENT.unpack_from(raw, base + LANE_BYTES + slot * EVENT.size)
            begin, end, core, kernel, phase, block, subblock, epoch, detail, complete, _, _ = fields
            if complete != 1 or begin <= 0 or end < begin or (phase == 1 and begin == end):
                raise ValueError(f'Incomplete source trace event at lane {lane}, slot {slot}')
            if kernel not in KERNELS or phase not in PHASES or block != lane % 64:
                raise ValueError(f'Unexpected source trace event at lane {lane}, kernel {kernel}, phase {phase}')
            if phase in PIPE_SCOPES and PIPE_SCOPES[phase][0] != core_type:
                raise ValueError(f'Unexpected core type for source pipe scope {phase} at lane {lane}')
            events.append(dict(lane=lane, begin=begin, end=end, core=core, kernel=kernel, phase=phase,
                               block=block, subblock=subblock, iteration=epoch, detail=detail, core_type=core_type))
    return sorted(events, key=lambda event: (event['begin'], -event['end'], event['lane']))


def verify(events, *, batch, windows, service_lanes, hot_groups, cold_groups, prefetch, final_get):
    # Required scopes cover business phases independently of optional branches.
    required = {
        20: {1, 2, 21, 22, 183, 184, 185, 191, 192},
        21: {1, 2, 23, 196, 197, 198},
        22: {1, 2, 21, 22, 25, 150, 151, 152, 155, 180, 183, 184, 185, 191, 192, 218, 251, 252, 253},
        23: {1, 2, 160, 161, 163, 166, 169, 211, 213, 214, 216},
    }
    if prefetch:
        required[22].update({153, 154, 204, 205, 223})
        required[23].add(168)
    if final_get:
        required[23].update({164, 165, 167, 215, 217})
    phases = defaultdict(set)
    for event in events:
        phases[event['kernel']].add(event['phase'])
    for kernel, want in required.items():
        if missing := sorted(want - phases[kernel]):
            raise ValueError(f'{KERNELS[kernel][0]} missing staged scopes {missing}')

    def bodies(groups, aiv_only):
        result = set(range(groups * 2)) if not aiv_only else set(range(groups))
        if not aiv_only:
            result.update(range(64, 64 + groups))
        return Counter(result)

    active = {20: bodies(hot_groups, False), 21: bodies(batch * 4, True),
              22: bodies(cold_groups, False), 23: bodies(batch * 4, True)}
    for kernel, want in active.items():
        found = Counter(event['lane'] for event in events if event['phase'] == 1 and event['kernel'] == kernel)
        if found != want:
            raise ValueError(f'{KERNELS[kernel][0]} kernel bodies do not match the AIC/AIV launch geometry')
    cold = [event for event in events if event['kernel'] == 22]
    service = Counter((event['iteration'], event['detail']) for event in cold if event['phase'] == 150)
    if service != Counter((epoch, lane) for epoch in range(windows) for lane in range(service_lanes)):
        raise ValueError('Cold Service must cover each window and lane exactly once')
    for phase in (21, 22, 180, 192, 218):
        epochs = {event['iteration'] for event in cold if event['phase'] == phase}
        if epochs != set(range(1, windows)):
            raise ValueError(f'{PHASES[phase]} must cover every Cold window')
    for phase, size in ((251, batch), (252, batch), (253, batch * 4)):
        observed = Counter((event['iteration'], event['detail']) for event in cold if event['phase'] == phase)
        if observed != Counter((windows - 1, unit) for unit in range(size)):
            raise ValueError(f'{PHASES[phase]} has missing or duplicate last-window work')


def render(events, counter_mhz):
    if not math.isfinite(counter_mhz) or counter_mhz <= 0:
        raise ValueError('Invalid measured device counter frequency')
    origin = min(event['begin'] for event in events)
    groups = defaultdict(list)
    trace = []

    def metadata(name, pid, tid, title):
        trace.append(dict(name=name, ph='M', pid=pid, tid=tid, args={'name': title}))

    def span(name, cat, pid, tid, begin, end, details):
        trace.append(dict(name=name, cat=cat, ph='X', pid=pid, tid=tid, ts=(begin - origin) / counter_mhz,
                          dur=(end - begin) / counter_mhz, args=details))

    metadata('process_name', 0, 0, 'LIFetch business phases')
    metadata('process_name', 1, 0, 'AIC logical lanes')
    metadata('process_name', 2, 0, 'AIV logical lanes')
    metadata('process_name', 3, 0, 'LIFetch business handoffs')
    for lane in sorted({event['lane'] for event in events}):
        metadata('thread_name', 1 if lane >= 64 else 2, lane,
                 f'{"AIC" if lane >= 64 else "AIV"} logical lane {lane % 64}')

    for event in events:
        lane, kernel, phase = (event[key] for key in ('lane', 'kernel', 'phase'))
        phase_name, kernel_entry = PHASES[phase], KERNELS[kernel][1]
        core_type = event['core_type']
        window_epoch = event['iteration'] if kernel == 22 and phase not in (1, 2) else None
        pipe = PIPE_SCOPES[phase][1] if phase in PIPE_SCOPES else None
        details = dict(kernel_id=kernel, kernel_entry=kernel_entry, phase_id=phase, block=event['block'],
                       subblock=event['subblock'], raw_core_id=event['core'], core_type=core_type,
                       begin_tick=event['begin'], end_tick=event['end'], iteration=event['iteration'],
                       detail=event['detail'])
        if window_epoch is not None:
            details['window_epoch'] = window_epoch
        if pipe is not None:
            details['pipe'] = pipe
        if phase in (25, 150, 151, 243, 244):
            details['service_lane'] = event['detail']
        if phase in (23, 155, 253):
            details.update(request_row=event['detail'] // 4, query=event['detail'] % 4)
        if phase in (251, 252):
            details['request_row'] = event['detail']
        if phase in range(180, 194) or phase == 218:
            packed = event['detail']
            details.update(request_row=packed >> 24, query_start=(packed >> 20) & 15)
            if phase == 218:
                details['cold_part'] = packed & ((1 << 20) - 1)
            elif (packed & ((1 << 20) - 1)) != (1 << 20) - 1:
                details['tile_index'] = packed & ((1 << 20) - 1)
        span(phase_name, f'KVIP source {pipe}' if pipe else KERNELS[kernel][0],
             1 if core_type == 'AIC' else 2, lane, event['begin'], event['end'], details)
        groups[(kernel, phase, window_epoch)].append(event)

    def bounds(items):
        return min(event['begin'] for event in items), max(event['end'] for event in items)

    stages, envelopes, handoffs = [], {}, []
    for (kernel, phase, epoch), items in sorted(groups.items()):
        begin, end = bounds(items)
        stages.append(dict(kernel_id=kernel, kernel_entry=KERNELS[kernel][1], phase_id=phase,
                           phase=PHASES[phase], window_epoch=epoch,
                           logical_aic_lanes=len({e['lane'] for e in items if e['lane'] >= 64}),
                           logical_aiv_lanes=len({e['lane'] for e in items if e['lane'] < 64}),
                           events=len(items), first_us=(begin - origin) / counter_mhz,
                           last_us=(end - origin) / counter_mhz, envelope_us=(end - begin) / counter_mhz,
                           sum_per_core_scope_us=sum(e['end'] - e['begin'] for e in items) / counter_mhz))
        if phase == 1:
            envelopes[kernel] = (begin, end)
            metadata('thread_name', 0, kernel, KERNELS[kernel][1])
            span(KERNELS[kernel][0], 'LIFetch source envelope', 0, kernel, begin, end,
                 dict(kernel_entry=KERNELS[kernel][1], begin_tick=begin, end_tick=end))

    dependencies = ('Score and hit publication to resident TopK',
                    'Resident threshold and seed to Cold',
                    'Cold windows and prefetch handoff to Terminal')
    for kernel, title in enumerate(dependencies, 20):
        before, after = envelopes[kernel][1], envelopes[kernel + 1][0]
        handoff = dict(name=title, before=KERNELS[kernel][1], after=KERNELS[kernel + 1][1],
                       begin_tick=before, end_tick=after, duration_us=(after - before) / counter_mhz)
        handoffs.append(handoff)
        if after > before:
            metadata('thread_name', 3, kernel, title)
            span(title, 'LIFetch source handoff', 3, kernel, before, after, handoff)
    return dict(traceEvents=trace, displayTimeUnit='us',
                metadata=dict(clock=CLOCK, counter_mhz=counter_mhz, pipe_scope_semantics=SCOPE_SEMANTICS,
                              source_handoffs=handoffs)), stages


def write_trace(directory, tracker, counter_mhz, state, algorithm, capture):
    from reporting import atomic_json

    raw = tracker.read()
    (directory / 'baseline_stage_trace.bin').write_bytes(raw)
    events = decode(raw)
    shape = state.rotating.runtime_shape
    combo = state.forward.fixed_q4_pressure
    totals = algorithm['totals']
    verify(events, batch=shape.batch, windows=shape.rounds, service_lanes=shape.serviceLanes,
           hot_groups=combo['score_groups'], cold_groups=combo['cold_score_groups'],
           prefetch=totals['prefetch'], final_get=totals['final'])
    trace, stages = render(events, counter_mhz)
    report = dict(correctness='PASS', execution='COMPLETED', traced_invocations=1,
                  batch=shape.batch, windows=shape.rounds, service_lanes=shape.serviceLanes,
                  hot_score_groups=combo['score_groups'], cold_score_groups=combo['cold_score_groups'],
                  total_prefetch=totals['prefetch'], total_final_get=totals['final'],
                  msprof_capture=capture, events=len(events), stages=stages, **trace['metadata'])
    atomic_json(directory / 'baseline_stage_trace.json', trace)
    atomic_json(directory / 'baseline_stage_report.json', report)
