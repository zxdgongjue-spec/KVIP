# SPDX-License-Identifier: Apache-2.0
"""A5-only eager batch-pipeline experiment with checked transport ownership."""
from __future__ import annotations

import ctypes
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path
import struct
import subprocess
from types import SimpleNamespace

from kvip.adapters.kvaf import KvafQueueView
from kvip.operators.lifetch.terminal_queues import ENTITY_BYTES
from kvip.runtime.build import build_failure
from reporting import atomic_json, timing_summary

GROUP_SIZE = 4
GROUP_COUNT = 4
SLOTS = 3
WINDOWS = 4
QUEUE_STRIDE = 24
MODES = ('native', 'grouped_serial', 'grouped_pipeline')


def group_inputs(inputs, start, stop):
    fields = ('identities', 'logical_keys', 'logical_scales', 'query', 'weights', 'query_scale',
              'query_ends', 'tail_bounds', 'logical_address_valid', 'page_keys', 'captured_layout',
              'captured_hot_valid', 'captured_lru', 'logical_score_keys', 'logical_score_valid')
    return replace(inputs, geometry={**inputs.geometry, 'batch': stop - start},
                   **{name: getattr(inputs, name)[start:stop] for name in fields})


class GroupQueueView(KvafQueueView):
    """Select one slot's physical lanes before either Service or Terminal binds."""

    def __init__(self, lease, output, addresses, valid, *, slot):
        super().__init__(lease, output, addresses, valid)
        self.slot = slot
        self.selected_args = None

    def build_device_args(self, library):
        if self.selected_args is None:
            args = super().build_device_args(library)
            offset = self.slot * QUEUE_STRIDE
            self.selected_args = args[offset:offset + QUEUE_STRIDE]
            if self.selected_args.shape[0] != QUEUE_STRIDE:
                raise RuntimeError('Insufficient exclusive KVAF queue views for this slot')
            # The terminal binder reads owner.device_args directly.
            self.device_args = self.selected_args
            self.core_status = self.core_status[offset:offset + QUEUE_STRIDE]
        return self.selected_args

    def physical_queue_ids(self):
        """Read real lane pointers, including doorbells, before enqueueing any Gets."""
        stream = self.lease.check_stream()
        fields = list(struct.iter_unpack('<10Q8I', self.device_args.cpu().numpy().tobytes()))
        base, state = fields[0][8:10]
        if not base or any(args[8] != base + i * ENTITY_BYTES for i, args in enumerate(fields)):
            raise RuntimeError('Expected contiguous live entities for the selected queue range')
        if not state or any(args[9] != state + i * self.abi_sizes['queue'] for i, args in enumerate(fields)):
            raise RuntimeError('Expected contiguous live cursors for the selected queue range')
        entities = self.lease.torch.empty(len(fields) * ENTITY_BYTES, dtype=self.lease.torch.uint8,
                                          device=self.device)
        snapshot = self.lease.library.kvip_transport_snapshot
        snapshot.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
        snapshot.restype = ctypes.c_uint32
        if snapshot(base, entities.data_ptr(), entities.numel(), stream.npu_stream):
            raise RuntimeError('Could not snapshot physical KVAF queues for batch pipeline')
        raw = entities.cpu().numpy().tobytes()
        return tuple((args[9], entity[7], entity[8], entity[13], entity[14])
                     for args, entity in zip(fields, struct.iter_unpack('<24Q', raw), strict=True))


def check_slot_queues(slot_queues):
    """Each cursor, SQ/CQ ring and doorbell must belong to a single active slot."""
    all_slots = {}
    for slot, queues in slot_queues:
        if slot in all_slots:
            if all_slots[slot] != queues:
                raise RuntimeError(f'Slot {slot} changed its physical queues on reuse')
            continue
        all_slots[slot] = queues
    for lane_type, addresses in enumerate(zip(*(queue for queues in all_slots.values() for queue in queues))):
        if 0 in addresses or len(set(addresses)) != len(addresses):
            raise RuntimeError(f'KVAF physical queue alias across active pipeline lanes (field {lane_type})')


def enqueue_groups(compute, owners, ready, done, grouped_serial, score, service, terminal):
    """Submitting every Score to one FIFO gives MIX kernels exclusive queue order."""
    for group in range(len(done)):
        slot = group % len(owners)
        if grouped_serial and group:
            compute.wait_event(done[group - 1])
        elif group >= len(owners):
            compute.wait_event(done[group - len(owners)])
        for window, event in enumerate(ready[group]):
            score(group, window)
            event.record(compute)
        owner = owners[slot]
        for window, event in enumerate(ready[group]):
            owner.wait_event(event)
            service(group, window)
        terminal(group)
        done[group].record(owner)


class PipelineLaunch:
    def __init__(self, state):
        forward, rotating = state.forward, state.rotating
        if forward.terminal is None:
            raise RuntimeError('Batch pipeline requires the current four-AIV Terminal')
        supported = rotating.library.kvip_lifetch_batch_pipeline_supported
        supported.argtypes, supported.restype = [ctypes.c_void_p], ctypes.c_uint32
        self.shape = forward.shape_pointer
        if supported(self.shape) != 1:
            raise RuntimeError('A5 batch pipeline does not support this RuntimeShape')
        self.binding = forward.bindings.data_ptr()
        self.terminal = forward.terminal.workspace.data_ptr()
        self.score_fn = self.function(rotating.library, 'score', True)
        self.service_fn = self.function(rotating.library, 'service', True)
        self.terminal_fn = self.function(rotating.library, 'terminal', False)
        self.lease = state.lease

    @staticmethod
    def function(library, name, has_window):
        fn = getattr(library, f'kvip_lifetch_batch_pipeline_{name}')
        fn.argtypes = [ctypes.c_void_p] * 3 + ([ctypes.c_uint32] if has_window else [])
        fn.restype = ctypes.c_uint32
        return fn

    def score(self, stream, window):
        code = self.score_fn(self.binding, stream.npu_stream, self.shape, window)
        if code:
            raise RuntimeError(f'Batch pipeline Score window {window} launch failed: {code}')

    def service(self, stream, window):
        self.lease.check_stream()
        code = self.service_fn(self.binding, stream.npu_stream, self.shape, window)
        if code:
            raise RuntimeError(f'Batch pipeline Service window {window} launch failed: {code}')

    def finish(self, stream):
        self.lease.check_stream()
        code = self.terminal_fn(self.terminal, stream.npu_stream, self.shape)
        if code:
            raise RuntimeError(f'Batch pipeline Terminal launch failed: {code}')


def build_library(directory, cann, headers, soc):
    """Both controls and the pipeline load the same experimental binary."""
    directory = Path(directory)
    output = directory / 'libv9_bridge.so'
    done = directory / 'complete.json'
    if done.is_file() and output.is_file():
        return output
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / 'build.log'
    source = Path(__file__).resolve().parents[4] / 'csrc/operators/lightning_indexer_with_fetch'
    with log.open('w') as stream:
        for stage, command in (
            ('configure', ['cmake', '-S', str(source), '-B', str(directory), '-DRUN_MODE=npu',
                           f'-DASCEND_CANN_PACKAGE_PATH={cann}', f'-DKVAF_INCLUDE={headers}',
                           f'-DSOC_VERSION={soc}', '-DKVIP_BATCH_PIPELINE=ON']),
            ('compile/link', ['cmake', '--build', str(directory), '--target', 'v9_bridge', '--parallel', '2']),
        ):
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise build_failure(log, stage, result.returncode)
    if not output.is_file():
        raise RuntimeError(f'Batch pipeline did not build {output}; see {log}')
    atomic_json(done, {'soc': soc, 'batch_pipeline': True})
    return output


class Replay:
    def __init__(self, torch, device, join, compute, owners, states, mode, quarantine):
        self.torch, self.device, self.join, self.compute = torch, device, join, compute
        self.owners, self.states, self.mode, self.quarantine = owners, states, mode, quarantine
        self.stages = [] if mode == 'native' else [PipelineLaunch(state) for state in states]
        self.begin, self.end = self.event(), self.event()
        self.ready = [[self.event(timing=False) for _ in range(WINDOWS)] for _ in self.stages]
        self.done = [self.event(timing=False) for _ in self.stages]

    def event(self, timing=True):
        return self.torch.npu.Event(enable_timing=timing)

    def run(self, *, intervals=False, profiler_ranges=False):
        from contextlib import nullcontext

        stage_records = []

        def launch(stage, group, window, stream, invoke):
            entry = None
            if intervals:
                entry = (stage, group, window, self.event(), self.event())
                stage_records.append(entry)
            # Optional host ranges identify submissions; the profiler's NPU
            # kernel intervals establish physical device overlap.
            span = (self.torch.profiler.record_function(f'pipeline/{stage}/g{group}/w{window}')
                    if profiler_ranges else nullcontext())
            with self.torch.npu.stream(stream), span:
                if entry:
                    entry[3].record(stream)
                invoke()
                if entry:
                    entry[4].record(stream)

        try:
            self.begin.record(self.join)
            if self.mode == 'native':
                launch('native', 0, None, self.join, self.states[0].forward)
            else:
                self.compute.wait_event(self.begin)

                def score(group, window):
                    launch('score', group, window, self.compute,
                           partial(self.stages[group].score, self.compute, window))

                def service(group, window):
                    stream = self.owners[group % len(self.owners)]
                    launch('service', group, window, stream, partial(self.stages[group].service, stream, window))

                def terminal(group):
                    stream = self.owners[group % len(self.owners)]
                    launch('terminal', group, None, stream, partial(self.stages[group].finish, stream))

                enqueue_groups(self.compute, self.owners, self.ready, self.done,
                               self.mode == 'grouped_serial', score, service, terminal)
                for event in self.done:
                    self.join.wait_event(event)
            self.end.record(self.join)
            self.end.synchronize()
        except BaseException as error:
            self.quarantine('batch_pipeline_completion_uncertain', error)
            raise
        elapsed = self.begin.elapsed_time(self.end) * 1000
        return elapsed, [
            dict(stage=stage, group=group, window=window, start_us=self.begin.elapsed_time(start) * 1000,
                 end_us=self.begin.elapsed_time(end) * 1000, elapsed_us=start.elapsed_time(end) * 1000)
            for stage, group, window, start, end in stage_records
        ]


def triple_overlap(intervals):
    """Intersection of device Event envelopes, including any covered dispatch gaps."""
    edges = []
    for interval in intervals:
        if interval['start_us'] < interval['end_us']:
            edges.extend(((interval['start_us'], 1, interval['stage']),
                          (interval['end_us'], -1, interval['stage'])))
    active = {}
    previous = total = 0.0
    for timestamp, delta, stage in sorted(edges):
        if all(active.get(name, 0) for name in ('score', 'service', 'terminal')):
            total += timestamp - previous
        active[stage] = active.get(stage, 0) + delta
        previous = timestamp
    return total


def reset_states(torch, states, owners, quarantine):
    try:
        for group, state in enumerate(states):
            with torch.npu.stream(owners[group % len(owners)]):
                state.reset()
        torch.npu.synchronize()
    except BaseException as error:
        quarantine('batch_pipeline_reset_completion_uncertain', error)
        raise


def validate_round(torch, states, owners, quarantine):
    requests = []
    for group, state in enumerate(states):
        with torch.npu.stream(owners[group % len(owners)]):
            try:
                rows = state.rotating.rows.cpu().numpy()
                pending = rows[:, [11, 211, 248]]
                if pending.any():
                    raise RuntimeError(f'Group {group} has prefetch error or pending Gets: {pending.tolist()}')
                state.check_completion()
            except BaseException as error:
                quarantine('batch_pipeline_get_completion_uncertain', error)
                raise
            state.validate(state.initial)
            for row in rows:
                requests.append(dict(prefetch=int(row[9]), final_get=int(row[247]),
                                     prefetch_per_window=[int(count) for count in row[16:16 + WINDOWS]]))
    return requests


def fixture_digest(directory):
    digest = hashlib.sha256()
    names = ('fixture.json', *(f'arg{index}.bin' for index in range(5)),
             'physical_to_logical.npy', 'logical_scores.npy')
    for name in names:
        with (Path(directory) / name).open('rb') as source:
            digest.update(hashlib.file_digest(source, 'sha256').digest())
    return digest.hexdigest()


def run_batch_pipeline(config, directory, torch, torch_npu, runtime, table, inputs, pages, addresses,
                       *, device, aics, cann, headers, soc):
    from execution import FusedReplayState, NativeFetchSession, progress

    mode = config.batch_pipeline_mode
    quarantine = NativeFetchSession.quarantine
    directory = Path(directory)
    if (mode not in MODES or config.transport != 'urma' or inputs.geometry['batch'] != GROUP_COUNT * GROUP_SIZE
            or aics != 32 or 'Ascend950' not in soc or config.intermediate_fetches != WINDOWS
            or config.prefetch_tokens <= 0):
        raise ValueError(f'Unsupported batch pipeline fixture or hardware: mode={mode}, aics={aics}, soc={soc}')
    progress(directory.parent, 'BATCH PIPELINE compiling or loading experiment library')
    library = build_library(config.batch_pipeline_build, cann, headers, soc)
    join = torch.npu.current_stream(device)
    grouped = mode != 'native'
    # Both grouped controls use the same streams, queue slots and device
    # kernel sequence. They differ only by the dependency at the group boundary.
    owners = [torch.npu.Stream(device=device) for _ in range(SLOTS)] if grouped else [join]
    compute = torch.npu.Stream(device=device) if grouped else join
    states = []
    slot_leases = {}
    try:
        slots = []
        for group in range(GROUP_COUNT if grouped else 1):
            start, stop = (group * GROUP_SIZE, (group + 1) * GROUP_SIZE) if grouped else (0, 16)
            slot = group % len(owners)
            fixture = group_inputs(inputs, start, stop)
            payload = SimpleNamespace(token_payload=lambda row, ids, start=start: pages.token_payload(start + row, ids))
            try:
                with torch.npu.stream(owners[slot]):
                    state = FusedReplayState(
                        torch, runtime, table, library, fixture, payload, addresses[start:stop],
                        transport='urma', device=device, aics=aics, windows=WINDOWS,
                        prefetch_tokens=config.prefetch_tokens, prefetch_rank_start=config.prefetch_rank_start,
                        soc=soc, lease=slot_leases.get(slot),
                        queue_view_factory=partial(GroupQueueView, slot=slot) if grouped else None,
                    )
                    states.append(state)
                    slot_leases.setdefault(slot, state.lease)
                    if grouped:
                        slots.append((slot, state.rotating.owner.physical_queue_ids()))
            except BaseException as error:
                quarantine('batch_pipeline_setup_completion_uncertain', error)
                raise
            state.failure_directory = directory / f'group{group}'
            state.failure_directory.mkdir()
        if not grouped and states[0].forward.terminal is None:
            raise RuntimeError('Native baseline did not select the current four-AIV Terminal')
        if grouped:
            check_slot_queues(slots)
        replay = Replay(torch, device, join, compute, owners, states, mode, quarantine)
        progress(directory.parent, f'BATCH PIPELINE {mode} checking eager completion, KV and Index')
        for _ in range(config.warmup):
            reset_states(torch, states, owners, quarantine)
            replay.run()
            validate_round(torch, states, owners, quarantine)
        samples, counts = [], []
        for _ in range(config.iterations):
            reset_states(torch, states, owners, quarantine)
            elapsed, _ = replay.run()
            samples.append(elapsed)
            counts.append(validate_round(torch, states, owners, quarantine))
        reset_states(torch, states, owners, quarantine)
        _, intervals = replay.run(intervals=True)
        validate_round(torch, states, owners, quarantine)
        report = dict(
            mode=mode, correctness='PASS', execution='COMPLETED', geometry=inputs.geometry,
            eager_kv_ready=timing_summary(samples), timing_scope='A5 eager device events with host enqueue gaps',
            stage_intervals_us=intervals, event_envelope_triple_overlap_us=triple_overlap(intervals),
            get_requests=counts, trace=None,
            slot_queue_offsets=[slot * QUEUE_STRIDE for slot in range(len(owners))] if grouped else [0],
            library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
            fixture_sha256=fixture_digest(inputs.source_paths[0]),
            prefetch_tokens=config.prefetch_tokens, prefetch_rank_start=config.prefetch_rank_start,
            soc=soc, aics=aics, groups=len(states), slots=len(owners),
        )
        atomic_json(directory / 'pipeline.json', report)
        if config.batch_pipeline_trace:
            reset_states(torch, states, owners, quarantine)
            activities = [torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]
            profiler = torch_npu.profiler.profile(activities=activities)
            with profiler:
                replay.run(profiler_ranges=True)
            validate_round(torch, states, owners, quarantine)
            path = directory / 'npu_trace.json'
            profiler.export_chrome_trace(str(path))
            report['trace'] = str(path)
            atomic_json(directory / 'pipeline.json', report)
        return report
    finally:
        try:
            for group, state in reversed(list(enumerate(states))):
                with torch.npu.stream(owners[group % len(owners)]):
                    state.close()
        except BaseException as error:
            quarantine('batch_pipeline_lease_release_uncertain', error)
            raise
