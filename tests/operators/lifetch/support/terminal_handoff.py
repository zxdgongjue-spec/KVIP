# SPDX-License-Identifier: Apache-2.0
"""Build and check evidence for the MR25 rotating-Service handoff benchmark."""

import ctypes
import hashlib
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
STAGES = ('hot_us', 'threshold_us', 'cold_service_us', 'terminal_us')
DYNAMIC_STAGES = ('hot_us', 'resident_cold_service_us', 'terminal_us')
READY_SERVICE_MODES = ('static_ready', 'dynamic')
P2_W1_KERNELS = {
    'p2': ('v9_li_kernel', 'v9_li_kernel_1', 'v9_li_kernel_2', 'v9_parallel_terminal'),
    'p2_w1': ('v9_li_kernel', 'v9_li_kernel_threshold_p2_w1',
              'v9_li_kernel_cold_p2_w1', 'v9_parallel_terminal'),
}
COLD_GM_MODES = ('cold_gm_fenced', 'cold_gm_pipeline')
COLD_GM_STAGE_FIELDS = {
    'cold_gm_fenced': ('hot_us', 'cold_raw_threshold_us', 'cold_service_us', 'terminal_us'),
    'cold_gm_pipeline': ('hot_us', 'cold_gm_pipeline_us', 'terminal_us'),
}
COLD_GM_KERNELS = {
    'p2': P2_W1_KERNELS['p2'],
    'cold_gm_fenced': ('v9_li_kernel_hot_cold_gm', 'v9_li_kernel_cold_gm_batch',
                       'v9_li_kernel_cold_t2', 'v9_parallel_terminal'),
    'cold_gm_pipeline': ('v9_li_kernel_hot_cold_gm', 'v9_li_kernel_cold_gm_batch',
                         'v9_parallel_terminal'),
}
from small_batch import (
    COLD_KERNELS as SMALL_COLD_KERNELS, COLD_SEPARATE_KERNELS, PARTITIONED_MODES, TOPK_KERNELS as SMALL_TOPK_KERNELS,
)

CHECKED_KERNELS = {**P2_W1_KERNELS, **COLD_GM_KERNELS,
                   **{f'small_{mode}': P2_W1_KERNELS['p2'] for mode in ('p2', 'windows', 'windows_w2', 'get')},
                   **{f'small_{mode}': SMALL_COLD_KERNELS for mode in PARTITIONED_MODES},
                   'small_cold_separate': COLD_SEPARATE_KERNELS,
                   'small_topk': SMALL_TOPK_KERNELS}
COLD_GM_STAGE_FIELDS['small_topk'] = (STAGES[0], 'resident_histogram_us', *STAGES[1:])
DYNAMIC_MAILBOX_BYTES = 192 * 128
COUNT_COLUMNS = ('prefetch_requests', 'final_get_requests', 'total_get_requests')


def build_library(directory, cann, headers, soc, *, local_counts, hot_partition=False, stage_trace=False,
                  ub_candidates=False, post_chunk=0, dedicated_drain_aiv1=True):
    from kvip.runtime.build import build_failure

    directory = Path(directory)
    library = directory / 'libv9_bridge.so'
    if library.is_file():
        return library
    directory.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'csrc/operators/lightning_indexer_with_fetch'
    option = 'ON' if local_counts else 'OFF'
    log = directory / 'build.log'
    commands = [
        ['cmake', '-S', str(source), '-B', str(directory), '-DRUN_MODE=npu',
         f'-DASCEND_CANN_PACKAGE_PATH={cann}', f'-DKVAF_INCLUDE={headers}',
         f'-DSOC_VERSION={soc}', f'-DKVIP_SERVICE_LOCAL_COUNTS={option}',
         f'-DKVIP_HOT_SCORE_PARTITION={"ON" if hot_partition else "OFF"}',
         f'-DKVIP_SERVICE_UB_CANDIDATES={"ON" if ub_candidates else "OFF"}',
         f'-DKVIP_SERVICE_POST_CHUNK={post_chunk}',
         f'-DKVIP_DEDICATED_DRAIN_AIV1={"ON" if dedicated_drain_aiv1 else "OFF"}',
         f'-DKVIP_STAGE_TRACE={"ON" if stage_trace else "OFF"}'],
        ['cmake', '--build', str(directory), '--target', 'v9_bridge', '--parallel', '2'],
    ]
    with log.open('w') as output:
        for stage, command in zip(('configure', 'compile/link'), commands):
            print(command, file=output, flush=True)
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
            if result.returncode:
                raise build_failure(log, stage, result.returncode)
    if not library.is_file():
        raise RuntimeError(f'Missing paired-bench library; see {log}')
    return library


def bind_hot_score(state, *, partitioned):
    return bind_score_partition(state, hot=partitioned)[0]


def dense_terminal(state):
    workspace = state.workspace
    if not (workspace.query_lengths.eq(4).all().item()
            and workspace.key_lengths.eq(state.history).all().item()):
        raise ValueError('Score partition comparison requires dense Q4 and full key lengths')
    forward = state.forward
    if forward.terminal is None:
        raise ValueError('Score partition comparison requires the MR25 Terminal route')
    return forward


def bind_score_partition(state, *, hot, cold=None):
    """Select the experimental Score launches for a dense Terminal pipeline."""
    workspace = state.workspace
    forward = dense_terminal(state)
    library = forward.rotating.library

    def parts(name, enabled):
        select = getattr(library, f'kvip_partitioned_{name}_score_parts')
        select.argtypes, select.restype = [ctypes.c_void_p], ctypes.c_uint32
        planned = select(forward.shape_pointer)
        if planned < 2:
            raise ValueError(f'{name.title()} partition is unsupported for this Terminal shape')
        return planned if enabled else 2

    hot_parts = parts('hot', hot)
    cold_parts = parts('cold', cold) if cold is not None else 2
    if cold_parts > 2:
        queries = workspace.query_lengths.numel() * 4
        required_bytes = queries * (2 * state.history + 64 * cold_parts)
        if workspace.compute.nbytes < required_bytes or workspace.compute.data_ptr() % 64:
            raise ValueError('Cold partition requires aligned progress space after the Score rows')
    entry = {
        (True, False): 'kvip_parallel_tail_hot_partitioned_forward',
        (False, True): 'kvip_parallel_tail_cold_partitioned_forward',
        (True, True): 'kvip_parallel_tail_history_partitioned_forward',
    }.get((hot_parts > 2, cold_parts > 2))
    if entry:
        function = getattr(library, entry)
        function.argtypes = [ctypes.c_void_p] * 4
        function.restype = ctypes.c_uint32
        forward.terminal_function = function
    return hot_parts, cold_parts


def bind_unified_score(state, mode):
    """Keep native P2 or bind a full-history Score before the same Terminal."""
    forward = dense_terminal(state)
    if mode == 'p2':
        return 2
    if mode not in ('full_p2', 'full_uniform'):
        raise ValueError(f'Unknown unified Score mode {mode!r}')
    workspace = state.workspace
    if not (workspace.tail_bounds.eq(0).all().item()
            and workspace.inputs.query_ends.eq(state.history).all().item()):
        raise ValueError('Unified Score requires full query bounds and no recent tail')
    library = forward.rotating.library
    select = library.kvip_unified_score_parts
    select.argtypes, select.restype = [ctypes.c_void_p], ctypes.c_uint32
    planned = select(forward.shape_pointer)
    if planned < 2:
        raise ValueError('Unified Score is unsupported for this Terminal shape')
    parts = planned if mode == 'full_uniform' else 2
    name = 'kvip_parallel_tail_unified_partitioned_forward' if parts > 2 else 'kvip_parallel_tail_unified_p2_forward'
    function = getattr(library, name)
    function.argtypes, function.restype = [ctypes.c_void_p] * 4, ctypes.c_uint32
    forward.terminal_function = function
    return parts


def bind_dedicated_service(state, mode):
    """Bind an opt-in P2 or Cold scheduler route to the existing fused Terminal."""
    forward = dense_terminal(state)
    if mode == 'p2':
        return
    if mode in COLD_GM_MODES:
        workspace = state.workspace
        if not (workspace.tail_bounds.eq(0).all().item()
                and workspace.inputs.query_ends.eq(state.history).all().item()):
            raise ValueError('Cold GM requires full query bounds and no recent tail')
        queries = state.batch * 4
        if (workspace.compute.nbytes < queries * (state.history * 2 + 128)
                or workspace.compute.data_ptr() % 128):
            raise ValueError('Cold GM requires aligned per-query readiness lines after the Score rows')
        library = forward.rotating.library
        supported = library.kvip_cold_gm_supported
        supported.argtypes, supported.restype = [ctypes.c_void_p], ctypes.c_uint32
        if supported(forward.shape_pointer) != 1:
            raise ValueError('Cold GM is unsupported for this Terminal shape')
        function = getattr(library, f'kvip_parallel_tail_{mode}_forward')
        function.argtypes, function.restype = [ctypes.c_void_p] * 4, ctypes.c_uint32
        forward.terminal_function = function
        return
    if mode == 'p2_w1':
        workspace = state.workspace
        if not (workspace.tail_bounds.eq(0).all().item()
                and workspace.inputs.query_ends.eq(state.history).all().item()):
            raise ValueError('P2 W1 requires full query bounds and no recent tail')
        library = forward.rotating.library
        supported = library.kvip_p2_w1_supported
        supported.argtypes, supported.restype = [ctypes.c_void_p], ctypes.c_uint32
        if supported(forward.shape_pointer) != 1:
            raise ValueError('P2 W1 is unsupported for this Terminal shape')
        function = library.kvip_parallel_tail_p2_w1_forward
        function.argtypes, function.restype = [ctypes.c_void_p] * 4, ctypes.c_uint32
        forward.terminal_function = function
        return
    if mode not in ('dedicated', *READY_SERVICE_MODES):
        raise ValueError(f'Unknown dedicated Service mode {mode!r}')
    workspace = state.workspace
    if not (workspace.tail_bounds.eq(0).all().item()
            and workspace.inputs.query_ends.eq(state.history).all().item()):
        raise ValueError('Dedicated Service requires full query bounds and no recent tail')
    queries = state.batch * 4
    cold_windows = state.rotating.runtime_shape.rounds - 1
    ledger_bytes = queries * cold_windows * 3 * 64
    if (workspace.compute.nbytes < queries * state.history * 2 + ledger_bytes
            or workspace.compute.data_ptr() % 64):
        raise ValueError('Dedicated Service requires aligned progress space after the Score rows')
    library = forward.rotating.library
    if mode in READY_SERVICE_MODES:
        required = queries * state.history * 2 + ledger_bytes + DYNAMIC_MAILBOX_BYTES
        if workspace.compute.nbytes < required or workspace.compute.data_ptr() % 128:
            raise ValueError('Ready Service requires an aligned mailbox after the Cold progress ledger')
        supported = library.kvip_dynamic_service_supported
    else:
        supported = library.kvip_dedicated_cold_supported
    supported.argtypes, supported.restype = [ctypes.c_void_p], ctypes.c_uint32
    if supported(forward.shape_pointer) != 1:
        raise ValueError('Dedicated Service is unsupported for this Terminal shape')
    function = getattr(library, f'kvip_parallel_tail_{mode}_service_forward')
    function.argtypes, function.restype = [ctypes.c_void_p] * 4, ctypes.c_uint32
    forward.terminal_function = function


def run_staged_eager(state, mode, progress):
    """Synchronize the selected launches to localize an eager failure."""
    forward = state.forward
    rotating = forward.rotating
    stream = rotating.owner.lease.check_stream()
    shape = rotating.runtime_shape
    bindings = forward.bindings.data_ptr()
    terminal = forward.terminal.workspace.data_ptr()
    hot = 'v9_li_kernel'
    if mode in READY_SERVICE_MODES:
        hot = 'v9_li_kernel_hot_ready_service'
    elif mode in COLD_GM_MODES:
        hot = 'v9_li_kernel_hot_cold_gm'
    stages = [('Hot', hot, shape.liCores, bindings)]
    if mode in COLD_GM_MODES:
        stages.append(('Cold GM batch', 'v9_li_kernel_cold_gm_batch', shape.liCores, bindings,
                       int(mode == 'cold_gm_pipeline')))
        if mode == 'cold_gm_fenced':
            stages.append(('Cold GM consume', 'v9_li_kernel_cold_t2', shape.liCores, bindings))
    elif mode == 'dynamic':
        stages.append(('Resident+Cold+Service', 'v9_li_kernel_dynamic_service', shape.liCores, bindings))
    else:
        threshold, cold = {
            'p2': ('_1', '_2'),
            'p2_w1': ('_threshold_p2_w1', '_cold_p2_w1'),
            'dedicated': ('_threshold_dedicated_service', '_cold_dedicated_service'),
            'static_ready': ('_threshold_dedicated_service', '_cold_static_ready_service'),
        }[mode]
        stages.extend((
            ('Threshold', 'v9_li_kernel' + threshold, shape.liCores, bindings),
            ('Cold+Service', 'v9_li_kernel' + cold, shape.liCores, bindings),
        ))
    stages.append(('Terminal', 'v9_parallel_terminal', shape.batch * 4, terminal))
    progress('staged eager binding synchronization')
    stream.synchronize()
    progress('staged eager binding synchronized')
    for label, symbol, blocks, workspace, *parameters in stages:
        launch = getattr(rotating.library, f'aclrtlaunch_{symbol}')
        launch.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_uint32] * len(parameters)
        launch.restype = ctypes.c_uint32
        progress(f'staged eager {label} launch ({symbol})')
        result = launch(blocks, stream.npu_stream, workspace, *parameters)
        if result:
            raise RuntimeError(f'{symbol} launch failed: {result}')
        progress(f'staged eager {label} synchronization')
        stream.synchronize()
        progress(f'staged eager {label} completed')


def fixture_hash(path):
    files = [*(f'arg{i}.bin' for i in range(5)), 'physical_to_logical.npy', 'logical_scores.npy']
    digest = hashlib.sha256()
    for name in files:
        with (path / name).open('rb') as stream:
            digest.update(hashlib.file_digest(stream, 'sha256').digest())
    return digest.hexdigest()


def terminal_stages(measured, *, dynamic=False, kernel_route=None):
    """Measure the actual launch boundaries within each checked graph invocation."""
    names = COLD_GM_STAGE_FIELDS.get(kernel_route, DYNAMIC_STAGES if dynamic else STAGES)
    rows = []
    if kernel_route is not None:
        intervals = measured['accounting']['kernel_intervals']
        seen = sorted((*intervals['li'], *intervals['topk_resolve_update']))
        expected = CHECKED_KERNELS[kernel_route] * len(measured['samples'])
        actual = tuple(name.lower() for _, _, name in seen)
        if len(actual) != len(expected) or any(
                not re.fullmatch(symbol + r'(?:_\d+)*', name)
                for name, symbol in zip(actual, expected)):
            raise ValueError(f'Profiler did not execute the {kernel_route} kernels: {actual}')
    for sample in measured['samples']:
        intervals = sample['intervals_us']
        li = intervals.get('li', ())
        terminal = intervals.get('topk_resolve_update', ())
        if len(li) != len(names) - 1 or len(terminal) != 1:
            raise ValueError(f'Unexpected launch count for {names}: {intervals}')
        phases = [*li, *terminal]
        if (any(start >= end for start, end in phases)
                or any(left[1] > right[0] for left, right in zip(phases, phases[1:]))):
            raise ValueError(f'MR25 phases do not follow the expected order: {phases}')
        rows.append(dict(zip(names, (end - start for start, end in phases))))
    return rows


def request_counts(rows, posts, windows, budget):
    """Decode the completed Row and FinalPost snapshots after CQ join."""
    if not 1 <= windows <= 32 or rows.ndim != 2 or rows.shape[1:] != (256,) or posts.shape != (len(rows), 4, 32):
        raise ValueError('Unexpected MR25 counter ABI or window count')
    result = []
    for request, (row, final_posts) in enumerate(zip(rows, posts)):
        prefetch = int(row[9])
        by_window = row[16:16 + windows].tolist()
        by_query = final_posts[:, 0].tolist()
        final = int(row[247])
        if (any(row[index] for index in (11, 211, 248)) or final_posts[:, 1:3].any()
                or min(by_window + by_query) < 0 or not 0 <= prefetch <= budget
                or sum(by_window) != prefetch or sum(by_query) != final):
            raise ValueError(f'MR25 request {request} has invalid completed Get counters')
        result.append(dict(request=request, prefetch_by_window=by_window, final_get_by_query=by_query,
                           **dict(zip(COUNT_COLUMNS, (prefetch, final, prefetch + final)))))
    return result


def cold_gm_timeline(state, mode, clock_mhz):
    """Read the AIV handoff stamps from a separate diagnostic replay."""
    from diagnostics import union

    queries = state.batch * 4
    offset = queries * state.history * 2
    lanes = state.workspace.compute[offset:offset + queries * 128].cpu().numpy().view('<u8').reshape(queries, 16)
    records = []
    for aiv, line in enumerate(lanes):
        if int(line[0]) != 0x100000001:
            raise ValueError(f'Cold GM AIV {aiv} did not publish both readiness flags')
        raw_begin, raw_end, threshold_begin, threshold_end = map(int, line[1:5])
        if not (0 < raw_begin < raw_end and 0 < threshold_begin < threshold_end):
            raise ValueError(f'Cold GM AIV {aiv} has incomplete producer timestamps')
        record = dict(aiv=aiv, request=aiv // 4, raw_score=[raw_begin, raw_end],
                      threshold=[threshold_begin, threshold_end])
        if mode == 'cold_gm_pipeline':
            begin, end = map(int, line[5:7])
            if not (max(raw_end, threshold_end) <= begin < end):
                raise ValueError(f'Cold GM AIV {aiv} has invalid consumption timestamps')
            record['classify_and_service'] = [begin, end]
        records.append(record)

    def overlap(left, right):
        return sum(max(0, min(last_a, last_b) - max(first_a, first_b))
                   for first_a, last_a in left for first_b, last_b in right) / clock_mhz

    score = union(record['raw_score'] for record in records)
    threshold = union(record['threshold'] for record in records)
    result = dict(clock_mhz=clock_mhz, lanes=records,
                  score_threshold_overlap_us=overlap(score, threshold))
    if mode == 'cold_gm_pipeline':
        service = union(record['classify_and_service'] for record in records)
        result['score_consumer_overlap_us'] = overlap(score, service)
    return result
