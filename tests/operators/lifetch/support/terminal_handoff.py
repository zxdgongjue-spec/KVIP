# SPDX-License-Identifier: Apache-2.0
"""Build and check evidence for the MR25 rotating-Service handoff benchmark."""

import ctypes
import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
STAGES = ('hot_us', 'threshold_us', 'cold_service_us', 'terminal_us')
COUNT_COLUMNS = ('prefetch_requests', 'final_get_requests', 'total_get_requests')


def build_library(directory, cann, headers, soc, *, local_counts, hot_partition=False, stage_trace=False):
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


def fixture_hash(path):
    files = [*(f'arg{i}.bin' for i in range(5)), 'physical_to_logical.npy', 'logical_scores.npy']
    digest = hashlib.sha256()
    for name in files:
        with (path / name).open('rb') as stream:
            digest.update(hashlib.file_digest(stream, 'sha256').digest())
    return digest.hexdigest()


def terminal_stages(measured):
    """Split MR25's fixed launch order within each checked graph invocation."""
    rows = []
    for sample in measured['samples']:
        intervals = sample['intervals_us']
        li = intervals.get('li', ())
        terminal = intervals.get('topk_resolve_update', ())
        if len(li) != 3 or len(terminal) != 1:
            raise ValueError(f'Expected Hot, threshold, Cold and Terminal: {intervals}')
        phases = [*li, *terminal]
        if (any(start >= end for start, end in phases)
                or any(left[1] > right[0] for left, right in zip(phases, phases[1:]))):
            raise ValueError(f'MR25 phases do not follow the expected order: {phases}')
        rows.append(dict(zip(STAGES, (end - start for start, end in phases))))
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
