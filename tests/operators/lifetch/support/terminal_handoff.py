# SPDX-License-Identifier: Apache-2.0
"""Build and check evidence for the MR25 rotating-Service handoff benchmark."""

import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
STAGES = ('hot_us', 'threshold_us', 'cold_service_us', 'terminal_us')
COUNT_COLUMNS = ('prefetch_requests', 'final_get_requests', 'total_get_requests')


def build_library(directory, cann, headers, soc, *, local_counts):
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
         f'-DSOC_VERSION={soc}', f'-DKVIP_SERVICE_LOCAL_COUNTS={option}'],
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
