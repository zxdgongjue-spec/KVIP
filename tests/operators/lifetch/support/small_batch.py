# SPDX-License-Identifier: Apache-2.0
"""Binding and evidence for single-change P2 comparisons on A5."""

import ctypes
import hashlib

import numpy as np

ADAPTIVE_ROUTES = {1: 'cold_p8', 2: 'cold_w2', 8: 'cold_full_w2'}
P4_WINDOW_Q_WORKERS = {'cold_w2': 1, 'cold_w2_q2': 2, 'cold_w2_q1': 4}
P2_WINDOW_Q_WORKERS = {'p2_w2': 1, 'p2_w2_q2': 2, 'p2_w2_q1': 4}
P2_W4_WINDOW_Q_WORKERS = {'p2_w4_q1': 4}
WINDOW_Q_WORKERS = P4_WINDOW_Q_WORKERS | P2_WINDOW_Q_WORKERS | P2_W4_WINDOW_Q_WORKERS
WINDOW_Q_BATCHES = (1, 2, 4, 8)
SUPPORTED_BATCHES = {
    'p2': (1, 2, 4, 8, 16),
    'cold': (1, 2, 4, 8),
    **dict.fromkeys(P4_WINDOW_Q_WORKERS, WINDOW_Q_BATCHES),
    **dict.fromkeys(P2_WINDOW_Q_WORKERS | P2_W4_WINDOW_Q_WORKERS, (16,)),
    'cold_p8': (1,),
    'cold_p8_bulk': (1,),
    'cold_full': (1, 2, 8),
    'cold_full_w3': (2, 8),
    'cold_full_w2': (2, 8),
    'windows': (2, 4, 8, 16),
    'windows_early': (2,),
    'windows_w2': (2, 8),
    'topk': (1, 8),
    'get': (1, 8),
    'adaptive': tuple(ADAPTIVE_ROUTES),
}
MODES = ('p2', 'cold', 'windows', 'topk', 'get')
FULL32_MODES = {'cold_full', 'cold_full_w3', 'cold_full_w2'}
PARTITIONED_MODES = FULL32_MODES | P4_WINDOW_Q_WORKERS.keys() | {'cold', 'cold_p8', 'cold_p8_bulk', 'adaptive'}
WINDOW_OVERRIDES = {'windows': 3, 'windows_early': 3, 'windows_w2': 2,
                    **dict.fromkeys(P2_WINDOW_Q_WORKERS | P4_WINDOW_Q_WORKERS, 2),
                    **dict.fromkeys(P2_W4_WINDOW_Q_WORKERS, 4), 'cold_full_w3': 3, 'cold_full_w2': 2}
TRACE_ROUTES = dict(p2='p2', **dict.fromkeys(PARTITIONED_MODES, 'small_cold'),
                    **dict.fromkeys(P2_WINDOW_Q_WORKERS | P2_W4_WINDOW_Q_WORKERS, 'p2'),
                    windows='p2', windows_w2='p2', windows_early='p2', topk='small_topk', get='p2')
TRACE_ROUTES['adaptive'] = 'small_retired'
KERNELS = ('v9_li_kernel', 'v9_li_kernel_1', 'v9_li_kernel_2', 'v9_parallel_terminal')
COLD_KERNELS = (KERNELS[0], 'v9_li_kernel_threshold_cold_partitioned',
                'v9_li_kernel_cold_ready_partitioned', KERNELS[3])
RETIRED_KERNELS = (KERNELS[0], 'v9_li_kernel_threshold_retired_service',
                   'v9_li_kernel_cold_retired_service', KERNELS[3])
TOPK_KERNELS = (KERNELS[0], 'v9_li_kernel_resident_topk_high', *KERNELS[1:])
GET_BALANCE_FLAG = 2
WINDOW_PROBE_OPTION_OFFSET = 224
WINDOW_READY_LINE_BYTES = 128
WINDOW_Q_OPTION_OFFSET = 232
WINDOW_Q_BYTES_PER_REQUEST = 5 * WINDOW_READY_LINE_BYTES


def selected_mode(mode, batch):
    return ADAPTIVE_ROUTES[batch] if mode == 'adaptive' else mode


def selected_windows(mode, batch, default=4):
    mode = selected_mode(mode, batch)
    return 2 if batch == 1 and mode in ('cold_p8', 'cold_p8_bulk') else WINDOW_OVERRIDES.get(mode, default)


def cold_score_parts(mode, batch):
    mode = selected_mode(mode, batch)
    if mode in FULL32_MODES:
        return 32 // batch
    if mode in P4_WINDOW_Q_WORKERS or mode == 'cold':
        return 4
    return 8 if mode in ('cold_p8', 'cold_p8_bulk') else 2


def bind(state, mode):
    from terminal_handoff import dense_terminal

    forward = dense_terminal(state)
    shape = state.rotating.runtime_shape
    groups = 2 * state.batch
    baseline_windows = min(4, groups)
    if (state.batch not in SUPPORTED_BATCHES.get(mode, ()) or shape.history != 131072
            or shape.resident != 16384 or shape.indexHeads != 32 or shape.aicCount != 32
            or shape.liCores != groups or shape.rounds != selected_windows(mode, state.batch, baseline_windows)
            or shape.serviceLanes != 2 * (groups // baseline_windows)
            or not state.workspace.tail_bounds.eq(0).all().item()
            or not state.workspace.inputs.query_ends.eq(state.history).all().item()):
        raise ValueError('The small-batch experiments require native dense P2 and the fixed A5 fixture')

    library = state.rotating.library
    retired = mode == 'adaptive'
    mode = selected_mode(mode, state.batch)
    parts = 2
    symbol_mode = 'cold_full' if mode in FULL32_MODES else mode.removesuffix('_w2').removesuffix('_bulk')
    if retired:
        symbol_mode = 'cold_retired'
    elif mode in P4_WINDOW_Q_WORKERS:
        symbol_mode = 'cold'
    if mode in PARTITIONED_MODES:
        choose = getattr(library, f'kvip_small_batch_{symbol_mode}_parts')
        choose.argtypes, choose.restype = [ctypes.c_void_p], ctypes.c_uint32
        parts = choose(forward.shape_pointer)
        expected = cold_score_parts(mode, state.batch)
        required = shape.batch * 4 * (shape.history * 2 + 64 * parts)
        alignment = 64
        if retired:
            alignment = 128
            required += (32 + shape.rounds * shape.serviceLanes) * alignment
        if (parts != expected or state.workspace.compute.nbytes < required
                or state.workspace.compute.data_ptr() % alignment):
            raise ValueError(f'Cold P{expected} geometry or classified-progress capacity is unsupported')
    elif mode == 'topk':
        forward.resident_high_table = state.torch.empty(
            (shape.batch * 4, 2, 256), dtype=state.torch.int32, device=state.device)
        address = forward.resident_high_table.data_ptr()
        pointer = state.torch.from_numpy(np.asarray([address], dtype='<u8').view('u1'))
        forward.workspace[216:224].copy_(pointer)
    elif mode == 'get':
        forward.balance_final_get = True
        forward.set_options()

    if mode in ('cold_p8_bulk', 'windows_early'):
        option = 1
        if mode == 'windows_early':
            forward.window_header_ready = state.torch.zeros(
                (shape.rounds * shape.serviceLanes, WINDOW_READY_LINE_BYTES),
                dtype=state.torch.uint8, device=state.device)
            option = forward.window_header_ready.data_ptr()
            if not option or option % WINDOW_READY_LINE_BYTES:
                raise ValueError('The read-ahead flags require aligned GM lines')
        value = np.asarray([option], dtype='<u8').view('u1')
        forward.workspace[WINDOW_PROBE_OPTION_OFFSET:WINDOW_PROBE_OPTION_OFFSET + 8].copy_(state.torch.from_numpy(value))

    q_workers = WINDOW_Q_WORKERS.get(mode, 1)
    if q_workers > 1:
        forward.window_q_scratch = state.torch.zeros(
            shape.batch * WINDOW_Q_BYTES_PER_REQUEST, dtype=state.torch.uint8, device=state.device)
        address = forward.window_q_scratch.data_ptr()
        if not address or address % WINDOW_READY_LINE_BYTES:
            raise ValueError('The split-Q window requires aligned GM state')
        value = np.asarray([address | q_workers], dtype='<u8').view('u1')
        forward.workspace[WINDOW_Q_OPTION_OFFSET:WINDOW_Q_OPTION_OFFSET + 8].copy_(state.torch.from_numpy(value))

    if mode in PARTITIONED_MODES or mode == 'topk':
        launch = getattr(library, f'kvip_small_batch_{symbol_mode}_forward')
        launch.argtypes, launch.restype = [ctypes.c_void_p] * 4, ctypes.c_uint32
        forward.terminal_function = launch
    return parts


def trace_evidence(state, report):
    forward = state.forward
    cold_history = state.history
    sources = forward.events_src.cpu().numpy()
    slots = forward.events_dst.cpu().numpy()
    times = forward.terminal.times.cpu().numpy().astype(np.uint64)
    rows = state.rotating.rows.cpu().numpy()
    for b, req in enumerate(report['requests']):
        early, final = req['prefetch_requests'], req['final_get_requests']
        total = early + final
        if int(rows[b, 8]) != total or total > sources.shape[1]:
            raise ValueError(f'Request {b}: outstanding assignment count differs from completed Gets')
        indexed = sources[b, :total]
        targets = slots[b, :total]
        if np.any(indexed < 0) or np.any(indexed >= cold_history) or np.any(targets < 0) or np.any(targets >= state.resident):
            raise ValueError(f'Request {b}: incomplete Get source or destination capture')
        logical = state.input_layout[b, indexed]
        if np.any(logical < 0) or np.any(logical >= cold_history):
            raise ValueError(f'Request {b}: invalid logical source mapping')
        offsets = np.cumsum([0, *req['prefetch_by_window']])
        if report.get('small_batch_mode') in WINDOW_Q_WORKERS.keys() | {'cold_p8', 'cold_p8_bulk', 'windows', 'windows_early'}:
            req['prefetch_tokens_in_submit_order_by_window'] = [
                logical[a:z].tolist() for a, z in zip(offsets, offsets[1:])]
        req['prefetch_tokens_by_window'] = [sorted(logical[a:z].tolist()) for a, z in zip(offsets, offsets[1:])]
        req['final_get_tokens'] = sorted(logical[early:].tolist())
        req['final_get_slots'] = [[int(token), int(slot)] for token, slot in zip(logical[early:], targets[early:])]
        req['final_original_owner_counts'] = times[b, :, 23].tolist()
        req['final_posted_counts'] = times[b, :, 20].tolist()
        req['final_posted_ordinal_start'] = times[b, :, 22].tolist()
        if (req['final_posted_counts'] != req['final_get_by_query']
                or sum(req['final_original_owner_counts']) != final):
            raise ValueError(f'Request {b}: Terminal post counters disagree')
        covered = sorted(ordinal for begin, count in zip(req['final_posted_ordinal_start'], req['final_posted_counts'])
                         for ordinal in range(begin, begin + count))
        if covered != list(range(early, total)):
            raise ValueError(f'Request {b}: Final Get queue ranges do not cover the unique requests exactly')
    return report


def capture_resident_seed(state, stream, mode):
    """Hash the resident publication before any Cold window can overwrite it."""
    forward = state.forward
    li_groups = state.rotating.runtime_shape.liCores
    launches = [(KERNELS[0], li_groups)]
    if mode == 'topk':
        launches.append((TOPK_KERNELS[1], state.batch * 8))
    threshold = (RETIRED_KERNELS[1] if mode == 'adaptive'
                 else COLD_KERNELS[1] if mode in PARTITIONED_MODES else KERNELS[1])
    launches.append((threshold, li_groups))
    for symbol, blocks in launches:
        launch = getattr(state.rotating.library, f'aclrtlaunch_{symbol}')
        extra = [cold_score_parts(mode, state.batch)] if symbol in (COLD_KERNELS[1], RETIRED_KERNELS[1]) else []
        launch.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_uint32] * len(extra)
        launch.restype = ctypes.c_uint32
        status = launch(blocks, stream.npu_stream, forward.bindings.data_ptr(), *extra)
        if status:
            raise RuntimeError(f'{symbol} resident-only replay failed with status {status}')
    stream.synchronize()
    score_id_bytes = 4 * 2048 * 6
    row_bytes = score_id_bytes + 4 * 256
    seed = state.workspace.prefix_storage[:state.batch * row_bytes].cpu().numpy().reshape(state.batch, row_bytes)
    values = seed[:, :score_id_bytes]
    thresholds = seed[:, score_id_bytes:].copy().view('<u2').reshape(state.batch, 4, 128)[..., 0]
    digest = hashlib.sha256(values.tobytes())
    digest.update(thresholds.tobytes())
    return digest.hexdigest()
