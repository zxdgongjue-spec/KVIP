# SPDX-License-Identifier: Apache-2.0
"""Binding and evidence for single-change P2 comparisons on A5."""

import ctypes
import hashlib

import numpy as np

SUPPORTED_BATCHES = {
    'p2': (1, 2, 4, 8, 16),
    'cold': (1, 2, 4, 8),
    'cold_p8': (1,),
    'cold_p8_bulk': (1,),
    'windows': (2, 4, 8, 16),
    'windows_early': (2,),
    'topk': (1, 8),
    'get': (1, 8),
}
MODES = ('p2', 'cold', 'windows', 'topk', 'get')
COLD_SCORE_PARTS = {'cold': 4, 'cold_p8': 8, 'cold_p8_bulk': 8}
THREE_WINDOW_MODES = ('windows', 'windows_early')
WINDOW_SCOPE_PHASES = {
    151: 'previous_wait', 243: 'window_ready', 246: 'p8_cursor', 247: 'early_header_enqueue', 250: 'bulk_cursor_mte2',
}
TRACE_ROUTES = dict(p2='p2', cold='small_cold', cold_p8='small_cold', cold_p8_bulk='small_cold',
                    windows='p2', windows_early='p2', topk='small_topk', get='p2')
KERNELS = ('v9_li_kernel', 'v9_li_kernel_1', 'v9_li_kernel_2', 'v9_parallel_terminal')
COLD_KERNELS = (KERNELS[0], 'v9_li_kernel_threshold_cold_partitioned',
                'v9_li_kernel_cold_ready_partitioned', KERNELS[3])
TOPK_KERNELS = (KERNELS[0], 'v9_li_kernel_resident_topk_high', *KERNELS[1:])
GET_BALANCE_FLAG = 2
WINDOW_PROBE_OPTION_OFFSET = 224
WINDOW_READY_LINE_BYTES = 128


def bind(state, mode):
    from terminal_handoff import dense_terminal

    forward = dense_terminal(state)
    shape = state.rotating.runtime_shape
    groups = 2 * state.batch
    baseline_windows = min(4, groups)
    if (state.batch not in SUPPORTED_BATCHES.get(mode, ()) or shape.history != 131072
            or shape.resident != 16384 or shape.indexHeads != 32 or shape.aicCount != 32
            or shape.liCores != groups or shape.rounds != (3 if mode in THREE_WINDOW_MODES else baseline_windows)
            or shape.serviceLanes != 2 * (groups // baseline_windows)
            or not state.workspace.tail_bounds.eq(0).all().item()
            or not state.workspace.inputs.query_ends.eq(state.history).all().item()):
        raise ValueError('The small-batch experiments require native dense P2 and the fixed A5 fixture')

    library = state.rotating.library
    parts = 2
    entry_mode = 'cold_p8' if mode == 'cold_p8_bulk' else mode
    if mode in COLD_SCORE_PARTS:
        choose = getattr(library, f'kvip_small_batch_{entry_mode}_parts')
        choose.argtypes, choose.restype = [ctypes.c_void_p], ctypes.c_uint32
        parts = choose(forward.shape_pointer)
        required = shape.batch * 4 * (shape.history * 2 + 64 * parts)
        if (parts != COLD_SCORE_PARTS[mode] or state.workspace.compute.nbytes < required
                or state.workspace.compute.data_ptr() % 64):
            raise ValueError(f'Cold P{COLD_SCORE_PARTS[mode]} geometry or classified-progress capacity is unsupported')
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
            if option % WINDOW_READY_LINE_BYTES:
                raise ValueError('The read-ahead flags require aligned GM lines')
        value = np.asarray([option], dtype='<u8').view('u1')
        forward.workspace[WINDOW_PROBE_OPTION_OFFSET:WINDOW_PROBE_OPTION_OFFSET + 8].copy_(state.torch.from_numpy(value))

    if mode in COLD_SCORE_PARTS or mode == 'topk':
        launch = getattr(library, f'kvip_small_batch_{entry_mode}_forward')
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
        if report.get('small_batch_mode') in ('cold_p8', 'cold_p8_bulk', *THREE_WINDOW_MODES):
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
    launches.append((COLD_KERNELS[1] if mode in COLD_SCORE_PARTS else KERNELS[1], li_groups))
    for symbol, blocks in launches:
        launch = getattr(state.rotating.library, f'aclrtlaunch_{symbol}')
        extra = [COLD_SCORE_PARTS[mode]] if symbol == COLD_KERNELS[1] else []
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
