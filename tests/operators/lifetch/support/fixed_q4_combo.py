# SPDX-License-Identifier: Apache-2.0
"""Opt-in binding for the fixed-score × last-window query experiment."""

import ctypes

import numpy as np

ARMS = {
    'p2_q4': (False, 1),
    'fixed_q4': (True, 1),
    'p2_q1': (False, 4),
    'fixed_q1': (True, 4),
}
PRESSURE_MODES = {
    'baseline': (32, False),
    'cold_p4': (16, False),
    'overlap': (32, True),
    'combined': (16, True),
}
CREDIT_OFFSET = 224
SCRATCH_OFFSET = 232
LINES = 5
LINE_BYTES = 128


def bind(state, arm, library, pressure_mode=None):
    fixed, workers = ARMS[arm]
    forward = state.forward
    shape = state.rotating.runtime_shape
    terminal = forward.terminal
    if terminal is None or terminal.fixed_score_ready is None:
        raise ValueError('Fixed-Q4 score groups are unavailable for this experimental shape')
    groups = terminal.fixed_score_ready.shape[0] // 2
    if not fixed:
        launch = state.rotating.library.kvip_parallel_tail_forward
        launch.argtypes = [ctypes.c_void_p] * 4
        launch.restype = ctypes.c_uint32
        forward.terminal_function = launch
        groups = shape.liCores
    if workers == 4:
        scratch = state.torch.zeros((shape.batch, LINES, LINE_BYTES),
                                    dtype=state.torch.uint8, device=state.device)
        address = scratch.data_ptr()
        if not address or address % LINE_BYTES:
            raise ValueError('Q1 mailbox requires aligned GM lines')
        pointer = np.asarray([address], dtype='<u8').view('u1')
        forward.workspace[SCRATCH_OFFSET:SCRATCH_OFFSET + 8].copy_(state.torch.from_numpy(pointer))
        forward.fixed_q4_window_scratch = scratch
    route = dict(arm=arm, score_groups=groups, windows=shape.rounds, service_lanes=shape.serviceLanes,
                 q_workers=workers, library=str(library))
    if pressure_mode is not None:
        if arm != 'fixed_q1' or shape.batch != 4 or shape.rounds != 4 or groups != 32:
            raise ValueError('The credit experiment requires BS4/W4 fixed Score and last-window Q1')
        cold_groups, deferred = PRESSURE_MODES[pressure_mode]
        option = np.asarray([int(deferred)], dtype='<u8').view('u1')
        forward.workspace[CREDIT_OFFSET:CREDIT_OFFSET + 8].copy_(state.torch.from_numpy(option))
        pressure_launch = state.rotating.library.kvip_fixed_q4_pressure_forward
        pressure_launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint32] * 2
        pressure_launch.restype = ctypes.c_uint32

        def launch(*args):
            return pressure_launch(*args, cold_groups, 0)

        def profile_launch(*args):
            return pressure_launch(*args, cold_groups, 1)

        forward.terminal_function = launch
        forward.pressure_profile_launcher = profile_launch
        route.update(pressure_mode=pressure_mode, cold_score_groups=cold_groups, deferred_credit=deferred)
        forward.fixed_q4_pressure = route
    return route


def validate_query_join(forward):
    scratch = getattr(forward, 'fixed_q4_window_scratch', None)
    if scratch is not None:
        flags = scratch[:, 1:, :4].contiguous().cpu().numpy().view('<u4')
        if not np.all(flags == 1):
            raise RuntimeError('Fixed-Q4 Q1 replay did not publish all four query flags')
