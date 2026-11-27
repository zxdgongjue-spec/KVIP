# SPDX-License-Identifier: Apache-2.0
"""Bind the existing four-window LI graph to the experimental BS32 terminal."""
import ctypes
from functools import lru_cache, partial
import importlib.util
from pathlib import Path

from four_rank_tail_image import tail_text

KERNEL = "v9_test_two_row_terminal"


def install_profiler(kernel=KERNEL):
    """Keep the original graph timer and require the candidate's real hardware task."""
    import fixed_q4_packed_timing as timing

    # The native checkout supplies all normal replay modules. Load just the
    # extended stage parser from this pinned measurement controller.
    spec = importlib.util.spec_from_file_location("b24_two_row_stage_parser", Path(__file__).with_name("large_batch_pair.py"))
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    timing.fixed_stage_records = partial(parser.fixed_stage_records, expected_tail=kernel)


def install(large_two_cold, repo, sdk_build, soc):
    from two_row_tail import prepare_tail

    original_attach = large_two_cold.attach
    counter_mhz = None
    workspaces = set()

    def attach(state, launch):
        nonlocal counter_mhz
        if state.batch != 32 or launch.window_policy != "four-ranks" or launch.trace_only:
            raise ValueError("Two-row measurement requires the fixed four-window BS32 path")
        original_attach(state, launch)
        regular = state.rotating.library
        hot = ctypes.cast(regular.aclrtlaunch_v9_li_large_hot, ctypes.c_void_p)
        normal = Path(regular._name).resolve()
        if counter_mhz is None:
            counter_mhz = state.forward.calibrate_counter_mhz()
        bound = prepare_tail(repo, normal, sdk_build, soc, state.forward, counter_mhz=counter_mhz)
        pointer = bound.workspace.data_ptr()
        if pointer in workspaces:
            raise ValueError("Two captured calls share a mutable tail mailbox")
        workspaces.add(pointer)
        state.large_two_row_terminal = bound

        def forward(bindings, terminal, stream, shape):
            return bound.forward(launch, hot, bindings, terminal, stream, shape)

        state.forward.terminal_function = forward

    large_two_cold.attach = attach


@lru_cache(maxsize=None)
def _images(normal, candidate):
    return tail_text(normal), tail_text(candidate, kernel="two_row_kernel")


def identity(state):
    bound = state.large_two_row_terminal
    candidate = Path(bound.library_path).resolve()
    original, measured = _images(Path(state.rotating.library._name).resolve(), candidate)
    if original == measured:
        raise ValueError("Two-row timing unexpectedly uses the original terminal device image")
    return dict(original_tail_text=original, tail_text=measured, measured_tail_symbol=KERNEL,
                tail_library=str(candidate), cold_build=state.large_window_launch.source_fingerprint)
