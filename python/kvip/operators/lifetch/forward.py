# SPDX-License-Identifier: Apache-2.0
"""One public LIFetch pipeline for URMA and UBmem."""

import ctypes
import os

from kvip.operators.lifetch.terminal import TerminalLeaseBinding
from kvip.operators.lifetch.streaming import StreamingLIFetchForward


class LIFetchForward(StreamingLIFetchForward):
    def __init__(
        self, rotating, update, *, q_workers=2, window_q=None, large_window_queries_per_worker=None,
    ):
        """Bind the final-window route before graph capture.

        The default enables supported Q1 or AUTO routes. Set window_q=False
        to select the native window path for this binding.
        """
        if window_q is None:
            switch = os.environ.get("KVIP_LIFETCH_WINDOW_Q", "1")
            if switch not in ("0", "1"):
                raise ValueError("KVIP_LIFETCH_WINDOW_Q must be 0 or 1")
            window_q = switch == "1"
        if not isinstance(window_q, bool):
            raise ValueError("window_q must be a boolean")
        large_mode = large_window_queries_per_worker
        if large_mode is not None and (type(large_mode) is not int or large_mode not in range(5) or not window_q):
            raise ValueError("An explicit large-window policy requires window_q and a mode in 0..4")
        parallel_terminal = False
        if not hasattr(rotating, "lease"):
            supported = rotating.library.kvip_parallel_tail_supported
            supported.argtypes, supported.restype = [ctypes.c_void_p], ctypes.c_uint32
            parallel_terminal = bool(supported(rotating.shape_pointer))
        super().__init__(
            rotating, update, rotating.torch, rotating.prefetch_limit, fused_final=True, q_workers=q_workers,
            parallel_terminal=parallel_terminal,
        )
        self.prepare_metadata = True
        self.ubmem_overlap_fetch = hasattr(rotating, "lease")
        self.parallel_finalize = True
        self.terminal = None
        self.set_options()
        if parallel_terminal:
            self.terminal = TerminalLeaseBinding(rotating)
            self.terminal.attach_workspace(self)
            self.terminal_function = (
                rotating.library.kvip_fixed_q4_small_forward if self.terminal.fixed_score_ready is not None
                else rotating.library.kvip_parallel_tail_forward
            )
            self.terminal_function.argtypes = [ctypes.c_void_p] * 4
            self.terminal_function.restype = ctypes.c_uint32
            if window_q:
                self._bind_window_q(rotating, large_mode)
            rotating.parallel_terminal = self.terminal
        elif large_mode is not None:
            raise ValueError("A large-window policy requires a supported URMA terminal")

    def _bind_window_q(self, rotating, large_mode):
        # Check the native capability at bind time. Bridges predating the
        # final-window interface and non-matching shapes keep the native route.
        library = rotating.library
        for large, prefix in ((True, "kvip_large_window_final"), (False, "kvip_fixed_q4_window_q1")):
            if not large and large_mode is not None:
                break
            supported = getattr(library, f"{prefix}_supported", None)
            launch = getattr(library, f"{prefix}_forward", None)
            if supported is None or launch is None:
                continue
            supported.argtypes = [ctypes.c_void_p]
            supported.restype = ctypes.c_uint32
            if not supported(rotating.shape_pointer):
                continue
            self.terminal.attach_window_q(large=large)
            launch.argtypes = [ctypes.c_void_p] * 4 + ([ctypes.c_uint32] if large else [])
            launch.restype = ctypes.c_uint32
            if large:
                mode = 0 if large_mode is None else large_mode
                self.terminal.large_window_queries_per_worker = mode

                def launch_with_mode(bindings, workspace, stream, shape):
                    return launch(bindings, workspace, stream, shape, mode)

                self.terminal_function = launch_with_mode
            else:
                self.terminal_function = launch
            return
        if large_mode is not None:
            raise ValueError("The explicit large-window policy is unavailable for this shape or bridge")

    def __call__(self):
        if self.terminal is None:
            return super().__call__()
        stream = self.rotating.owner.lease.check_stream()
        result = self.terminal_function(
            self.bindings.data_ptr(), self.terminal.workspace.data_ptr(), stream.npu_stream, self.shape_pointer
        )
        if result:
            raise RuntimeError(f"LIFetch terminal launch failed: {result}")

    def set_options(self, *, diagnostics=False, deferred_fetch=False):
        terminal = getattr(self, "terminal", None)
        if terminal is not None and deferred_fetch:
            raise ValueError("Parallel terminal completes Fetch internally")
        super().set_options(diagnostics=diagnostics, deferred_fetch=deferred_fetch)
        if terminal is not None:
            value = self.torch.tensor(
                list(int(diagnostics).to_bytes(4, "little")), dtype=self.torch.uint8, device=self.rotating.device
            )
            terminal.workspace[52:56].copy_(value)
