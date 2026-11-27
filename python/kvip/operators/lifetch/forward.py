# SPDX-License-Identifier: Apache-2.0
"""One public LIFetch pipeline for URMA and UBmem."""

import ctypes

from kvip.operators.lifetch.terminal import TerminalLeaseBinding
from kvip.operators.lifetch.streaming import StreamingLIFetchForward


class LIFetchForward(StreamingLIFetchForward):
    def __init__(self, rotating, update, *, q_workers=2):
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
            rotating.parallel_terminal = self.terminal
            if self.terminal.fixed_score_ready is not None:
                self.terminal_function = rotating.library.kvip_fixed_q4_small_forward
            elif getattr(rotating, "large_four_window_credit", False):
                self.terminal_function = getattr(
                    rotating.library, "kvip_large_four_window_credit_forward", rotating.library.kvip_parallel_tail_forward
                )
            else:
                self.terminal_function = rotating.library.kvip_parallel_tail_forward
            self.terminal_function.argtypes = [ctypes.c_void_p] * 4
            self.terminal_function.restype = ctypes.c_uint32

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
