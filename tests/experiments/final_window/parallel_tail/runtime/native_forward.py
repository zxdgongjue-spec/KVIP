"""Experimental native dispatch; no host tensor work occurs during forward.

This is a call-chain adapter, not a qualification claim. The AB runner must
validate the generated library and full state lifecycle before acceptance.
"""
import ctypes

from live_binding import TerminalLeaseBinding


class ParallelTerminalForward:
    def __init__(self, forward, prefix_library, terminal_library):
        self.original = forward
        self.prefix_library = prefix_library
        self.terminal_library = terminal_library
        version = terminal_library.kvip_parallel_tail_abi_version
        size = terminal_library.kvip_parallel_tail_binding_bytes
        for function in (version, size):
            function.argtypes = []
            function.restype = ctypes.c_uint32
        if version() != 1 or size() != 128:
            raise ValueError('Private terminal runtime ABI mismatch')
        self.prefix = prefix_library.kvip_parallel_tail_prefix
        self.prefix.argtypes = [ctypes.c_void_p] * 3
        self.prefix.restype = ctypes.c_uint32
        self.prefix_pointer = ctypes.cast(self.prefix, ctypes.c_void_p)
        self.binding = TerminalLeaseBinding(forward.rotating)
        self.binding.attach_workspace(forward)
        self.function = terminal_library.kvip_parallel_tail_forward
        self.function.argtypes = [ctypes.c_void_p] * 5
        self.function.restype = ctypes.c_uint32

    def __call__(self):
        original = self.original
        rotating = original.rotating
        stream = (rotating.lease.check_stream() if hasattr(rotating, 'lease')
                  else rotating.torch.npu.current_stream(rotating.device))
        code = self.function(self.prefix_pointer, original.bindings.data_ptr(),
                             self.binding.workspace.data_ptr(), stream.npu_stream,
                             original.shape_pointer)
        if code:
            raise RuntimeError(f'Private LIFetch terminal launch failed: {code}')

    def __getattr__(self, name):
        # Snapshot collection continues to use the original public buffers.
        return getattr(self.original, name)

    def set_options(self, *, diagnostics=False, deferred_fetch=False):
        if deferred_fetch:
            raise ValueError('Private terminal owns its CQ completion; no deferred-fetch mode')
        self.original.set_options(diagnostics=diagnostics)
        torch = self.original.torch
        value = torch.tensor(list(int(diagnostics).to_bytes(4, 'little')),
                             dtype=torch.uint8, device=self.original.rotating.device)
        # Startup/separate diagnostic capture only; never called by forward.
        self.binding.workspace[52:56].copy_(value)
