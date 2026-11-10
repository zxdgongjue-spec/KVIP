# SPDX-License-Identifier: Apache-2.0
"""One public launch for LIFetch and its mandatory Index ownership update."""

import ctypes


class LIFetchForward:
    def __init__(self, rotating, update):
        if rotating.device != update.device:
            raise ValueError("Fetch and Index update must use the same device")
        if update.bindings.numel() != 28:
            raise ValueError("LIFetch requires the 28-field Index update/output binding ABI")
        self.rotating, self.update = rotating, update
        # Choose the transport at bind time; keep the per-forward launch and
        # stream checks identical to the original transport implementations.
        self.ubmem = hasattr(rotating, "lease")
        self.function = (rotating.library.kvip_ubmem_forward if self.ubmem
                         else rotating.library.kvip_lifetch_forward)
        self.function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.function.restype = ctypes.c_uint32

    def __call__(self):
        if self.ubmem:
            stream = self.rotating.lease.check_stream()
            shape_pointer = ctypes.byref(self.rotating.runtime_shape)
        else:
            stream = self.rotating.torch.npu.current_stream(self.rotating.device)
            shape_pointer = self.rotating.shape_pointer
        result = self.function(
            self.rotating.bindings.data_ptr(),
            self.update.bindings.data_ptr(),
            stream.npu_stream,
            shape_pointer,
        )
        if result:
            raise RuntimeError(f"LIFetch/Index update launch failed: {result}")
