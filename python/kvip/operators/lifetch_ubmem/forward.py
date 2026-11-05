# SPDX-License-Identifier: Apache-2.0
"""One graph-safe UBmem LI/TopK/Fetch/Index-update submission."""

import ctypes


class UbmemForward:
    def __init__(self, rotating, update):
        if rotating.device != update.device or update.bindings.numel() != 26:
            raise ValueError("UBmem forward requires the same-device 26-field Index update binding")
        self.rotating, self.update = rotating, update
        self.function = rotating.library.kvip_ubmem_forward
        self.function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.function.restype = ctypes.c_uint32

    def __call__(self):
        stream = self.rotating.lease.check_stream()
        result = self.function(self.rotating.bindings.data_ptr(), self.update.bindings.data_ptr(), stream.npu_stream)
        if result:
            raise RuntimeError(f"UBmem LIFetch/Index update launch failed: {result}")
