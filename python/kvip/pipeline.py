# SPDX-License-Identifier: Apache-2.0
"""One public launch for LIFetch and its mandatory Index ownership update."""

import ctypes


class FusedForward:
    def __init__(self, rotating, update):
        if rotating.device != update.device:
            raise ValueError("Fetch and Index update must use the same device")
        if update.bindings.numel() != 24:
            raise ValueError("LIFetch requires the 24-field Index update/output binding ABI")
        self.rotating, self.update = rotating, update
        self.function = rotating.library.kvip_lifetch_forward
        self.function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.function.restype = ctypes.c_uint32

    def __call__(self):
        stream = self.rotating.torch.npu.current_stream(self.rotating.device)
        result = self.function(self.rotating.bindings.data_ptr(), self.update.bindings.data_ptr(), stream.npu_stream)
        if result:
            raise RuntimeError(f"LIFetch/Index update launch failed: {result}")
