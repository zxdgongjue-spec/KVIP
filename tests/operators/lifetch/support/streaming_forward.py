"""Isolated LI/Resolver/Index/URMA streaming experiment binding.

All storage is prepared before graph capture. Round state is initialized by
the experiment's existing v9_init kernel, not by Python in forward.
"""
import ctypes

import numpy as np


class StreamingForward:
    def __init__(self, rotating, update, torch, prefetch_limit):
        self.rotating, self.update, self.torch = rotating, update, torch
        shape = rotating.runtime_shape
        self.batch, self.history, self.resident = shape.batch, shape.history, shape.resident
        if not (1 <= self.batch <= 16 and self.history == 131072 and self.resident == 16384 and shape.indexHeads == 32):
            raise ValueError('streaming experiment supports B1..16, history=131072, resident=16384, index_heads=32')
        if prefetch_limit > 512:
            raise ValueError('streaming experiment supports at most 512 prefetches per request')
        keys, scales, addresses, logical = update.storage[:4]
        b, device = self.batch, rotating.device
        self.flags = torch.zeros((b, 5, 32), dtype=torch.int32, device=device)
        self.events_src = torch.empty((b, 512 + 8192), dtype=torch.int32, device=device)
        self.events_dst = torch.empty_like(self.events_src)
        self.worker_status = torch.zeros((b, 32), dtype=torch.int32, device=device)
        self.times = torch.zeros((b, 64), dtype=torch.int64, device=device)
        self.query_bitmaps = torch.empty((b, 4, 512), dtype=torch.int32, device=device)
        self.query_errors = torch.zeros((b, 4, 32), dtype=torch.int32, device=device)
        pointers = [rotating.context.data_ptr(), keys.data_ptr(), scales.data_ptr(), addresses.data_ptr(),
                    logical.data_ptr(), rotating.hot_valid.data_ptr(), self.flags.data_ptr(),
                    self.events_src.data_ptr(), self.events_dst.data_ptr(), self.worker_status.data_ptr(),
                    self.times.data_ptr()]
        geometry = [self.history, self.resident, 512, 512 + 8192]
        trailing = [self.query_bitmaps.data_ptr(), self.query_errors.data_ptr(), update.bindings.data_ptr()]
        blob = np.concatenate((np.asarray(pointers, dtype='<u8').view('u1'),
                               np.asarray(geometry, dtype='<u4').view('u1'),
                               np.asarray(trailing, dtype='<u8').view('u1')))
        if len(blob) != 128:
            raise AssertionError(f'streaming workspace ABI size={len(blob)}, expected 128')
        self.workspace = torch.from_numpy(blob.copy()).to(device)
        rotating.bindings = torch.tensor(
            [x.data_ptr() for x in rotating.arguments] + [self.workspace.data_ptr()],
            dtype=torch.uint64, device=device,
        )
        self.function = rotating.library.kvip_lifetch_streaming_forward
        self.function.argtypes = [ctypes.c_void_p] * 5
        self.function.restype = ctypes.c_uint32

    def __call__(self):
        stream = self.torch.npu.current_stream(self.rotating.device)
        result = self.function(self.rotating.bindings.data_ptr(), self.update.bindings.data_ptr(),
                               self.workspace.data_ptr(), stream.npu_stream, self.rotating.shape_pointer)
        if result:
            raise RuntimeError(f'streaming LIFetch launch failed: {result}')
