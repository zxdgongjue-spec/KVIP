"""Isolated LI/Resolver/Index/URMA streaming experiment binding.

All storage is prepared before graph capture. Round state is initialized by
the experiment's existing v9_init kernel, not by Python in forward.
"""
import ctypes

import numpy as np


class StreamingForward:
    def __init__(self, rotating, update, torch, prefetch_limit, *, fused_final=False, q_workers=None):
        self.rotating, self.update, self.torch = rotating, update, torch
        shape = rotating.runtime_shape
        self.batch, self.history, self.resident = shape.batch, shape.history, shape.resident
        if not (1 <= self.batch <= 16 and self.history == 131072 and self.resident == 16384 and shape.indexHeads == 32):
            raise ValueError('streaming experiment supports B1..16, history=131072, resident=16384, index_heads=32')
        if prefetch_limit > 512:
            raise ValueError('streaming experiment supports at most 512 prefetches per request')
        keys, scales, addresses, logical = update.storage[:4]
        b, device = self.batch, rotating.device
        self.fused_final = fused_final
        self.q_workers = q_workers
        self.flags = torch.zeros((b, 12 if q_workers else (7 if fused_final else 5), 32), dtype=torch.int32, device=device)
        self.events_src = torch.empty((b, 512 + 8192), dtype=torch.int32, device=device)
        self.events_dst = torch.empty_like(self.events_src)
        self.worker_status = torch.zeros((b, 32), dtype=torch.int32, device=device)
        # First 64 words/request are resolver/update, then 16 words per Q.
        self.times = torch.zeros((b * (256 if q_workers else (224 if fused_final else 128)),), dtype=torch.int64, device=device)
        self.query_bitmaps = torch.empty((b, 4, 512), dtype=torch.int32, device=device)
        self.query_errors = torch.zeros((b, 4, 32), dtype=torch.int32, device=device)
        pointers = [rotating.context.data_ptr(), keys.data_ptr(), scales.data_ptr(), addresses.data_ptr(),
                    logical.data_ptr(), rotating.hot_valid.data_ptr(), self.flags.data_ptr(),
                    self.events_src.data_ptr(), self.events_dst.data_ptr(), self.worker_status.data_ptr(),
                    self.times.data_ptr()]
        geometry = [self.history, self.resident, 512 | ((q_workers or 0) << 16), 512 + 8192]
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
        self.function.argtypes = [ctypes.c_void_p] * 5 + ([ctypes.c_uint32] if q_workers else [])
        self.function.restype = ctypes.c_uint32

    def __call__(self):
        stream = self.torch.npu.current_stream(self.rotating.device)
        args = [self.rotating.bindings.data_ptr(), self.update.bindings.data_ptr(),
                self.workspace.data_ptr(), stream.npu_stream, self.rotating.shape_pointer]
        if self.q_workers:
            args.append(self.q_workers)
        result = self.function(*args)
        if result:
            raise RuntimeError(f'streaming LIFetch launch failed: {result}')

    def set_deferred_fetch(self, enabled):
        # Isolation-only mode, set before launch; never used for graph timing.
        value = np.asarray([512 | ((self.q_workers or 0) << 16) | (0x40000000 if enabled else 0)], dtype='<u4').view('u1')
        self.workspace[96:100].copy_(self.torch.from_numpy(value.copy()))

    def set_diagnostics(self, enabled):
        # Existing prefetchCount field is unused by the streaming kernel; its
        # high bit gates timestamps only in the separate profiled pass.
        value = np.asarray([512 | ((self.q_workers or 0) << 16) | (0x80000000 if enabled else 0)], dtype='<u4').view('u1')
        self.workspace[96:100].copy_(self.torch.from_numpy(value.copy()))

    def diagnostic_cycles(self):
        raw = self.times.cpu().numpy().astype(np.uint64)
        rows = raw[:self.batch * 64].reshape(self.batch, 64)
        li = raw[self.batch * 64:self.batch * 128].reshape(self.batch, 4, 16)
        fetch = raw[self.batch * 128:].reshape(self.batch, 128 if self.q_workers else 96) if self.fused_final else None
        flags = self.flags.cpu().numpy()
        result = []
        for b in range(self.batch):
            p, u = rows[b, :32], rows[b, 32:]
            queries = []
            for q in range(4):
                t = li[b, q]
                if not all(int(v) for v in t[:8]) or not int(t[10]) or not int(t[11]):
                    raise AssertionError(f'missing LI diagnostic timestamps: batch={b} query={q}')
                queries.append(dict(mapping_wait=int(t[1]-t[0]), barrier=int(t[2]-t[1]),
                                    load_first=int(t[3]-t[2]), classify_first=int(t[4]-t[3]),
                                    wait_second=int(t[5]-t[4]), classify_second=int(t[6]-t[5]),
                                    bitmap_merge=int(t[11]-t[10]), publish=int(t[7]-t[11]),
                                    overlap=bool(t[8]), topk_only=bool(t[9])))
            producer = []
            for q in range(4):
                t = p[10+5*q:15+5*q]
                if not all(int(v) for v in t):
                    raise AssertionError(f'missing resolver diagnostic timestamps: batch={b} query={q}')
                producer.append(dict(resolve=int(t[1]-t[0]), output=int(t[2]-t[1]),
                                     pack=int(t[3]-t[2]), publish=int(t[4]-t[3])))
            update = [int(u[13+2*q]-u[12+2*q]) for q in range(5)]
            waits = [int(u[12]-u[10])] + [int(u[12+2*q]-u[11+2*q]) for q in range(1, 5)]
            fetch_detail = None
            if fetch is not None and self.q_workers:
                f = fetch[b]
                fetch_detail = dict(queries=[dict(wait=int(f[q*24+1]-f[q*24]),
                    prepare=int(f[q*24+2]-f[q*24+1]), safe_submit_or_sync=int(f[q*24+4]-f[q*24+3]),
                    prefetch_join=int(f[q*24+6]-f[q*24+5]) if f[q*24+5] else 0,
                    conflict_get=int(f[q*24+8]-f[q*24+7]), safe_post=int(f[q*24+12]),
                    safe_cq_wait=int(f[q*24+13]), conflict_post=int(f[q*24+14]),
                    conflict_cq_wait=int(f[q*24+15]), safe_groups=int(f[q*24+16]),
                    conflict_groups=int(f[q*24+17]), safe_mode=int(f[q*24+20]), total=int(f[q*24+9]-f[q*24]),
                    final_prefetch_join=int(f[q*24+19]-f[q*24+18]) if f[q*24+18] else 0)
                    for q in range(4)], finalize=dict(wait=int(f[97]-f[96]),
                    residency=int(f[98]-f[97]), status=int(f[99]-f[98])))
            elif fetch is not None:
                f = fetch[b]
                fetch_detail = dict(queries=[dict(wait=int(f[q*10+1]-f[q*10]),
                    prepare=int(f[q*10+2]-f[q*10+1]), safe_get=int(f[q*10+4]-f[q*10+3]),
                    prefetch_join=int(f[q*10+6]-f[q*10+5]) if f[q*10+5] else 0,
                    conflict_get=int(f[q*10+8]-f[q*10+7]),
                    safe_post=int(f[64+q*6]), safe_cq_wait=int(f[65+q*6]),
                    conflict_post=int(f[66+q*6]), conflict_cq_wait=int(f[67+q*6]),
                    safe_groups=int(f[68+q*6]), conflict_groups=int(f[69+q*6]),
                    total=int(f[q*10+9]-f[q*10])) for q in range(4)],
                    final_prefetch_join=int(f[50]-f[49]), wait_producer_update=int(f[51]-f[50]),
                    collect=int(f[52]-f[51]), residency=int(f[53]-f[52]),
                    write_status=int(f[54]-f[53]), span=int(f[54]-f[48]))
            counts = [int(flags[b, phase, 1]) for phase in range(5)]
            result.append(dict(batch=b, li=queries, producer=producer, update=update, fetch=fetch_detail,
                               prefetch_changes=counts[0], q_new_misses=counts[1:],
                               update_wait=waits,
                               input_and_bitmap=int(p[2]-p[1]), initialize=int(p[3]-p[2]),
                               metadata=int(u[23]-u[22]), consumer_start=int(u[10]),
                               consumer_finish=int(u[24]), producer_finish=int(p[8])))
        return result
