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
        self.legacy_layout = False
        self.parallel_finalize = False
        self.flags = torch.zeros((b, 12 if q_workers else (7 if fused_final else 5), 32), dtype=torch.int32, device=device)
        self.events_src = torch.empty((b, 512 + 8192), dtype=torch.int32, device=device)
        self.events_dst = torch.empty_like(self.events_src)
        self.worker_status = torch.zeros((b, 32), dtype=torch.int32, device=device)
        # First 64 words/request are resolver/update, then 16 words per Q.
        self.times = torch.zeros((b * (256 if q_workers else (224 if fused_final else 128)) + ((2*b+4*q_workers)*32 if q_workers else 0),), dtype=torch.int64, device=device)
        self.query_bitmaps = torch.empty((b, 4, 512), dtype=torch.int32, device=device)
        self.query_errors = torch.zeros((b, 4, 32), dtype=torch.int32, device=device)
        # Dense fallback holds two uint32 permutations and two uint64 snapshots.
        self.dense_shadow = (torch.empty((b, 3 * self.history), dtype=torch.int64, device=device)
                             if q_workers else None)
        # Each Q worker owns independent packed indices, offsets and destinations.
        self.packed_fetch = (torch.empty((4 * q_workers, 9 * b * 2048 + ((3 * b + 31) // 32) * 32 + 64),
                                        dtype=torch.int32, device=device)
                             if q_workers else None)
        pointers = [rotating.context.data_ptr(), keys.data_ptr(), scales.data_ptr(), addresses.data_ptr(),
                    logical.data_ptr(), rotating.hot_valid.data_ptr(), self.flags.data_ptr(),
                    self.events_src.data_ptr(), self.events_dst.data_ptr(), self.worker_status.data_ptr(),
                    self.times.data_ptr()]
        geometry = [self.history, self.resident, 512 | ((q_workers or 0) << 16), 512 + 8192]
        trailing = [self.query_bitmaps.data_ptr(), self.query_errors.data_ptr(), update.bindings.data_ptr()]
        blob = np.concatenate((np.asarray(pointers, dtype='<u8').view('u1'),
                               np.asarray(geometry, dtype='<u4').view('u1'),
                               np.asarray(trailing, dtype='<u8').view('u1')))
        if q_workers:
            blob = np.concatenate((blob, np.asarray([self.dense_shadow.data_ptr(), self.packed_fetch.data_ptr()], dtype='<u8').view('u1'),
                                   np.zeros(112, dtype='u1')))
        expected_size = 256 if q_workers else 128
        if len(blob) != expected_size:
            raise AssertionError(f'streaming workspace ABI size={len(blob)}, expected {expected_size}')
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

    def calibrate_counter_mhz(self):
        """Approximate cycle frequency against a long independent graph event."""
        if not self.q_workers:
            return None
        torch = self.torch
        output = torch.empty(4, dtype=torch.int64, device=self.rotating.device)
        function = self.rotating.library.kvip_counter_calibration
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        function.restype = ctypes.c_uint32
        stream = torch.npu.current_stream(self.rotating.device)
        capture = torch.npu.Stream(device=self.rotating.device)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=capture):
            if function(output.data_ptr(), capture.npu_stream):
                raise RuntimeError('counter calibration launch failed')
        capture.synchronize()
        samples = []
        for _ in range(3):
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record(stream)
            graph.replay()
            end.record(stream)
            end.synchronize()
            values = output.cpu().numpy().astype(np.uint64)
            elapsed_us = start.elapsed_time(end) * 1000
            cycles = int(values[1]) - int(values[0])
            if elapsed_us <= 0 or cycles < 20_000_000:
                raise RuntimeError('counter calibration returned invalid timing')
            samples.append(cycles / elapsed_us)
        graph.reset()
        return float(np.median(samples))

    def set_deferred_fetch(self, enabled):
        # Isolation-only mode, set before launch; never used for graph timing.
        value = np.asarray([512 | ((self.q_workers or 0) << 16) | (0x20000000 if self.legacy_layout else 0) | (0x10000000 if self.parallel_finalize else 0) | (0x40000000 if enabled else 0)], dtype='<u4').view('u1')
        self.workspace[96:100].copy_(self.torch.from_numpy(value.copy()))

    def set_diagnostics(self, enabled):
        # Existing prefetchCount field is unused by the streaming kernel; its
        # high bit gates timestamps only in the separate profiled pass.
        value = np.asarray([512 | ((self.q_workers or 0) << 16) | (0x20000000 if self.legacy_layout else 0) | (0x10000000 if self.parallel_finalize else 0) | (0x80000000 if enabled else 0)], dtype='<u4').view('u1')
        self.workspace[96:100].copy_(self.torch.from_numpy(value.copy()))

    def diagnostic_cycles(self):
        raw = self.times.cpu().numpy().astype(np.uint64)
        rows = raw[:self.batch * 64].reshape(self.batch, 64)
        li = raw[self.batch * 64:self.batch * 128].reshape(self.batch, 4, 16)
        fetch = raw[self.batch * 128:self.batch * (256 if self.q_workers else 224)].reshape(self.batch, 128 if self.q_workers else 96) if self.fused_final else None
        flags = self.flags.cpu().numpy()
        posts = (self.rotating.final_posts.cpu().numpy() if hasattr(self, "rotating") else None)
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
                    group_prepare=int(f[q*24+18]), group_pack=int(f[q*24+19]))
                    for q in range(4)], finalize=dict(prefetch_join=int(f[100]-f[96]), prefetch_polls=int(f[101]),
                    prefetch_cq_poll_cycles=int(f[102]), prefetch_last_cq_observed=int(f[103]), wait=int(f[97]-f[100]),
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
            final_counts = ([dict(total=int(post[0]), safe=int(post[5]), conflict=int(post[0]-post[5]))
                             for post in posts[b]] if posts is not None else None)
            result.append(dict(batch=b, li=queries, producer=producer, update=update, fetch=fetch_detail,
                               final_counts=final_counts, prefetch_changes=counts[0], q_new_misses=counts[1:],
                               update_wait=waits,
                               input_and_bitmap=int(p[2]-p[1]), initialize=int(p[3]-p[2]),
                               metadata=int(u[23]-u[22]), consumer_start=int(u[10]),
                               consumer_finish=int(u[24]), producer_finish=int(p[8])))
        if self.q_workers and hasattr(self, "rotating"):
            events = self.rotating.events.cpu().numpy().reshape(-1, 128).view('<u8')
            first = events[:, 11]
            last = events[:, 12]
            first_nonzero = first[first != 0]
            last_nonzero = last[last != 0]
            rows = self.rotating.rows.cpu().numpy().view(np.uint32).reshape(self.batch, -1)
            prefetch_count = int(rows[:, 9].astype(np.uint64).sum())
            finish = [int(fetch[b, 103]) for b in range(self.batch) if int(rows[b, 9]) and int(fetch[b, 103])]
            if prefetch_count and len(first_nonzero) and finish:
                start = int(first_nonzero.min())
                end = max(finish)
                result[0]['prefetch_scope'] = dict(
                    requests=prefetch_count, bytes=prefetch_count * 656,
                    first_doorbell=start, last_doorbell=int(last_nonzero.max()),
                    last_cq_observed=end, interval_cycles=max(0, end - start),
                    note='first doorbell to last CQ observed; includes polling delay and LI-window gaps')
        if self.q_workers:
            result[0]['core_timeline_raw'] = dict(
                batch=self.batch, workers_per_query=self.q_workers,
                clock_mhz=getattr(self, 'counter_clock_mhz', None),
                cores=raw[self.batch*256:].reshape(2*self.batch+4*self.q_workers, 32).tolist(),
                producer_update=raw[:self.batch*64].reshape(self.batch, 64).tolist(),
                fetch=fetch.tolist(), flags=flags.tolist())
        return result
