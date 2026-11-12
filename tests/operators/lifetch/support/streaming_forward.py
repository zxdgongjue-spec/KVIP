"""Diagnostic extensions for the production streaming LIFetch binding."""
import ctypes

import numpy as np

from kvip.operators.lifetch.forward import LIFetchForward


class StreamingForward(LIFetchForward):
    def __init__(self, rotating, update, torch, prefetch_limit, *, fused_final=True, q_workers=2):
        if torch is not rotating.torch or prefetch_limit != rotating.prefetch_limit or not fused_final:
            raise ValueError("Diagnostics must use the exact public LIFetch binding and budget")
        super().__init__(rotating, update, q_workers=q_workers)

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
        self.set_options(deferred_fetch=enabled)

    def set_diagnostics(self, enabled):
        self.set_options(diagnostics=enabled)

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
            if fetch_detail is not None and hasattr(self.rotating, "ubmem_status"):
                fetch_detail["transport"] = "ubmem"
                fetch_detail["ubmem_queries"] = [dict(
                    query_ready_wait=int(fetch[b,q*24+1]-fetch[b,q*24]),
                    dependency_classify=int(fetch[b,q*24+2]-fetch[b,q*24+1]),
                    safe_get=int(fetch[b,q*24+4]-fetch[b,q*24+3]),
                    dependency_wait=int(fetch[b,q*24+6]-fetch[b,q*24+5]),
                    conflict_get=int(fetch[b,q*24+8]-fetch[b,q*24+7]),
                    safe_entries=int(fetch[b,q*24+21]), conflict_entries=int(fetch[b,q*24+15]),
                    dependency_mask=int(fetch[b,q*24+19]),
                    safe_calls=int(fetch[b,q*24+16]), conflict_calls=int(fetch[b,q*24+17]),
                    completion_and_index_join=int(fetch[b,q*24+9]-fetch[b,q*24+8]))
                    for q in range(4)]
                fetch_detail["note"] = "UBmem safe/conflict Gets include KVAF MTE pipeline drain; no CQ"
            counts = [int(flags[b, phase, 1]) for phase in range(5)]
            final_counts = ([dict(total=int(post[0]), safe=int(post[5]), conflict=int(post[0]-post[5]))
                             for post in posts[b]] if posts is not None else None)
            result.append(dict(batch=b, li=queries, producer=producer, update=update, fetch=fetch_detail,
                               final_counts=final_counts, prefetch_changes=counts[0], q_new_misses=counts[1:],
                               update_wait=waits,
                               input_and_bitmap=int(p[2]-p[1]), initialize=int(p[3]-p[2]),
                               metadata=int(u[23]-u[22]), consumer_start=int(u[10]),
                               consumer_finish=int(u[24]), producer_finish=int(p[8])))
        if self.q_workers and hasattr(self, "rotating") and not hasattr(self.rotating, "ubmem_status"):
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
                    requests=prefetch_count, bytes=prefetch_count * self.rotating.runtime_shape.tokenBytes,
                    first_doorbell=start, last_doorbell=int(last_nonzero.max()),
                    last_cq_observed=end, interval_cycles=max(0, end - start),
                    note='first doorbell to last CQ observed; includes polling delay and LI-window gaps')
        if self.q_workers:
            result[0]['core_timeline_raw'] = dict(
                batch=self.batch, workers_per_query=self.q_workers, resolver_cores=self.resolver_cores,
                token_bytes=self.rotating.runtime_shape.tokenBytes,
                clock_mhz=getattr(self, 'counter_clock_mhz', None),
                cores=raw[self.batch*256:].reshape(self.resolver_cores*self.batch+4*self.q_workers, 32).tolist(),
                producer_update=raw[:self.batch*64].reshape(self.batch, 64).tolist(),
                fetch=fetch.tolist(), flags=flags.tolist())
        return result
