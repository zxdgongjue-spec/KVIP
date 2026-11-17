"""Opt-in staged score and current streaming tail, prepared before graph capture."""

import ctypes
import numpy as np
from kvip.operators.lifetch.streaming import StreamingLIFetchForward
from kvip.operators.lifetch.workspace import device_words, score_buffer
from kvip.operators.lifetch.schedule import split_score_tasks


class ScoreWave(ctypes.Structure):
    _fields_ = [(x, ctypes.c_uint64) for x in ("bindings", "scores", "tasks")] + [
        (x, ctypes.c_uint32) for x in ("rows", "stride", "queryGroup", "readyEnd")
    ]


class PrepareWave(ctypes.Structure):
    _fields_ = [(x, ctypes.c_uint64) for x in ("bindings", "tasks")] + [
        (x, ctypes.c_uint32) for x in ("rows", "stride", "queryGroup", "reserved")
    ]


class Plan(ctypes.Structure):
    _fields_ = [
        ("canonical", ctypes.c_uint64),
        ("hot", ctypes.POINTER(ScoreWave)),
        ("cold", ctypes.POINTER(ScoreWave)),
    ] + [(x, ctypes.c_uint32) for x in ("hotCount", "coldCount", "aivBudget", "fetchWorkers")] + [
        ("prepareWaves", ctypes.c_uint64),
    ] + [(x, ctypes.c_uint32) for x in ("prepareCount", "postMode", "prepareMode", "serviceMode")]


class StagedLIFetchForward(StreamingLIFetchForward):
    def __init__(self, rotating, update, *, q_workers=2, aiv_budget=None, route="auto",
                 score_post="separate", score_prepare="wave", score_service="separate",
                 score_hot_partitions=0, score_cold_partitions=0):
        if (score_post not in ("separate", "scatter", "direct") or score_prepare not in ("wave", "invocation", "phase")
                or score_service not in ("separate", "inline")
                or (score_service == "inline" and score_post == "separate")):
            raise ValueError("Invalid score post/prepare mode")
        # Reject stale bridges before passing the extended private Plan.
        for name, expected in (("kvip_staged_plan_bytes", ctypes.sizeof(Plan)), ("kvip_staged_plan_version", 3)):
            fn = getattr(rotating.library, name, None)
            if fn is None:
                raise ValueError("Rebuild LIFetch: staged private Plan ABI v3 is required")
            fn.argtypes, fn.restype = [], ctypes.c_uint32
            if fn() != expected:
                raise ValueError("Rebuild LIFetch: staged private Plan ABI mismatch")
        self.score_post, self.score_prepare, self.score_service = score_post, score_prepare, score_service
        if not getattr(rotating, "staged", False):
            raise ValueError("Prepare LIFetchBinding(staged=True) before allocating its Context")
        if hasattr(rotating, "lease"):
            raise ValueError("The first staged adapter uses the current URMA streaming operator")
        super().__init__(
            rotating,
            update,
            rotating.torch,
            rotating.prefetch_limit,
            fused_final=True,
            q_workers=q_workers,
            aiv_budget=aiv_budget,
            _staged=True,
        )
        self.parallel_finalize = True
        self.prepare_metadata = True
        self.set_options()
        torch, device, s = self.torch, rotating.device, rotating.runtime_shape
        if route not in ("auto", "history", "bsq") or s.rounds < 2:
            raise ValueError("Staged execution requires Hot and Cold phases and a valid route")
        # Startup validation: never read back scores or metadata during forward.
        args = rotating.arguments
        qlens = args[5].detach().cpu().numpy().view(np.uint32)
        klens = args[6].detach().cpu().numpy().view(np.uint32)
        if np.any(qlens[: s.batch] != 4) or np.any(klens[: s.batch] != s.history):
            raise ValueError("First staged score adapter requires allocated uniform Q4/H pages")
        # Owners are populated per round after startup; the selector validates
        # distinct active owners before launching, including during graph replay.
        pool = getattr(rotating, "streaming_scratch", {})

        def allocate(name, shape, dtype, *layout):
            return score_buffer(pool, torch, device, name, shape, dtype, *layout)

        self.canonical = allocate("canonical", (s.batch, 4, s.history), torch.uint16)
        self.storage = [self.canonical]
        cap = min(s.aicCount, self.schedule.aiv_budget // 2)

        def waves(begin, end, partitions):
            qg, _, tasks = split_score_tasks(s.batch, begin, end, s.aicCount, self.schedule.aiv_budget, route,
                                            partitions=partitions)
            result = []
            for first in range(0, len(tasks), cap):
                ts = tasks[first : first + cap]
                rows = len(ts)
                stride = max(x[3] - x[2] for x in ts)
                lengths = torch.tensor([x[3] - x[2] for x in ts], dtype=torch.int32, device=device)
                layout = (begin, end, qg, first)
                q = allocate("query", (rows * qg * s.indexHeads * 128,), torch.uint8, *layout)
                w = allocate("weights", (rows * qg * s.indexHeads,), torch.bfloat16, *layout)
                z = allocate("qscale", (rows * qg * s.indexHeads,), torch.float32, *layout)
                pages = allocate("pages", (rows * stride // 128,), torch.int32, *layout)
                qlen = torch.full((rows,), qg, dtype=torch.int32, device=device)
                score = allocate("scores", (rows * qg * stride,), torch.uint16, *layout)
                selected = allocate("selected", (rows * qg * 2048,), torch.int32, *layout)
                task = torch.tensor(ts, dtype=torch.int32, device=device)
                bindings = [0] * 16
                for index, tensor in (
                    (0, q),
                    (1, args[1]),
                    (2, w),
                    (3, z),
                    (4, args[4]),
                    (5, qlen),
                    (6, lengths),
                    (7, pages),
                    (9, selected),
                    (12, score),
                ):
                    bindings[index] = tensor.data_ptr()
                binding = device_words(torch, bindings, device)
                self.storage.extend((q, w, z, pages, qlen, lengths, score, selected, task, binding))
                result.append(
                    ScoreWave(
                        binding.data_ptr(),
                        score.data_ptr(),
                        task.data_ptr(),
                        rows,
                        stride,
                        qg,
                        end if first + rows == len(tasks) else 0,
                    )
                )
            return result

        hot = waves(0, s.resident, score_hot_partitions)
        cold = []
        begin = s.resident
        for epoch in range(1, s.rounds):
            end = s.resident + ((s.history - s.resident) // 128 * epoch // (s.rounds - 1)) * 128
            cold.extend(waves(begin, end, score_cold_partitions))
            begin = end
        self.hot = (ScoreWave * len(hot))(*hot)
        self.cold = (ScoreWave * len(cold))(*cold)
        # Pointer-rich descriptors are private to this forward, never in the shared scratch pool.
        desc = (PrepareWave * (len(hot) + len(cold)))(*[
            PrepareWave(w.bindings, w.tasks, w.rows, w.stride, w.queryGroup, 0) for w in hot + cold
        ])
        self.prepare_descriptors = torch.tensor(list(bytes(desc)), dtype=torch.uint8, device=device)
        self.storage.append(self.prepare_descriptors)
        self.plan = Plan(
            self.canonical.data_ptr(), self.hot, self.cold, len(hot), len(cold), self.schedule.aiv_budget, q_workers,
            self.prepare_descriptors.data_ptr(), len(desc),
            {"separate": 0, "scatter": 1, "direct": 2}[score_post],
            {"wave": 0, "invocation": 1, "phase": 2}[score_prepare],
            {"separate": 0, "inline": 1}[score_service]
        )
        self.score_geometry = dict(
            route=route, post=score_post, prepare=score_prepare, service=score_service,
            requested_hot_partitions=score_hot_partitions, requested_cold_partitions=score_cold_partitions,
            hot_tasks=sum(w.rows for w in hot), cold_tasks=sum(w.rows for w in cold),
            resident_topk_h_partitions=1, final_topk_h_partitions=1, fetch_workers_per_query=q_workers,
            effective_aic=s.aicCount, aiv_budget=self.schedule.aiv_budget,
            hot_waves=len(hot), cold_waves=len(cold), windows=s.rounds,
            waves=[dict(rows=w.rows, query_group=w.queryGroup, stride=w.stride, ready_end=w.readyEnd)
                   for w in hot + cold],
        )
        self.function = rotating.library.kvip_lifetch_staged_forward
        self.function.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_void_p, ctypes.POINTER(Plan)]
        self.function.restype = ctypes.c_uint32

    def __call__(self):
        stream = self.torch.npu.current_stream(self.rotating.device)
        result = self.function(
            self.rotating.bindings.data_ptr(),
            self.workspace.data_ptr(),
            stream.npu_stream,
            self.shape_pointer,
            ctypes.byref(self.plan),
        )
        if result:
            raise RuntimeError(f"staged LIFetch launch failed: {result}")
