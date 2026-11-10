# SPDX-License-Identifier: Apache-2.0
"""Candidate-ready group-call timing, not a reconstruction of the service step.

The archive contains one group's already selected LRU candidates. It cannot
reproduce scheduler/LRU work or the shared preparation of all layer groups.
Only the legacy inverse-map construction and candidate staging can be replayed
here. Allocation, validation, archive IO, state reset and oracle stay untimed.
"""

from time import perf_counter_ns

import numpy as np

from kvip.adapters.vllm.inputs import reserved_hot_reference
from kvip.diagnostics.replay_measure import statistics


class CandidatePreparation:
    def __init__(self, torch, archive, bundle):
        rotating = bundle.rotating
        self.compact = bool(bundle.tuning.get("compact_reserved", False))
        self.candidates = archive.array("pre.lru_slots").copy()
        ends = archive.array("pre.query_ends")
        resident = archive.manifest["geometry"]["resident"]
        expected = reserved_hot_reference(self.candidates, ends, resident)
        self.active_rows = np.flatnonzero(np.any(ends.reshape(-1, 4) > 0, axis=1))
        if not np.array_equal(expected[self.active_rows], archive.array("pre.reserved_hot")[self.active_rows]):
            raise ValueError("Captured protection map cannot be reproduced from its LRU candidates")
        self.ordinals = np.arange(self.candidates.shape[1], dtype=np.int32)
        self.lru_device, self.reserved_device = rotating.lru_slots, rotating.reserved_hot
        self.lru_host = torch.empty(self.candidates.shape, dtype=torch.int32, device="cpu", pin_memory=True)
        self.lru_values = self.lru_host.numpy()
        self.reserved_host = None
        if not self.compact:
            self.reserved_host = torch.empty(expected.shape, dtype=torch.int32, device="cpu", pin_memory=True)
            self.reserved_values = self.reserved_host.numpy()
        self.metadata = dict(
            candidate_h2d_bytes=int(self.candidates.nbytes),
            dense_protection_h2d_bytes=0 if self.compact else int(expected.nbytes),
            dense_protection_generated_on_device=self.compact,
            batch_capacity=self.candidates.shape[0],
            active_rows=len(self.active_rows),
            group_count=1,
            includes_service_guard_row=False,
        )

    def poison(self):
        # Outside timing, ensure neither copy can silently use reset's valid
        # input. Compact init must recreate its protection table every call.
        self.lru_device.fill_(-2)
        self.reserved_device.fill_(-2)

    def prepare_cpu(self):
        # Pinned buffers and ndarray views already exist. Reproduce the dense
        # fill/scatter, without the diagnostic reference's allocation/checks.
        np.copyto(self.lru_values, self.candidates)
        if self.reserved_host is not None:
            self.reserved_values.fill(-1)
            for row in self.active_rows:
                self.reserved_values[row, self.lru_values[row]] = self.ordinals

    def upload(self):
        self.lru_device.copy_(self.lru_host, non_blocking=True)
        if self.reserved_host is not None:
            self.reserved_device.copy_(self.reserved_host, non_blocking=True)


def measure_candidate_call(torch, stream, bundle, args, replay_graph, retained):
    """Host wall time has no device timing events inside its measured interval."""
    preparation = bundle.candidate_preparation
    wall, cpu, reads = [], [], []
    for step in range(args.warmup + args.iterations):
        bundle.reset()
        preparation.poison()
        stream.synchronize()
        start = perf_counter_ns()
        preparation.prepare_cpu()
        prepared = perf_counter_ns()
        preparation.upload()
        replay_graph()
        stream.synchronize()  # Owns pinned source reuse and output completion.
        finished = perf_counter_ns()
        counts = bundle.verify()
        if step >= args.warmup:
            wall.append((finished - start) / 1000)
            cpu.append((prepared - start) / 1000)
            reads.append(counts)
    result = dict(
        scope="single_group_candidate_ready_to_output_ready",
        wall=statistics(wall),
        cpu_prepare=statistics(cpu),
        reads_samples=reads,
        payload=preparation.metadata,
        full_service_step_measured=False,
        method="Host monotonic clock; CPU staging/inverse map, H2D enqueue, graph replay and stream completion",
        exclusions=(
            "LRU candidate selection/feedback, scheduler and owner bookkeeping, shared multi-group preparation, "
            "ABC maintenance, new Index ingest, Q packing, Put, archive IO, DDR restoration and oracle"
        ),
        interpretation=(
            "Isolated single-group invocation; not a full Target step, TPOT, or asynchronously exposed service cost. "
            "Do not multiply this latency by the number of layer groups. CPU is a wall-time subset; "
            "separately measured device/H2D timings are not additive components of wall time."
        ),
    )
    if args.stage_iterations:
        # These events perturb submission. Keep them out of the host-wall pass,
        # and label H2D as a separate diagnostic, like existing stage timings.
        begin, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
        retained.setdefault("events", []).extend((begin, end))
        samples = []
        for step in range(1 + args.stage_iterations):
            bundle.reset()
            preparation.poison()
            stream.synchronize()
            preparation.prepare_cpu()
            begin.record(stream)
            preparation.upload()
            end.record(stream)
            replay_graph()
            stream.synchronize()
            elapsed = begin.elapsed_time(end) * 1000
            bundle.verify()
            if step:
                samples.append(elapsed)
        result["h2d_diagnostic"] = dict(
            **statistics(samples),
            method="Separate same-stream event interval around candidate/protection H2D; excludes CPU construction",
        )
    return result
