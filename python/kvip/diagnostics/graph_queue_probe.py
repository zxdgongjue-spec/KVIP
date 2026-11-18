# SPDX-License-Identifier: Apache-2.0
"""Measure submission sensitivity outside the headline and production graph."""

import math
from time import perf_counter_ns

# 128 MiB of scratch. All copies precede the start marker on the graph's stream.
SCRATCH_BYTES = 64 << 20
COPIES = 16
FENCED = (True, False, False, True)
PROTOCOL = "synchronized_start_with_empty_control"


def probe(torch, stream, begin, end, replay, prepare, verify, first_step, expected_get):
    if not all(callable(getattr(begin, method, None)) for method in ("query", "synchronize")):
        return dict(status="unavailable", reason="timing_event_query_or_wait_missing", samples=[])

    def empty():
        pass

    def interval(call, fenced):
        for _ in range(COPIES):
            target.copy_(source, non_blocking=True)
        begin.record(stream)
        if fenced:
            # Force the start to execute before issuing the graph. Fencing
            # before record() leaves both arms pending on async host queues.
            begin.synchronize()
        t0 = perf_counter_ns()
        call()
        host_us = (perf_counter_ns() - t0) / 1000
        end.record(stream)
        pending = not begin.query()
        end.synchronize()
        elapsed = float(begin.elapsed_time(end)) * 1000
        if not math.isfinite(elapsed) or elapsed < 0 or (call is replay and elapsed == 0):
            raise ValueError(f"Invalid queue-probe duration: {elapsed}")
        return dict(begin_pending_after_submit=pending, elapsed_us=elapsed, host_call_us=host_us)

    rows = []
    with torch.npu.stream(stream):
        source = torch.zeros(SCRATCH_BYTES, dtype=torch.uint8, device=stream.device)
        target = torch.empty_like(source)
        target.copy_(source)
        stream.synchronize()
        for i, fenced in enumerate(FENCED):
            step = first_step + i
            prepare(step)
            stream.synchronize()
            # Begin.synchronize() includes device notification and a host
            # wake-up. Measure the same bracket without submitting a graph.
            # Alternate graph/control order within *each* arm.
            if i < 2:
                blank = interval(empty, fenced)
            graph = interval(replay, fenced)
            if i >= 2:
                blank = interval(empty, fenced)
            gets = verify(step)
            if gets != expected_get:
                raise AssertionError(f"Queue probe changed final Get: {gets} != {expected_get}")
            rows.append(dict(fenced=fenced, **graph, empty=blank, final_get=gets))
    separated = all(
        r["begin_pending_after_submit"] != r["fenced"] and r["empty"]["begin_pending_after_submit"] != r["fenced"]
        for r in rows
    )
    return dict(
        protocol=PROTOCOL,
        status="measured" if separated else "queue_states_unresolved",
        scratch_bytes_per_tensor=SCRATCH_BYTES,
        leading_copy_bytes=SCRATCH_BYTES * COPIES,
        samples=rows,
        included_in_headline_timing=False,
    )
