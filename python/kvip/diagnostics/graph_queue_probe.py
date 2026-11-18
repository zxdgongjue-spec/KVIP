# SPDX-License-Identifier: Apache-2.0
"""Isolate graph submission sensitivity with identical leading work on the replay stream."""

import math

# 128 MiB of scratch. The copy runs before both event pairs and never touches graph state.
SCRATCH_BYTES = 64 << 20
COPIES = 16
FENCED = (True, False, False, True)


def probe(torch, stream, begin, end, replay, prepare, verify, first_step, expected_get):
    if not callable(getattr(begin, "query", None)):
        return dict(status="unavailable", reason="timing_event_query_missing", samples=[])

    rows = []
    with torch.npu.stream(stream):
        source = torch.zeros(SCRATCH_BYTES, dtype=torch.uint8, device=stream.device)
        target = torch.empty_like(source)
        target.copy_(source)
        stream.synchronize()
        for i, fence in enumerate(FENCED):
            step = first_step + i
            prepare(step)
            stream.synchronize()
            for _ in range(COPIES):
                target.copy_(source, non_blocking=True)
            if fence:
                stream.synchronize()
            begin.record(stream)
            replay()
            end.record(stream)
            # Query only after the whole replay and its end have been submitted.
            pending = not begin.query()
            end.synchronize()
            elapsed = float(begin.elapsed_time(end)) * 1000
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError(f"Invalid queue-probe duration: {elapsed}")
            gets = verify(step)
            if gets != expected_get:
                raise AssertionError(f"Queue probe changed final Get: {gets} != {expected_get}")
            rows.append(dict(fenced=fence, begin_pending_after_submit=pending, elapsed_us=elapsed, final_get=gets))
    evidence = all(row["begin_pending_after_submit"] for row in rows if not row["fenced"])
    return dict(
        status="measured" if evidence else "insufficient_queue_lead",
        scratch_bytes_per_tensor=SCRATCH_BYTES,
        leading_copy_bytes=SCRATCH_BYTES * COPIES,
        samples=rows,
        included_in_headline_timing=False,
    )
