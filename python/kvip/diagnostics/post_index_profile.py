# SPDX-License-Identifier: Apache-2.0
"""Decode the low Post/Index/CQ tensor without reusing old streaming markers."""

import numpy as np


def decode_post_index(raw, *, active, diagnostic=False):
    if not active:
        # High uses the original tail. Its unused low workspace may be stale.
        return dict(status="not_entered", active=False)
    raw = np.asarray(raw)
    if raw.dtype != np.int64 or raw.ndim != 2 or raw.shape[1] != 16 or not 1 <= len(raw) <= 64:
        raise ValueError("Unexpected post-Index diagnostic ABI")
    rows = raw.view(np.uint64)
    if np.any(rows[:, 0]) or np.any(rows[:, 3]):
        raise ValueError("Post-Index device status reports an error")
    if not np.array_equal(rows[:, 1], rows[:, 15]):
        raise ValueError("Post-Index completed count differs from planned Get count")
    if np.any(rows[:, 2] > 1):
        raise ValueError("Invalid post-Index fallback marker")
    if not np.all(rows[:, 2] == rows[0, 2]):
        raise ValueError("Post-Index cores disagree about bounded-Get fallback")
    fallback = bool(np.any(rows[:, 2]))
    result = dict(status="pass", active=True, fallback=fallback,
                  fallback_cores=int(np.count_nonzero(rows[:, 2])),
                  submitted_requests=sum(map(int, rows[:, 1])),
                  completed_requests=sum(map(int, rows[:, 15])),
                  core_count=len(rows), rows=rows.tolist())
    stamps = rows[:, [10, 4, 5, 6, 7, 8, 9]]
    if not diagnostic:
        if np.any(stamps):
            raise ValueError("Post-Index diagnostic timestamps were enabled in headline timing")
        return result
    ordered = (np.all(stamps[:, 1:6] >= stamps[:, :5])
               and np.all(stamps[:, 6] >= stamps[:, 3] if fallback else stamps[:, 6] >= stamps[:, 5]))
    if np.any(stamps == 0) or not ordered:
        raise ValueError("Incomplete or unordered post-Index phase markers")
    start = int(stamps[:, 0].min())
    names = ("plan", "pack", "post_begin", "post", "index", "cq")
    result["phase_status"] = "available"
    result["phase_policy"] = "bounded_sync_get_then_index" if fallback else "post_then_index_then_cq"
    result["phase_end_from_first_begin_cycles"] = {
        name: int(stamps[:, column].max()) - start for column, name in enumerate(names, 1)
    }
    intervals = {"plan": (0, 1), "pack": (1, 2), "post": (3, 4), "index": (4, 5), "cq": (5, 6)}
    if fallback:
        # Sync Get drains before Index. It has no independent post-Index CQ
        # interval even when the kernel stamps its final completion afterwards.
        del intervals["cq"]
    result["per_core_phase_cycles"] = {
        name: dict(median_cycles=float(np.median(stamps[:, end] - stamps[:, begin])),
                   max_cycles=int((stamps[:, end] - stamps[:, begin]).max()))
        for name, (begin, end) in intervals.items()
    }
    if fallback:
        result["cq_scope"] = "Included in bounded synchronous Get; no independent post-Index CQ wait"
    result["scope"] = "Separate instrumented replay; phase boundaries include synchronization. Cross-core/overlapping intervals must not be summed or substituted for graph latency."
    return result
