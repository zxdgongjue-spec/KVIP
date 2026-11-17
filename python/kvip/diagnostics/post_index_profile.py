# SPDX-License-Identifier: Apache-2.0
"""Decode the low Post/Index/CQ tensor without reusing old streaming markers."""

import numpy as np


INDEX_DETAIL_MAGIC = 0x50495831
INDEX_DETAIL_SAME_VF_MAGIC = 0x50495832


def _distribution(values):
    values = list(map(int, values))
    return dict(min=min(values), p50=float(np.median(values)), max=max(values),
                mean=float(np.mean(values)), total=sum(values),
                active_cores=sum(value > 0 for value in values), samples=values)


def _event_counts(records, events):
    if events is None or len(events) != 2:
        raise ValueError("PIX2 requires post-replay source/destination event readback")
    source, destination = map(np.asarray, events)
    if (source.dtype != np.int32 or destination.dtype != np.int32
            or source.shape != destination.shape or source.ndim != 2 or source.shape[1] != 10240
            or not 1 <= len(source) <= len(records)):
        raise ValueError("Unexpected PIX2 event table layout")
    batch, cores = len(source), len(records)
    if not np.array_equal(records[:, 11], np.arange(cores) % batch):
        raise ValueError("PIX2 packed-row assignment differs from the producer ABI")
    counts = [sum(map(int, records[row::batch, 1])) for row in range(batch)]
    if any(count > 8192 for count in counts):
        raise ValueError("PIX2 request event count exceeds the final-event capacity")
    nonself = np.concatenate([source[row, 2048:2048+count] != destination[row, 2048:2048+count]
                              for row, count in enumerate(counts)])
    offsets = [len(nonself) * core // cores for core in range(cores + 1)]
    assigned = np.diff(offsets).astype(np.uint64)
    moved = np.asarray([np.count_nonzero(nonself[first:last]) for first, last in zip(offsets, offsets[1:])],
                       dtype=np.uint64)
    return assigned, moved


def decode_index_work(raw, records, *, fallback, wait_first=False, events=None):
    """Read versioned, diagnostic-only per-AIV work and barrier markers."""
    unavailable = dict(status="unavailable", reason="No PIX1 Index-work markers in this binary")
    if raw is None:
        return unavailable
    raw = np.asarray(raw)
    if raw.dtype != np.int64 or raw.shape != (len(records), 32):
        raise ValueError("Unexpected post-Index work diagnostic layout")
    rows = raw.view(np.uint64)
    present = np.isin(rows[:, 2], [INDEX_DETAIL_MAGIC, INDEX_DETAIL_SAME_VF_MAGIC])
    if not np.any(present):
        # Old binaries used these words for different role markers.
        return unavailable
    if not np.all(present) or not np.all(rows[:, 2] == rows[0, 2]):
        raise ValueError("Incomplete post-Index work marker coverage")
    ticks = rows[:, 3:8]
    ordered = (np.all(ticks[:, 1:4] >= ticks[:, :3])
               and np.all(ticks[:, 4] >= ticks[:, 0])
               and np.all(ticks[:, 4] <= ticks[:, 1] if wait_first
                          else ticks[:, 4] >= ticks[:, 3]))
    if (np.any(ticks == 0) or not ordered
            or np.any(records[:, 7] > ticks[:, 0])
            or np.any(ticks[:, 2] > records[:, 8])
            or np.any(records[:, 8] > ticks[:, 3])
            or ((not fallback or wait_first) and np.any(ticks[:, 4] != records[:, 9]))):
        raise ValueError("Invalid post-Index work/barrier timestamps")
    moved, assigned, byte_count = rows[:, 8], rows[:, 9], rows[:, 10]
    same_vf = bool(rows[0, 2] == INDEX_DETAIL_SAME_VF_MAGIC)
    if same_vf:
        expected_assigned, moved = _event_counts(records, events)
        if not np.array_equal(assigned, expected_assigned):
            raise ValueError("PIX2 assigned events differ from reconstructed global ranges")
        byte_count = moved * 592
    if (np.any(moved > assigned)
            or sum(map(int, assigned)) != sum(map(int, records[:, 1]))
            or int(assigned.max()) - int(assigned.min()) > 1
            or any(int(value) != int(count) * 592 for value, count in zip(byte_count, moved))):
        raise ValueError("Invalid post-Index assigned/moved tuple counts")
    intervals = {
        "post_barrier": ticks[:, 0] - records[:, 7],
        "index_setup": (np.zeros(len(rows), dtype=np.uint64) if wait_first
                        else ticks[:, 1] - ticks[:, 0]),
        "index_work": ticks[:, 2] - ticks[:, 1],
        "index_publish": records[:, 8] - ticks[:, 2],
        "index_barrier": ticks[:, 3] - records[:, 8],
        "cq_finalize" if fallback else "cq_wait": ticks[:, 4] - ticks[:, 0 if wait_first else 3],
    }
    if wait_first:
        intervals["cq_barrier"] = ticks[:, 1] - ticks[:, 4]
    return dict(
        status="available", abi="PIX2" if same_vf else "PIX1", core_count=len(rows),
        index_implementation="headline_same_VF" if same_vf else "instrumented_counting_VF",
        count_source="post_replay_CPU_event_readback" if same_vf else "device_diagnostic_counters",
        global_span_cycles=dict(index_work=int(ticks[:, 2].max()) - int(ticks[:, 1].min()),
                                post_barrier_to_completion=int(max(ticks[:, 3].max(), ticks[:, 4].max()))
                                - int(ticks[:, 0].min())),
        assigned_events=_distribution(assigned), moved_tuples=_distribution(moved),
        logical_index_rw_bytes=sum(map(int, byte_count)),
        per_core_cycles={name: {key: value for key, value in _distribution(values).items()
                               if key not in ("total", "active_cores")}
                         for name, values in intervals.items()},
        scope=("Separate diagnostic replay. Index work includes setup, VF launch, execution and its completion wait; "
               "barriers and CQ are separate. 592 bytes/non-self event is logical tuple read/write traffic, "
               "not measured HBM traffic: L2 caching and protocol overhead are excluded. Diagnostic counting "
               "is only in PIX1; PIX2 uses the headline VF and counts events after replay. Per-core cycles overlap and must not be summed "
               "or converted to bandwidth."),
    )


def decode_post_index(raw, *, active, diagnostic=False, index_detail=None, events=None):
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
    if np.any(rows[:, 2] > 3):
        raise ValueError("Invalid post-Index fallback marker")
    if not np.all(rows[:, 2] == rows[0, 2]):
        raise ValueError("Post-Index cores disagree about bounded-Get fallback")
    flags = int(rows[0, 2])
    fallback, wait_first = bool(flags & 1), bool(flags & 2)
    result = dict(status="pass", active=True, fallback=fallback,
                  wait_first=wait_first,
                  fallback_cores=len(rows) if fallback else 0,
                  submitted_requests=sum(map(int, rows[:, 1])),
                  completed_requests=sum(map(int, rows[:, 15])),
                  core_count=len(rows), rows=rows.tolist())
    stamps = rows[:, [10, 4, 5, 6, 7, 8, 9]]
    if not diagnostic:
        if np.any(stamps):
            raise ValueError("Post-Index diagnostic timestamps were enabled in headline timing")
        return result
    ordered = (np.all(stamps[:, 1:6] >= stamps[:, :5])
               and np.all(stamps[:, 6] >= stamps[:, 3] if fallback else stamps[:, 6] >= stamps[:, 4])
               and (fallback or np.all(stamps[:, 6] <= stamps[:, 5] if wait_first
                                      else stamps[:, 6] >= stamps[:, 5])))
    if np.any(stamps == 0) or not ordered:
        raise ValueError("Incomplete or unordered post-Index phase markers")
    start = int(stamps[:, 0].min())
    names = ("plan", "pack", "post_begin", "post", "index", "cq")
    result["phase_status"] = "available"
    result["phase_policy"] = ("bounded_sync_get_then_index" if fallback else
                              "post_then_cq_then_index" if wait_first else "post_then_index_then_cq")
    result["phase_end_from_first_begin_cycles"] = {
        name: int(stamps[:, column].max()) - start for column, name in enumerate(names, 1)
    }
    intervals = {"plan": (0, 1), "pack": (1, 2), "post": (3, 4), "index": (4, 5), "cq": (5, 6)}
    if fallback:
        # Sync Get drains before Index. It has no independent post-Index CQ
        # interval even when the kernel stamps its final completion afterwards.
        del intervals["cq"]
    elif wait_first:
        intervals["cq"] = (4, 6)
        intervals["index"] = (6, 5)
    result["per_core_phase_cycles"] = {
        name: dict(median_cycles=float(np.median(stamps[:, end] - stamps[:, begin])),
                   max_cycles=int((stamps[:, end] - stamps[:, begin]).max()))
        for name, (begin, end) in intervals.items()
    }
    if fallback:
        result["cq_scope"] = "Included in bounded synchronous Get; no independent post-Index CQ wait"
    result["index_work"] = decode_index_work(index_detail, rows, fallback=fallback, wait_first=wait_first, events=events)
    result["scope"] = "Separate instrumented replay; phase boundaries include synchronization. Cross-core/overlapping intervals must not be summed or substituted for graph latency."
    return result
