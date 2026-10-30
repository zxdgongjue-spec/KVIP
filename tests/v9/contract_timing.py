"""Decode device tick counters after completion, outside the timed interval."""

import numpy as np

STAGES = (
    "init_state",
    "init_queue_directory",
    "init_index_cache",
    "resolve_index_cache",
    "fetch_completion",
    "fetch_publication",
    "fetch_cq_wait_or_deferred_get",
    "init_clear_and_control",
    "resolve_topk_plan_worker0",
    "init_kernel_body",
    "init_setup",
    "resolve_kernel_body_worker0",
    "fetch_kernel_body",
)
PATHS = {0: "rebuild", 1: "incremental", 2: "stable", 3: "exchange", 4: "plan_delta", 5: "planner_records"}


def completion_diagnostics(status, rows, posts, row_status, error_detail, plan_count, sticky):
    """Small JSON evidence even when verification fails before steps.append()."""
    return {
        "sticky": int(np.asarray(sticky).reshape(-1)[0]),
        "requests": [
            {
                "request": b,
                "status": int(value),
                "error_detail": np.asarray(error_detail[b]).tolist(),
                "planned_misses": int(plan_count[b]),
                "prefetch_requests": int(rows[b, 9]),
                "prefetch_pending": int(rows[b, 211]),
                "prefetch_error": int(rows[b, 11]),
                "final_summary": np.asarray(row_status[b, :5]).tolist(),
                "workers": [
                    dict(zip(("count", "pending", "error", "deferred", "descriptor", "submitted"), map(int, p[:6])))
                    for p in posts[b]
                ],
            }
            for b, value in enumerate(status)
        ],
    }


def decode_device_stages(row_status):
    rows = np.ascontiguousarray(row_status, dtype=np.uint32)
    if rows.ndim != 2 or rows.shape[1] != 64:
        raise ValueError("V9 status rows must have 64 words (ABI 13) per request")
    ticks = rows.view(np.uint64)[:, [4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19]]
    return {
        "ticks": {name: ticks[:, i].tolist() for i, name in enumerate(STAGES)},
        "index_work": [
            {
                "init_copy_rows": int(row[24]),
                "init_path": PATHS.get(int(row[25]), "unknown"),
                "resolve_copy_rows": int(row[26]),
                "resolve_path": PATHS.get(int(row[27]), "unknown"),
                "dedicated_index_aiv": bool(row[28]),
                "init_h0_reused": bool(row[29]),
                "init_mapping_validation": "epoch" if row[29] == 2 else "exact",
            }
            for row in rows
        ],
    }


def summarize_device_stages(samples):
    if not samples:
        return None
    return {
        "unit": "raw GetSystemCycle ticks; no assumed frequency",
        "aggregation": "per-step maximum across requests, then percentiles across measured steps",
        "overlap": "resolve cache work overlaps other final workers and outstanding KV transfers",
        "kernel_body_scope": (
            "entry instrumentation through TPipe Destroy; includes nested probes, excludes final counter store "
            "and dispatch; resolve is worker 0, not all-worker maximum"
        ),
        "cq_scope": "CQ drain calls plus blocking deferred GET; subset of fetch_completion",
        "stages": {
            name: {
                "p50_ticks": float(np.median(values)),
                "p95_ticks": float(np.percentile(values, 95)),
                "samples": len(values),
            }
            for name in samples[0]["ticks"]
            for values in [[max(sample["ticks"][name]) for sample in samples]]
        },
    }


def decode_service_stages(events, rounds):
    """Read existing V9 window counters after completion; no new device fences."""
    if not rounds:
        return {}
    words = np.ascontiguousarray(events[:rounds], dtype=np.uint8).view(np.uint64).reshape(rounds, -1, 16)
    values = {
        name: 0
        for name in (
            "service",
            "handoff_and_service",
            "scan",
            "candidate",
            "post",
            "bitmap_bytes",
            "candidates",
            "posted",
        )
    }
    for window in words:
        active = [w for w in window if int(w[1]) > 0 and int(w[1]) >= int(w[0])]
        if not active:
            continue
        values["service"] += max(int(w[1]) - int(w[0]) for w in active)
        values["handoff_and_service"] += max(int(w[1]) - int(w[4]) for w in active)
        for name, column in (("scan", 5), ("candidate", 6), ("post", 7)):
            values[name] += max(int(w[column]) for w in active)
        for name, column in (("bitmap_bytes", 8), ("candidates", 9), ("posted", 10)):
            values[name] += sum(int(w[column]) for w in active)
    return values


def summarize_service_stages(samples):
    samples = [s for s in samples if s]
    if not samples:
        return None
    return {
        "unit": "raw ticks for durations; bytes/counts for work",
        "scope": "inside LI; each step sums per-window lane maxima for durations, lane sums for work; "
        "not additive to LI and not a scan/TopK critical-path decomposition",
        "stages": {
            name: {
                "p50": float(np.median([s[name] for s in samples])),
                "p95": float(np.percentile([s[name] for s in samples], 95)),
                "samples": len(samples),
            }
            for name in samples[0]
        },
    }


def decode_kernel_timeline(raw, batch, li_cores, final_workers):
    """Each duration uses one core's clock; never subtract timestamps across cores."""
    raw = np.asarray(raw, dtype=np.uint64)
    if raw.shape != (320, 16):
        raise ValueError("ABI 13 kernel timeline requires 320 isolated records")
    groups = {
        "init_all_lanes": list(range(batch)),
        "li_aiv_all_lanes": list(range(32, 32 + li_cores * 2)),
        "li_aic_all_lanes": list(range(256, 256 + li_cores)),
        "resolve_all_workers": [96 + b * 4 + w for b in range(batch) for w in range(final_workers)],
        "fetch_all_lanes": list(range(224, 224 + batch)),
    }
    ticks, spans = {}, {}
    for name, indices in groups.items():
        entries = []
        for i in indices:
            start, end = map(int, raw[i, :2])
            if not start or end < start:
                raise ValueError(f"missing/invalid kernel timeline record {i}")
            entries.append({"slot": i, "begin": start, "end": end, "ticks": end - start})
        ticks[name] = [e["ticks"] for e in entries]
        spans[name] = entries
    return ticks, spans
