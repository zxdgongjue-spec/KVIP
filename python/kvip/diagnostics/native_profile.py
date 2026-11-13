# SPDX-License-Identifier: Apache-2.0
"""Decode production streaming markers from separate instrumented replays.

All durations are raw device cycles within one role/query. They overlap across
roles and requests; neither their sum nor their maximum is a graph latency.
"""

import numpy as np


def decode_profile(raw, *, batch, workers, resolver_cores, transport, threshold_markers=False):
    raw = np.asarray(raw)
    expected = batch * 256 + (resolver_cores * batch + 4 * workers) * 32
    if raw.dtype != np.int64 or raw.size != expected or batch < 1 or workers < 1:
        raise ValueError("Unexpected production streaming timestamp layout")
    raw = raw.reshape(-1).view(np.uint64)
    producer_update = raw[: batch * 64].reshape(batch, 64)
    li = raw[batch * 64 : batch * 128].reshape(batch, 4, 16)
    fetch = raw[batch * 128 : batch * 256].reshape(batch, 128)
    cores = raw[batch * 256 :].reshape(-1, 32)
    metrics = {}

    def interval(name, rows, begin, end, *, required=False):
        pairs = [(int(row[begin]), int(row[end])) for row in rows]
        present = [(start, stop) for start, stop in pairs if start or stop]
        if any(start == 0 or stop < start for start, stop in present):
            raise ValueError(f"Invalid production diagnostic timestamps: {name}")
        if required and len(present) != len(pairs):
            raise ValueError(f"Missing production diagnostic timestamps: {name}")
        values = [stop - start for start, stop in present]
        metrics[name] = dict(
            status="available" if values else "unavailable",
            count=len(values),
            median_cycles=float(np.median(values)) if values else None,
            max_cycles=max(values) if values else None,
            samples_cycles=values,
        )

    li_queries = li.reshape(-1, 16)
    if threshold_markers:
        interval("early_threshold", li_queries, 12, 13, required=True)
        interval("final_topk", li_queries, 14, 15, required=True)
    else:
        # Old main has no contract for these slots; never interpret stale bits.
        for name in ("early_threshold", "final_topk"):
            metrics[name] = dict(status="unavailable", count=0, median_cycles=None, max_cycles=None, samples_cycles=[])
    interval("li_final_mapping_wait", li_queries, 0, 1)
    interval("li_final_classification", li_queries, 2, 7)
    producer = producer_update[:, :32]
    consumer = producer_update[:, 32:]
    interval("producer_span", producer, 1, 8)
    interval("index_consumer_span", consumer, 10, 24)
    interval("index_metadata", consumer, 22, 23)
    queries = fetch[:, :96].reshape(-1, 24)
    interval("fetch_query_ready_wait", queries, 0, 1)
    interval("fetch_prepare", queries, 1, 2)
    interval("fetch_dependency_wait", queries, 5, 6)
    get_queries = queries
    get_scope = "Per-request query Get intervals"
    ready_batches = transport == "urma" and np.any(queries[:, 20] == 4)
    if transport == "urma" and (np.any(queries[:, 20] == 3) or ready_batches):
        mode = 4 if ready_batches else 3
        if np.any(queries[:, 20] != mode):
            raise ValueError("Mixed grouped/per-request URMA diagnostic modes")
        shaped = queries.reshape(batch, 4, 24)
        fields = [3, 4, 7, 8, 12, 13, 14, 15, 16, 17]
        if ready_batches:
            fields += [18, 19, 21, 22, 23]
        for b in range(batch):
            if not np.array_equal(shaped[b][:, fields], shaped[b % workers][:, fields]):
                raise ValueError("Replicated shared Q-worker timestamps disagree")
        get_queries = shaped[: min(batch, workers)].reshape(-1, 24)
        get_scope = "One representative per (query, worker lane), b=lane; repeated group timestamps across requests are not counted twice"
        for name, positions in (("post_active", (12, 14)), ("cq_drain", (13, 15))):
            values = [sum(int(row[pos]) for pos in positions) for row in get_queries]
            metrics[name] = dict(
                status="available",
                count=len(values),
                median_cycles=float(np.median(values)),
                max_cycles=max(values),
                samples_cycles=values,
            )
    if ready_batches:
        # The first-start/last-end envelope contains inter-call polling/packing.
        # Keep it distinct from the sum of actual native Get call durations.
        interval("fetch_batch_envelope", get_queries, 3, 4)
        for name, field in (("fetch_native_get", 21), ("fetch_pack", 19), ("fetch_ready_prepare", 18)):
            values = [int(row[field]) for row in get_queries]
            metrics[name] = dict(status="available", count=len(values), median_cycles=float(np.median(values)),
                                 max_cycles=max(values), samples_cycles=values)
        for name in ("fetch_safe_submit_or_sync", "fetch_conflict_get"):
            metrics[name] = dict(status="unavailable", count=0, median_cycles=None, max_cycles=None, samples_cycles=[])
        get_scope += "; ready-batch native time is the sum of calls, envelope includes between-call work; counts are deduplicated by worker"
    else:
        interval("fetch_safe_submit_or_sync", get_queries, 3, 4)
        interval("fetch_conflict_get", get_queries, 7, 8)
    if transport == "urma":
        interval("final_prefetch_join", fetch, 96, 100)
        interval("final_index_and_query_join", fetch, 100, 97)
        interval("final_residency", fetch, 97, 98)
        interval("final_status", fetch, 98, 99)
    return dict(
        metrics=metrics,
        ready_batch_counts=(dict(
            native_calls=sum(int(row[23]) for row in get_queries),
            completed_requests=sum(int(row[22]) for row in get_queries),
            worker_count=len(get_queries),
        ) if ready_batches else None),
        raw=dict(
            batch=batch,
            workers_per_query=workers,
            resolver_cores=resolver_cores,
            producer_update=producer_update.tolist(),
            li=li.tolist(),
            fetch=fetch.tolist(),
            cores=cores.tolist(),
        ),
        scope=dict(
            early_threshold="Per-query causal preparation, recent-score load, Kth/full TopK and SaveResident bitmap/progress; excludes preceding recent/resident QK matmul",
            final_topk="Final TopK selection before final LI classification",
            get_counters=get_scope
            + "; post_active/CQ drain sum safe and conflict phases only within each worker, not across workers",
            scan="No standalone full-history scan or QK duration is derived from these markers",
            aggregation="Median/max across request/query role samples; overlapping intervals are not additive",
        ),
    )


def print_profile(name, report):
    """One compact line per independently instrumented replay; raw detail is saved."""
    names = (
        "early_threshold",
        "final_topk",
        "li_final_mapping_wait",
        "fetch_dependency_wait",
        "final_prefetch_join",
        "post_active",
        "cq_drain",
    )
    for index, sample in enumerate(report["samples"]):
        fields = []
        for key in names:
            metric = sample["metrics"].get(key)
            if not metric or metric["status"] == "unavailable":
                fields.append(f"{key}=unavailable")
            else:
                fields.append(f"{key}={metric['median_cycles']:.0f}/{metric['max_cycles']}")
        print(f"PROFILE {name} replay={index + 1} raw_cycles_median/max " + " ".join(fields), flush=True)
    print(
        "PROFILE_SCOPE separate instrumented replays; not headline timings; overlapping roles, no cycle-to-us conversion.",
        flush=True,
    )
