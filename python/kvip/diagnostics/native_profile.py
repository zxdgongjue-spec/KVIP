# SPDX-License-Identifier: Apache-2.0
"""Decode production streaming markers from separate instrumented replays.

All durations are raw device cycles within one role/query. They overlap across
roles and requests; neither their sum nor their maximum is a graph latency.
"""

import numpy as np


def decode_profile(raw, *, batch, workers, resolver_cores, transport, threshold_markers=False,
                   early_prefetch_drain=False, init_route_markers=False, row_route_publish=False,
                   direct_index_markers=False, compact_plan_markers=False):
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
    init_intervals = {
        "init_total": (25, 31),
        "init_route": (26, 27),
        "init_fill": (28, 29),
        "init_tail_setup": (29, 30),
        "init_finish": (30, 31),
    }
    for name, (begin, end) in init_intervals.items():
        if init_route_markers:
            interval(name, consumer, begin, end, required=True)
        else:
            # These slots have no contract in older binaries, even when nonzero.
            metrics[name] = dict(status="unavailable", count=0, median_cycles=None,
                                 max_cycles=None, samples_cycles=[])
    for name, rows in (("init_route_publish", consumer[:1]), ("init_route_probe", consumer[1:])):
        if init_route_markers:
            interval(name, rows, 26, 27, required=True)
        else:
            metrics[name] = dict(status="unavailable", count=0, median_cycles=None,
                                 max_cycles=None, samples_cycles=[])
    direct_index_counts = None
    if direct_index_markers:
        eligible, events = consumer[:, 2], consumer[:, 3]
        if (transport != "urma" or np.any(eligible > 1) or np.any(events > 8192)
                or np.any((eligible == 0) & (events != 0))):
            raise ValueError("Invalid direct Index update counters")
        rows = int(np.count_nonzero(eligible))
        direct_index_counts = dict(
            direct_rows=rows, fallback_rows=batch - rows,
            direct_events=sum(int(value) for value in events),
            rows=[dict(row=b, direct=bool(eligible[b]), events=int(events[b]))
                  for b in range(batch)],
        )
    compact_plan_counts = None
    if compact_plan_markers:
        active = producer[:, 9]
        if transport != "urma" or np.any(active > 1):
            raise ValueError("Invalid compact low plan counters")
        rows = int(np.count_nonzero(active))
        compact_plan_counts = dict(
            active_rows=rows, fallback_rows=batch - rows, rows=[bool(value) for value in active]
        )
    # The dedicated role markers bracket the entire Consumer, including an
    # optional early CQ drain before InitShadow. Keep the older work interval
    # separate so moving work across its start cannot look like saved time.
    index_roles = cores[np.arange(batch) * resolver_cores + 1] if resolver_cores > 1 else cores[:0]
    interval("index_role_span", index_roles, 0, 1)
    publication_roles = cores[np.arange(batch) * resolver_cores + 2] if resolver_cores > 2 else cores[:0]
    publication_queries = publication_roles[:, 4:28].reshape(-1, 6)
    # OutputWorker ticks bracket descriptor packing after raw-query readiness.
    # This includes the location DMA; it is distinct from the Get worker's
    # later ready-batch packing, which is reported as fetch_pack below.
    interval("publication_pack", publication_queries, 2, 3)
    if early_prefetch_drain:
        if transport != "urma":
            raise ValueError("Early prefetch drain markers require URMA")
        interval("early_prefetch_drain", consumer, 0, 1, required=True)
    interval("producer_span", producer, 1, 8)
    interval("index_consumer_span", consumer, 10, 24)
    interval("index_metadata", consumer, 22, 23)
    # These markers predate the direct-Index capability, which gives us a
    # conservative version boundary. Do not interpret older unversioned slots.
    detail = {
        "producer_setup": (producer, 1, 2),
        "producer_initialize": (producer, 2, 3),
        "producer_first_query_ready": (producer, 1, 4),
        "producer_after_last_query": (producer, 7, 8),
        "index_initial_wait_and_setup": (consumer, 10, 12),
    }
    for name, (rows, begin, end) in detail.items():
        if direct_index_markers:
            interval(name, rows, begin, end)
        else:
            metrics[name] = dict(status="unavailable", count=0, median_cycles=None,
                                 max_cycles=None, samples_cycles=[])
    producer_queries = producer[:, 10:30].reshape(batch * 4, 5)
    if direct_index_markers:
        # Compact Q4 computes outside these per-query intervals; exclude its
        # rows rather than presenting publication-only time as fast planning.
        plan_rows = producer_queries
        if compact_plan_markers:
            plan_rows = producer[:, 10:30][producer[:, 9] == 0].reshape(-1, 5)
        interval("producer_query_plan", plan_rows, 0, 1)
        interval("producer_query_location_dma", producer_queries, 3, 4)
    else:
        for name in ("producer_query_plan", "producer_query_location_dma"):
            metrics[name] = dict(status="unavailable", count=0, median_cycles=None,
                                 max_cycles=None, samples_cycles=[])

    index_apply, index_control = [], []
    if direct_index_markers:
        for row in consumer:
            # A successful empty phase stamps an equal start/end. Metadata
            # may be absent when the request has no changes at all.
            pairs = [(int(row[12 + 2 * p]), int(row[13 + 2 * p])) for p in range(5)]
            if not any(start or stop for start, stop in pairs):
                continue
            if any(start == 0 or stop < start for start, stop in pairs):
                raise ValueError("Invalid production diagnostic timestamps: index apply phases")
            begin, end = int(row[10]), int(row[24])
            meta_begin, meta_end = int(row[22]), int(row[23])
            if (begin == 0 or end < begin or
                    (meta_begin == 0 and meta_end != 0) or meta_end < meta_begin):
                raise ValueError("Invalid production diagnostic timestamps: index control remainder")
            apply = sum(stop - start for start, stop in pairs)
            control = end - begin - apply - (meta_end - meta_begin)
            if control < 0:
                raise ValueError("Index work exceeds its enclosing role interval")
            index_apply.append(apply)
            index_control.append(control)
    for name, values in (("index_apply_total", index_apply),
                         ("index_wait_and_control", index_control)):
        metrics[name] = dict(
            status="available" if values else "unavailable", count=len(values),
            median_cycles=float(np.median(values)) if values else None,
            max_cycles=max(values) if values else None, samples_cycles=values,
        )
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
        dependency = metrics["fetch_dependency_wait"]
        if dependency["status"] == "unavailable" and np.all(queries[:, 5:7] == 0):
            # The ready-worker profile tag establishes that these zero fields
            # are initialized diagnostics. Prefetch() writes fields 5/6 only
            # when it encounters a conflicting destination range.
            dependency["status"] = "not_entered"
            dependency["reason"] = "No ready-worker query entered the prefetch-conflict wait"
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
        direct_index_counts=direct_index_counts,
        compact_plan_counts=compact_plan_counts,
        ready_batch_counts=(dict(
            native_calls=sum(int(row[23]) for row in get_queries),
            completed_requests=sum(int(row[22]) for row in get_queries),
            worker_count=len(get_queries),
        ) if ready_batches else None),
        ready_batch_by_query=([
            dict(
                query=q,
                requests=sum(int(row[22]) for row in shaped[: min(batch, workers), q]),
                calls=sum(int(row[23]) for row in shaped[: min(batch, workers), q]),
                native_max_cycles=max(int(row[21]) for row in shaped[: min(batch, workers), q]),
                active_workers=sum(int(row[22]) > 0 for row in shaped[: min(batch, workers), q]),
                worker_count=min(batch, workers),
            )
            for q in range(4)
        ] if ready_batches else None),
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
            initialization=(
                "Per-request init block cycles: total includes route decision, InitializeRound fills, "
                "ABC/transport setup and final publication. Blocks overlap; route includes input "
                "cache maintenance. Stable field names route_publish/route_probe denote block 0/blocks 1 onward; "
                + ("row publication is enabled, so both intervals may include publishing one pending row. "
                   "Control writes use cache-line owners; unaligned inputs use the legacy fallback. "
                   if row_route_publish else "Only block 0 publishes pending/control state. ")
                + "No hardware clock conversion or sum across blocks."
            ),
            early_threshold="Per-query causal preparation, recent-score load, Kth/full TopK and SaveResident bitmap/progress; excludes preceding recent/resident QK matmul",
            final_topk="Final TopK selection before final LI classification",
            get_counters=get_scope
            + "; post_active/CQ drain sum safe and conflict phases only within each worker, not across workers",
            scan="No standalone full-history scan or QK duration is derived from these markers",
            aggregation="Median/max across request/query role samples; overlapping intervals are not additive",
            producer_detail=(
                "Capability-gated existing markers: setup includes validation, input/context copies and bitmap merge; "
                "initialize covers StagedInitializeReady. Query plan includes the VF and its fence; compact Q4 rows "
                "are excluded because their planning precedes these markers. First-query-ready ends at the raw "
                "Q0 publication timestamp, before its scalar cache-line release. After-last-query includes that "
                "release, OutputWorker completion wait and final bookkeeping; it is not pure planner work."
            ),
            compact_plan=(
                "Capability-gated Producer slot9 records actual compact-plan execution, not duration. "
                "Producer span includes the complete compact plan. Per-query Producer markers may "
                "cover publication only; never compare them alone as equivalent planning cost."
            ),
            index=(
                "index_role_span brackets Consumer role entry/exit, including early prefetch drain; "
                "index_consumer_span brackets the common work start through metadata commit and excludes early drain "
                "and final status publication. Direct Index counters, when capability-gated, count actual "
                "direct-path rows/events and are not timing intervals. index_apply_total sums only the five "
                "nonoverlapping apply intervals within one Consumer. index_wait_and_control subtracts those and "
                "metadata from that same Consumer's span; it includes waiting, shadow setup and scalar control, "
                "not only CQ wait. Medians of components must not be added to reconstruct the median total."
            ),
            prefetch_completion=(
                "early_prefetch_drain is the Consumer-owned CQ drain; final_prefetch_join is the later resolver acquire"
                if early_prefetch_drain else "final_prefetch_join includes the resolver-owned CQ drain"
            ),
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
            if metric and metric["status"] == "not_entered":
                fields.append(f"{key}=not_entered")
            elif not metric or metric["status"] == "unavailable":
                fields.append(f"{key}=unavailable")
            else:
                fields.append(f"{key}={metric['median_cycles']:.0f}/{metric['max_cycles']}")
        print(f"PROFILE {name} replay={index + 1} raw_cycles_median/max " + " ".join(fields), flush=True)
    print(
        "PROFILE_SCOPE separate instrumented replays; not headline timings; overlapping roles, no cycle-to-us conversion.",
        flush=True,
    )
