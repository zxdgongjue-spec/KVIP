# SPDX-License-Identifier: Apache-2.0
"""Device attribution for the fixed-Q4 serial/fused, single-graph comparison."""

from collections import defaultdict
from functools import partial
from math import isfinite
from statistics import mean, median

from large_batch_pair import fixed_invocation_intervals, fixed_kernel_stages, fixed_stage_records
from performance import profile_graph_batch, split_graph_invocations


def _terminal_finishes(records):
    # One stream launch can have overlapping AIC and AIV hardware rows. The
    # invocation interval validator below verifies their task identity and roles.
    finishes = []
    for start, stop, *_ in sorted(records, key=lambda record: record[:2]):
        if finishes and start < finishes[-1]:
            finishes[-1] = max(finishes[-1], stop)
        else:
            finishes.append(stop)
    return finishes


def _profile_result(events, scope, elapsed_us, *, fused, batch, calls):
    if calls < 1 or not isfinite(elapsed_us) or elapsed_us <= 0:
        raise ValueError("Expected a positive graph event time and invocation count")
    records = fixed_stage_records(events, scope, fused=fused, batch=batch, strict_bounds=True)
    finishes = _terminal_finishes(records["tail" if fused else "fetch"])
    if len(finishes) != calls:
        raise ValueError(f"{scope}: expected {calls} terminal launches, found {len(finishes)}")

    samples = []
    # Normalized stage names used by the standard replay JSON writer.
    groups = defaultdict(list)
    previous = float("-inf")
    for i, end in enumerate(finishes):
        partition = {k: [r for r in rows if previous <= r[0] < end] for k, rows in records.items()}
        intervals = fixed_invocation_intervals(partition, f"{scope}/call{i}")
        sample = fixed_kernel_stages(intervals, fused)
        for key, runs in intervals.items():
            stage = ("resolve_update" if key == "tail" else "li") if fused else key
            groups[stage].extend(runs)
        samples.append(sample)
        previous = end
    if any(r[0] >= previous for rows in records.values() for r in rows):
        raise ValueError(f"{scope}: kernels follow the last terminal launch")

    first = min(a for a, _ in groups["li"])
    total_span = finishes[-1] - first
    accounting = dict(intervals_us=dict(groups), kernel_span_us=total_span, event_us=elapsed_us)
    profile = split_graph_invocations(accounting, calls, serial=not fused)
    averages = {k: mean(sample[k] for sample in samples) for k in samples[0]}
    averages.update(
        graph=elapsed_us / calls,
        event_minus_span=profile["outside_invocations_us"] / calls,
        between_calls=(total_span - sum(s["span"] for s in samples)) / calls,
        event_edges=(elapsed_us - total_span) / calls,
    )
    profile["packed"] = dict(
        calls=calls, graph_event_us=elapsed_us, graph_mean_us=elapsed_us / calls,
        mean_us=averages, p50_us={k: median(s[k] for s in samples) for k in samples[0]}, samples=samples,
    )
    return profile


def packed_account(events, scope, elapsed_us, *, fused, batch, calls=10):
    """Graph event/calls, device-stage means and percentiles from these same calls.

    For each arm the graph mean comprises LI, tail, gaps between kernels in
    one call, gaps between calls, and graph edges. Only the graph as a whole
    has device timing events. Each invocation has its own observed kernel span.
    """
    return _profile_result(events, scope, elapsed_us, fused=fused, batch=batch, calls=calls)["packed"]


def profile_fixed_graph_batch(torch, graph, *, batch, calls, folder, serial=False, transport="urma", stream=None):
    parser = partial(_profile_result, fused=not serial, batch=batch, calls=calls)
    return profile_graph_batch(torch, graph, calls=calls, folder=folder, serial=serial,
                               transport=transport, stream=stream, result_parser=parser)
