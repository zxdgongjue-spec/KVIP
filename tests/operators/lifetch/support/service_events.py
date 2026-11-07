# SPDX-License-Identifier: Apache-2.0
"""Decode the per-epoch service events written by V9_INTERMEDIATE_PROFILING.

Each service lane stamps one 128-byte Event per epoch: how long it waited for
the previous epoch, and how its in-LI interruption split into bitmap scan,
candidate selection and WQE post. This is the only existing source of the
prefetch bubble breakdown; no extra kernel build flag is needed.
"""

import numpy as np

EVENT = np.dtype(
    [
        ("begin", "<u8"),
        ("end", "<u8"),
        ("core", "<u4"),
        ("epoch", "<u4"),
        ("complete", "<u4"),
        ("skipped", "<u4"),
        ("waitBegin", "<u8"),
        ("scanTicks", "<u8"),
        ("candidateTicks", "<u8"),
        ("postTicks", "<u8"),
        ("bitmapBytes", "<u8"),
        ("candidateCount", "<u8"),
        ("postedCount", "<u8"),
        ("pad", "<u4", 10),
    ]
)


def rows(rotating):
    raw = rotating.events.cpu().numpy().view(EVENT).reshape(-1)
    out = []
    for index, event in enumerate(raw):
        if int(event["end"]) < int(event["begin"]):
            continue
        out.append(
            dict(
                index=index,
                core=int(event["core"]),
                epoch=int(event["epoch"]),
                complete=int(event["complete"]),
                wait=int(event["begin"]) - int(event["waitBegin"]),
                span=int(event["end"]) - int(event["begin"]),
                scan=int(event["scanTicks"]),
                candidate=int(event["candidateTicks"]),
                post=int(event["postTicks"]),
                bitmap_bytes=int(event["bitmapBytes"]),
                candidates=int(event["candidateCount"]),
                posted=int(event["postedCount"]),
            )
        )
    return out


def table(rotating):
    """Per-epoch totals plus the worst lane, in device ticks."""
    events = rows(rotating)
    if not events:
        return "service events: none recorded (V9_INTERMEDIATE_PROFILING off or graph not replayed)"
    complete = sum(event["complete"] for event in events)
    lines = [
        f"service events: {len(events)} stamped, {complete} complete",
        "epoch\tlanes\twait_max\tspan_max\tscan\tscan_max\tcandidate\tpost\tbitmap_B\tcandidates\tposted",
    ]
    for epoch in sorted({event["epoch"] for event in events}):
        group = [event for event in events if event["epoch"] == epoch]
        lines.append(
            f"{epoch}\t{len(group)}"
            f"\t{max(e['wait'] for e in group)}"
            f"\t{max(e['span'] for e in group)}"
            f"\t{sum(e['scan'] for e in group)}"
            f"\t{max(e['scan'] for e in group)}"
            f"\t{sum(e['candidate'] for e in group)}"
            f"\t{sum(e['post'] for e in group)}"
            f"\t{sum(e['bitmap_bytes'] for e in group)}"
            f"\t{sum(e['candidates'] for e in group)}"
            f"\t{sum(e['posted'] for e in group)}"
        )
    span = sum(event["span"] for event in events) / max(1, len({e['core'] for e in events}))
    lines.append(f"mean lane span per epoch: {span:.0f} ticks; ticks are device cycles, not microseconds")
    return "\n".join(lines)
