# SPDX-License-Identifier: Apache-2.0
"""CPU-only fixture audit; reconcile measured counters without emulating NPU timing."""

import numpy as np


def selected_bounds(selected, history):
    """Summarize invalid LI outputs; UINT32_MAX is the padding sentinel."""
    values = np.asarray(selected).astype(np.uint32, copy=False)
    invalid = (values >= history) & (values != np.uint32(0xFFFFFFFF))
    examples = []
    for position in np.argwhere(invalid)[:8]:
        index = tuple(int(x) for x in position)
        examples.append(dict(index=index, value=int(values[index])))
    return dict(shape=list(values.shape), history=int(history),
                invalid_count=int(invalid.sum()), examples=examples)


def score_rows(raw, history):
    """Describe the first request's four uint16 score rows, without interpreting dtype."""
    values = np.asarray(raw, dtype=np.uint8).view('<u2').reshape(4, history)
    return [dict(query=q, nonzero=int(np.count_nonzero(row)),
                 zero=int(np.count_nonzero(row == 0)),
                 all_ones=int(np.count_nonzero(row == 0xffff)),
                 minimum=int(row.min()), maximum=int(row.max()))
            for q, row in enumerate(values)]


def actual_reads(mapping, selected, rows, resident):
    """Inspect completed replay outputs outside timing; count unique prefetches."""
    result = dict(
        prefetch_hot=0, prefetch_cold=0, unused_hot=0, unused_cold=0, final=int(rows[:, 247].astype(np.uint64).sum())
    )
    for row, tokens in zip(mapping, selected):
        fetched = np.flatnonzero(row >= 0)
        used = np.isin(fetched, tokens.reshape(-1))
        hot = fetched < resident
        result["prefetch_hot"] += int(hot.sum())
        result["prefetch_cold"] += int((~hot).sum())
        result["unused_hot"] += int((hot & ~used).sum())
        result["unused_cold"] += int((~hot & ~used).sum())
    prefetch = result["prefetch_hot"] + result["prefetch_cold"]
    if prefetch != int(rows[:, 9].astype(np.uint64).sum()):
        raise ValueError("Prefetch mapping and device request counts disagree")
    result["total"] = prefetch + result["final"]
    result["unused"] = result["unused_hot"] + result["unused_cold"]
    return result


def completion_failures(rows, posts, statuses, prefetch, final):
    """Decode existing device status buffers; no device work or state changes."""
    failures = []
    for b in range(len(rows)):
        cq = {
            name: words[2 * b : 2 * b + 2, :8].tolist()
            for name, words in (("prefetch_cq", prefetch), ("final_cq", final))
        }
        cq_bad = any(any(w[0] != 0x4B5641464247444F or w[2] for w in pair) for pair in cq.values())
        if not (rows[b, 11] or rows[b, 211] or rows[b, 248] or np.any(posts[b, :, 1:4]) or statuses[b, 0] or cq_bad):
            continue
        failures.append(
            dict(
                request=b,
                prefetch_error=int(rows[b, 11]),
                prefetch_pending=int(rows[b, 211]),
                final_pending=int(rows[b, 248]),
                final_requests=int(rows[b, 247]),
                resolver_status=statuses[b, :5].tolist(),
                posts_count_pending_error_deferred_descriptor=posts[b, :, :5].tolist(),
                **cq,
            )
        )
    return failures
