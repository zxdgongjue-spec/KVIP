# SPDX-License-Identifier: Apache-2.0
"""CPU-only fixture audit; reconcile measured counters without emulating NPU timing."""

import numpy as np


def actual_reads(mapping, selected, rows, resident, actual_key_lengths=None):
    """Count actual Prefetch Mapping in each request's valid physical K prefix.

    With Active-K Reset the out-of-K suffix is intentionally NOT cleared and
    may contain positive poison/sentinel values. It is not a Prefetch record.
    Continue to enforce device RequestCount == in-K positive Mapping count.
    """
    mapping = np.asarray(mapping)
    selected = np.asarray(selected)
    rows = np.asarray(rows)
    if (
        mapping.ndim != 2 or selected.ndim < 2 or rows.ndim != 2
        or selected.shape[0] != mapping.shape[0]
        or rows.shape[0] != mapping.shape[0]
    ):
        raise ValueError("Prefetch audit batch geometry mismatch")
    batch, history = mapping.shape
    if actual_key_lengths is None:
        lengths = (history,) * batch
    else:
        lengths = tuple(actual_key_lengths)
        if len(lengths) != batch or any(
            not isinstance(k, (int, np.integer)) or k < 0 or k > history
            for k in lengths
        ):
            raise ValueError("Prefetch audit invalid per-request actual K")

    result = dict(
        prefetch_hot=0, prefetch_cold=0, unused_hot=0, unused_cold=0,
        final=int(rows[:, 247].astype(np.uint64).sum()),
    )
    ignored_suffix = 0
    for index, (row, tokens) in enumerate(zip(mapping, selected)):
        k = int(lengths[index])
        if actual_key_lengths is not None and (
            np.any(tokens < 0) or np.any(tokens >= k)
        ):
            raise ValueError(f"Prefetch audit selected out-of-K Token: request={index} K={k}")
        # ONLY the current valid prefix can be consumed by the active request.
        fetched = np.flatnonzero(row[:k] >= 0)
        if actual_key_lengths is not None:
            ignored_suffix += int(np.count_nonzero(row[k:] >= 0))
        used = np.isin(fetched, tokens.reshape(-1))
        hot = fetched < resident
        result["prefetch_hot"] += int(hot.sum())
        result["prefetch_cold"] += int((~hot).sum())
        result["unused_hot"] += int((hot & ~used).sum())
        result["unused_cold"] += int((~hot & ~used).sum())
    prefetch = result["prefetch_hot"] + result["prefetch_cold"]
    device_prefetch = int(rows[:, 9].astype(np.uint64).sum())
    if prefetch != device_prefetch:
        raise ValueError(
            "Prefetch mapping and device request counts disagree: "
            f"mapping_valid_K={prefetch} device={device_prefetch} "
            f"actual_K={list(lengths)}"
        )
    result["total"] = prefetch + result["final"]
    result["unused"] = result["unused_hot"] + result["unused_cold"]
    if actual_key_lengths is not None:
        result["ignored_out_of_K_mapping"] = ignored_suffix
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
