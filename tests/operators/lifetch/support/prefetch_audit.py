# SPDX-License-Identifier: Apache-2.0
"""CPU-only fixture audit; reconcile measured counters without emulating NPU timing."""

import numpy as np


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
