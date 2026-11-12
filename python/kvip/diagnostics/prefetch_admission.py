# SPDX-License-Identifier: Apache-2.0
"""Read existing Row counters after completion; never instrument the kernel.

Suppressed candidates are Q4-unioned, mapping-deduplicated candidates examined
in one request/window but not admitted. Their sum is not a global unique-token
count or final cold-miss count. Exhausted windows skip scanning, so their unseen
candidates are not included. A window-limit attribution requires positive
suppression and a *strictly* smaller window allowance than every remaining
row/total/transport budget; ties stay explicitly unclassified.
"""

import numpy as np


ADMISSION_ACCOUNTING = {
    "scope": "epoch counters cover every packed row, including padding; remaining budget uses active rows only",
    "suppressed_candidates": (
        "examined candidates unioned across Q4 in one request/window after existing-map filtering; "
        "tokens can recur across windows or requests, so this is neither globally unique demand nor final cold misses"
    ),
    "window_limited_epochs": "positive suppression with window_limit strictly below all other remaining budgets",
    "other_or_tied_limited_epochs": "positive suppression attributed to a row/total/transport limit or a tie, not separated further",
    "budget_exhausted_epochs": "visited windows with zero admissible budget; scanning was skipped, so unseen candidates are unknown",
    "remaining_admissible_budget": (
        "sum of min(row-slot,total-prefetch,transport limits) minus admitted requests for active rows; "
        "unused capacity does not prove that raising a window limit improves useful prefetch"
    ),
}


def prefetch_admission(rows, *, window_limit, rounds, active_rows=None):
    """Derive admission evidence from the already-read 1024-byte native Rows."""
    values = np.asarray(rows)
    if (
        values.ndim != 2
        or values.shape[1] != 256
        or not len(values)
        or values.dtype not in (np.dtype("int32"), np.dtype("uint32"))
    ):
        raise ValueError("Prefetch admission requires int32/uint32 Row[B,256]")
    if (
        type(rounds) is not int
        or not 1 <= rounds <= 32
        or type(window_limit) is not int
        or not 1 <= window_limit <= 480
    ):
        raise ValueError("Invalid prefetch admission window geometry")
    active = np.ones(len(values), dtype=bool) if active_rows is None else np.asarray(active_rows)
    if active.dtype != np.dtype(bool) or active.shape != (len(values),):
        raise ValueError("Prefetch admission active rows must be a boolean B-vector")
    # Row ABI words: requests=9, suppressed=10, epochCount=16, epochSuppressed=48,
    # limits=208:211, epochOrder=214, sequence=246. Signed tensor storage carries
    # uint32 words; widen before all sums/subtractions to avoid wraparound.
    words = values.view(np.uint32).astype(np.int64)
    posted = words[:, 16 : 16 + rounds]
    suppressed = words[:, 48 : 48 + rounds]
    orders = words[:, 214 : 214 + rounds]
    visited = orders != 0
    if any(np.any(words[:, start + rounds : start + 32]) for start in (16, 48, 214)):
        raise ValueError("Prefetch admission contains counters outside configured windows")
    counts = visited.sum(axis=1)
    # Current FastService waits for the preceding epoch's complete event, but
    # account from the recorded service order rather than assuming its order.
    order_keys = np.where(visited, orders, rounds + 1)
    expected_order = np.where(np.arange(rounds) < counts[:, None], np.arange(rounds) + 1, rounds + 1)
    if not np.array_equal(np.sort(order_keys, axis=1), expected_order) or not np.array_equal(words[:, 246], counts):
        raise ValueError("Prefetch admission window order/sequence mismatch")
    if np.any((posted != 0) & ~visited) or np.any((suppressed != 0) & ~visited):
        raise ValueError("Unvisited prefetch windows contain admission counters")
    if not np.array_equal(posted.sum(axis=1), words[:, 9]) or not np.array_equal(words[:, 8], words[:, 9]):
        raise ValueError("Prefetch admission posted/slot/request counters disagree")
    if not np.array_equal(suppressed.sum(axis=1), words[:, 10]):
        raise ValueError("Prefetch admission suppression counters disagree")
    service_order = np.argsort(order_keys, axis=1)
    ordered_posts = np.take_along_axis(posted, service_order, axis=1)
    prior = np.empty_like(posted)
    np.put_along_axis(prior, service_order, np.cumsum(ordered_posts, axis=1) - ordered_posts, axis=1)
    effective = words[:, 208:211].min(axis=1)
    remaining = effective[:, None] - prior
    allowance = np.minimum(window_limit, np.maximum(remaining, 0))
    if np.any(posted > allowance) or np.any(words[:, 9] > effective):
        raise ValueError("Prefetch admission exceeds an actual row budget")
    limited = suppressed > 0
    if np.any(limited & ((posted != allowance) | (allowance == 0))):
        raise ValueError("Prefetch suppression does not match the admitted allowance")
    # Strict inequality deliberately excludes simultaneous window/total limits.
    window_limited = limited & (window_limit < remaining)
    left = effective - words[:, 9]
    return dict(
        rounds=rounds,
        window_limit=window_limit,
        packed_rows=len(values),
        active_rows=int(active.sum()),
        visited_epochs=int(visited.sum()),
        unvisited_epochs=int((~visited).sum()),
        window_limited_epochs=int(window_limited.sum()),
        other_or_tied_limited_epochs=int((limited & ~window_limited).sum()),
        budget_exhausted_epochs=int((visited & (allowance == 0)).sum()),
        suppressed_candidates=int(suppressed.sum()),
        remaining_admissible_budget=int(left[active].sum()),
        rows_suppressed_with_budget_left=int(np.count_nonzero(active & (left > 0) & (words[:, 10] > 0))),
        posted_by_epoch=posted.sum(axis=0).tolist(),
        suppressed_by_epoch=suppressed.sum(axis=0).tolist(),
    )


def mean_admission(samples):
    """Summarize compatible per-replay evidence; keep old saved reports readable."""
    blocks = [sample.get("prefetch_admission") for sample in samples]
    if not blocks or all(block is None for block in blocks):
        return None
    if any(not isinstance(block, dict) for block in blocks):
        raise ValueError("Mixed or invalid prefetch admission samples")
    first = blocks[0]
    if any(set(block) != set(first) for block in blocks):
        raise ValueError("Prefetch admission sample fields differ")
    result = {}
    for key in first:
        values = [block[key] for block in blocks]
        if key in ("rounds", "window_limit", "packed_rows", "active_rows"):
            if any(value != values[0] for value in values):
                raise ValueError("Prefetch admission geometry changed between samples")
            result[key] = values[0]
            continue
        data = np.asarray(values, dtype=np.float64)
        expected = (len(blocks), first["rounds"]) if key.endswith("_by_epoch") else (len(blocks),)
        if data.shape != expected or not np.all(np.isfinite(data)) or np.any(data < 0):
            raise ValueError("Invalid prefetch admission counters")
        mean = data.mean(axis=0)
        result[key] = mean.tolist() if data.ndim == 2 else float(mean)
    return result
