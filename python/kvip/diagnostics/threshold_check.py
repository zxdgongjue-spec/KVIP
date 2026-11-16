# SPDX-License-Identifier: Apache-2.0
"""Untimed proof that the native early threshold matches the requested seed."""

import numpy as np

from kvip.diagnostics.archive import cpu_array
from kvip.diagnostics.pipeline_options import early_threshold_skipped
from kvip.diagnostics.recent_threshold import recent_positions
from kvip.diagnostics.score_trace import sortable_to_bf16

_THRESHOLD_OFFSET = 4 * 2048 * 6
_PREFIX_ROW_BYTES = _THRESHOLD_OFFSET + 1024
_SENTINEL_BYTE = 0xA5


def poison_early_thresholds(bundle):
    """Untimed probe: overwrite only each query's two-byte threshold scalar."""
    prefix = bundle.rotating.arguments[13].reshape(-1)
    for row in range(bundle.rotating.batch):
        for query in range(4):
            start = row * _PREFIX_ROW_BYTES + _THRESHOLD_OFFSET + query * 256
            prefix[start:start + 2].fill_(_SENTINEL_BYTE)


def verify_early_threshold_sentinel(torch, bundle):
    """Detect a seed publication even when its speculative Get budget is zero."""
    batch = bundle.rotating.batch
    raw = cpu_array(torch, bundle.rotating.arguments[13].reshape(-1)[:batch * _PREFIX_ROW_BYTES])
    keys = raw.reshape(batch, _PREFIX_ROW_BYTES)[:, _THRESHOLD_OFFSET:].copy().view(np.uint16).reshape(batch, 4, 128)[:, :, 0]
    if not np.all(keys == (_SENTINEL_BYTE * 257)):
        raise AssertionError("Low-mode replay overwrote the skipped early-threshold sentinel")


def reference_thresholds(scores, query_ends, resident, bounds=None, *, rank=2048):
    if type(rank) is not int or not 1 <= rank <= 2048:
        raise ValueError("Threshold rank must be an integer in 1..2048")
    scores, ends = np.asarray(scores), np.asarray(query_ends).reshape(scores.shape[:2])
    if scores.dtype != np.uint16 or scores.ndim != 3 or scores.shape[1] != 4:
        raise ValueError("Expected sortable BF16 [B,4,H] scores")
    result = np.zeros(ends.shape, dtype=np.uint16)
    if bounds is not None and (
        np.asarray(bounds).shape != (len(scores), 2)
        or np.any(bounds < 0)
        or np.any(bounds[:, 1] < bounds[:, 0])
        or np.any(bounds > scores.shape[2])
    ):
        raise ValueError("Invalid recent threshold bounds")
    for b, q in np.ndindex(ends.shape):
        end = int(ends[b, q])
        if not 0 <= end <= scores.shape[2]:
            raise ValueError("Invalid query causal end")
        positions = np.arange(min(resident, end)) if bounds is None else recent_positions(resident, bounds[b], end)
        if len(positions) >= rank:
            result[b, q] = np.partition(scores[b, q, positions], len(positions) - rank)[-rank]
    return result


def verify_initial_threshold(torch, bundle, archive):
    scores = archive.array("post.score_keys")
    ends = archive.array("pre.query_ends")
    final = reference_thresholds(scores, ends, scores.shape[-1])
    if early_threshold_skipped(bundle.tuning):
        # This branch does not publish an early threshold. Never label stale
        # prefix scratch as computed/verified; final scores and TopK are still
        # checked by the complete replay oracle.
        return dict(status="not_computed", policy="recent", initial_keys=None,
                    final_keys=final.tolist(), included_in_timing=False,
                    reason="low_miss_skip with zero prefetch quota for all rows")
    b = archive.manifest["geometry"]["batch"]
    row_bytes = _PREFIX_ROW_BYTES
    raw = cpu_array(torch, bundle.rotating.arguments[13].reshape(-1)[: b * row_bytes]).reshape(b, row_bytes)
    # SaveResident publishes one uint16 scalar in each query's 256-byte bank.
    actual = raw[:, 4 * 2048 * 6 :].copy().view(np.uint16).reshape(b, 4, 128)[:, :, 0]
    resident = archive.manifest["geometry"]["resident"]
    bounds = archive.array("pre.threshold_bounds") if bundle.tuning["recent_threshold"] else None
    expected = reference_thresholds(scores, ends, resident, bounds)
    if not np.array_equal(actual, expected):
        mismatch = np.argwhere(actual != expected)[0]
        row, q = map(int, mismatch)
        raise AssertionError(
            f"Early threshold differs: row={row} q={q} actual={actual[row, q]} expected={expected[row, q]}"
        )

    def floats(keys):
        return (sortable_to_bf16(keys).astype(np.uint32) << 16).view(np.float32)

    early_value, final_value = floats(expected), floats(final)
    report = dict(
        status="bit_exact",
        policy="recent" if bounds is not None else "resident",
        initial_keys=actual.tolist(),
        final_keys=final.tolist(),
        mean_gap=float(np.mean(final_value - early_value)),
        equal_fraction=float(np.mean(actual == final)),
        included_in_timing=False,
    )
    rank = bundle.tuning.get("prefetch_rank", 0)
    if rank:
        gate = raw[:, 4 * 2048 * 6 :].copy().view(np.uint16).reshape(b, 4, 128)[:, :, 1]
        expected_gate = reference_thresholds(scores, ends, resident, bounds, rank=rank)
        if not np.array_equal(gate, expected_gate):
            raise AssertionError(f"Native speculative gate differs from seed rank {rank}")
        if np.any(gate < actual):
            raise AssertionError("Speculative rank gate fell below the safe K2048 threshold")
        report["prefetch_gate"] = dict(
            status="bit_exact", rank=rank, keys=gate.tolist(), queries=int(gate.size),
            above_final_queries=int(np.count_nonzero(gate > final)),
            equal_final_queries=int(np.count_nonzero(gate == final)),
            mean_raise_key_steps=float(np.mean(gate.astype(np.int32) - actual.astype(np.int32))),
            included_in_timing=False,
        )
    return report
