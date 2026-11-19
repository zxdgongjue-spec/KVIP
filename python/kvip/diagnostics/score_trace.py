# SPDX-License-Identifier: Apache-2.0
"""Read historical sortable-BF16 traces and optional real LI inputs on CPU.

Future/padding scores are scratch bits, never ranking candidates. This reader
does not enable capture on a newer kernel or alter the serving path.
"""

import numpy as np

from kvip.diagnostics.archive import cpu_array


KIND = "kvaf_decode_score_trace"
SCHEMA_VERSION = 4
TRUNK = 16384
SCORE_ARRAYS = {
    "post.score_keys",
    "post.threshold_records",
    "post.window_posted",
    "post.window_processed_ends",
    "post.prefetch_ordinals",
}
REAL_ARRAYS = {"real.keys", "real.key_scales", "real.query", "real.weights", "real.query_scale"}


def read_score_keys(torch, rotating):
    """Read every packed Q4 score from the current streaming workspace.

    The caller must synchronize the native forward first. This function only
    reads immutable completed output; it never installs a capture buffer or
    participates in timing. Reject core reuse, which would leave only the last
    request's scores in each core's workspace.
    """
    if getattr(rotating, "ranked_port", False):
        from kvip.diagnostics.ranked_port import score_keys

        return score_keys(torch, rotating)
    shape = rotating.runtime_shape
    batch, history, split, cores = (int(getattr(shape, name)) for name in ("batch", "history", "splitQ", "liCores"))
    if split not in (0, 1) or not 1 <= batch <= 16 or history <= 0 or history % 256:
        raise ValueError("Score readback requires aligned B1..16 full-history Q4 geometry")
    if cores * (2 if split else 4) != batch * 4:
        raise ValueError("Score workspace reuses a core across requests; a complete Q4 readback is unavailable")
    arguments = rotating.arguments
    if len(arguments) < 14:
        raise ValueError("Score readback requires the production LI argument layout")
    if not np.array_equal(cpu_array(torch, arguments[5]).reshape(-1), np.full(batch, 4)) or not np.array_equal(
        cpu_array(torch, arguments[6]).reshape(-1), np.full(batch, history)
    ):
        raise ValueError("Score readback requires all four queries and the complete Index history")
    workspace = arguments[12]
    size = batch * 4 * history * 2
    if workspace.dtype != torch.uint8 or not workspace.is_contiguous() or workspace.numel() < size:
        raise ValueError("Score workspace is too small or has an unsupported byte layout")
    return cpu_array(torch, workspace.reshape(-1)[:size]).view(np.uint16).reshape(batch, 4, history).copy()


def threshold_capacity(history):
    return (history + TRUNK - 1) // TRUNK + 2


def threshold_query_words(history):
    return ((threshold_capacity(history) * 4 + 31) // 32) * 32


def sortable_to_bf16(keys):
    keys = np.asarray(keys, dtype=np.uint16)
    return np.where(keys & np.uint16(0x8000), keys ^ np.uint16(0x8000), ~keys).astype(np.uint16)


def _expected_thresholds(g, end):
    resident, history = g["resident"], g["history"]
    valid_resident = min(resident, end)
    threshold_only = g["threshold_only"] and 2048 <= valid_resident <= TRUNK
    result = [(0, valid_resident, resident, 2 if threshold_only else 1)]
    if g["rolling_threshold"]:
        previous = resident
        for available in list(range(2 * TRUNK, history + 1, TRUNK)) + ([history] if history % TRUNK else []):
            valid = min(available, end)
            if valid > previous:
                result.append(((available + TRUNK - 1) // TRUNK - 1, valid, available, 3))
                previous = valid
    return result


def validate_scores(archive):
    a, g = archive.arrays, archive.manifest["geometry"]
    b, h, heads, windows = (g[name] for name in ("batch", "history", "heads", "rounds"))
    if (
        type(g.get("split_q")) is not bool
        or type(g.get("threshold_only")) is not bool
        or type(g.get("rolling_threshold")) is not bool
    ):
        raise ValueError("Score trace must record exact LI/threshold modes")
    if g["rolling_threshold"] and (g["resident"] != TRUNK or not g["split_q"] or g["threshold_only"]):
        raise ValueError("Rolling score trace has an unsupported native geometry")
    specs = {
        "post.score_keys": (np.uint16, (b, 4, h)),
        "post.threshold_records": (np.int32, (b, 4, threshold_capacity(h), 4)),
        "post.window_posted": (np.int32, (b, windows)),
        "post.window_processed_ends": (np.int32, (b, windows, 4)),
        "post.prefetch_ordinals": (np.int32, (len(a["post.prefetch_records"]),)),
    }
    real = archive.manifest.get("real_inputs_saved")
    if type(real) is not bool or real != archive.manifest["metadata"].get("capture_real_inputs", False):
        raise ValueError("Real LI sample marker differs from its capture request")
    if real:
        specs.update(
            {
                "real.keys": (np.uint8, (b, h, 128)),
                "real.key_scales": (np.float32, (b, h)),
                "real.query": (np.uint8, (b, 4, heads, 128)),
                "real.weights": (np.uint16, (b, 4, heads)),
                "real.query_scale": (np.float32, (b, 4, heads)),
            }
        )
    if (set(a) & REAL_ARRAYS) != (REAL_ARRAYS if real else set()):
        raise ValueError("Real LI input arrays do not match the explicit sample marker")
    for name, (dtype, shape) in specs.items():
        if a[name].dtype != dtype or a[name].shape != shape:
            raise ValueError(f"Score trace dtype/shape differs: {name}")
    records = a["post.threshold_records"]
    for row in range(b):
        for query in range(4):
            end = int(a["pre.query_ends"][row, query])
            scores = a["post.score_keys"][row, query]
            bits = sortable_to_bf16(scores[:end])
            if np.any(((bits & 0x7F80) == 0x7F80) & ((bits & 0x7F) != 0)):
                raise ValueError("Actual score prefix contains NaN/uninitialized sortable keys")
            selected = a["post.topk"][row, query]
            if np.any(selected >= end):
                raise ValueError("Physical TopK position exceeds its query score prefix")
            kth = int(np.partition(scores[:end], end - 2048)[end - 2048])
            chosen = scores[selected]
            if np.min(chosen) != kth or np.count_nonzero(chosen > kth) != np.count_nonzero(scores[:end] > kth):
                raise ValueError("Actual TopK does not match the complete captured score ranking")
            expected = np.zeros_like(records[row, query])
            for slot, valid, applies, kind in _expected_thresholds(g, end):
                threshold = int(np.partition(scores[:valid], valid - 2048)[valid - 2048]) if valid >= 2048 else 0
                expected[slot] = [valid, applies, threshold, kind]
            expected[-1] = [end, h, kth, 4]  # Explicitly derived; not a native log entry.
            if not np.array_equal(records[row, query], expected):
                raise ValueError("Actual native threshold log differs from captured score Kth values")
    posted, processed = a["post.window_posted"], a["post.window_processed_ends"]
    if np.any(posted < 0) or not np.array_equal(posted.sum(axis=1), a["post.read_counts"][:, 0]):
        raise ValueError("Window posted counts differ from actual prefetch transfers")
    if np.any(processed < 0) or np.any(processed > h) or np.any(np.diff(processed, axis=1) < 0):
        raise ValueError("Window processed bounds are outside monotonic physical history")
    for row in range(b):
        ordinals = a["post.prefetch_ordinals"][a["post.prefetch_records"][:, 0] == row]
        if not np.array_equal(np.sort(ordinals), np.arange(int(posted[row].sum()))):
            raise ValueError("Prefetch admission ordinals do not cover the exact posted sequence")
