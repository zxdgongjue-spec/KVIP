# SPDX-License-Identifier: Apache-2.0
"""Exact sortable-BF16 observations and optional sparse real LI inputs.

The native log is active only while one eager invocation owns Context word 8.
Its owners remain retained on unknown completion, including failures between
before() and after(). Scores outside each query's causal prefix are archived
as scratch bits, never asserted to be computed or used as ranking candidates.
"""

import ctypes

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
# Deliberately retain before publishing the native pointer. If forward throws
# without calling after, Python GC must not release the pending write target.
_RETAINED_CAPTURES = []


def threshold_capacity(history):
    return (history + TRUNK - 1) // TRUNK + 2


def threshold_query_words(history):
    return ((threshold_capacity(history) * 4 + 31) // 32) * 32


def sortable_to_bf16(keys):
    keys = np.asarray(keys, dtype=np.uint16)
    return np.where(keys & np.uint16(0x8000), keys ^ np.uint16(0x8000), ~keys).astype(np.uint16)


def _shape(rotating):
    if hasattr(rotating, "runtime_shape"):
        shape = rotating.runtime_shape
        return int(shape.batch), int(shape.history), int(shape.splitQ), int(shape.liCores)
    shape = rotating.shape
    split = int(shape.batch * 2 <= shape.aic_count)
    function = rotating.library.v9_li_cores
    function.argtypes, function.restype = [], ctypes.c_uint32
    return shape.batch, shape.history, split, int(function())


def _check_score_geometry(torch, rotating):
    batch, history, split, cores = _shape(rotating)
    queries_per_core = 2 if split else 4
    if not 1 <= batch <= 16 or history % 256 or cores * queries_per_core != batch * 4:
        raise ValueError("Score capture requires exactly one complete Q2/Q4 task per LI core")
    arguments = rotating.arguments
    if len(arguments) < 14:
        raise ValueError("Score capture requires the native LI argument layout")
    if not np.array_equal(cpu_array(torch, arguments[5]).reshape(-1), np.full(batch, 4)) or not np.array_equal(
        cpu_array(torch, arguments[6]).reshape(-1), np.full(batch, history)
    ):
        raise ValueError("Score capture requires fixed full-history key lengths and Q4")
    workspace = arguments[12]
    if workspace.dtype != torch.uint8 or not workspace.is_contiguous() or workspace.numel() < batch * 4 * history * 2:
        raise ValueError("Score workspace is too small or has an unsupported layout")
    return batch, history, split


def read_score_keys(torch, rotating):
    """Read packed [B,Q4,H] keys from UBmem or URMA's unchanged score workspace.

    Caller synchronizes the native forward first. The one-task/core condition
    excludes workspace reuse across request groups. Only query_ends prefixes
    are semantically valid; no claim is made about future/padding scratch.
    """
    batch, history, _ = _check_score_geometry(torch, rotating)
    raw = cpu_array(torch, rotating.arguments[12].reshape(-1)[: batch * 4 * history * 2])
    return raw.view(np.uint16).reshape(batch, 4, history).copy()


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


class ScoreCapture:
    """Temporary native-log owner, separate from the compact on-disk schema."""

    def __init__(self, torch, rotating):
        self.torch, self.rotating = torch, rotating
        self.buffer, self.armed = None, False

    def before(self, selector, archive):
        torch, rotating = self.torch, self.rotating
        b, h, split = _check_score_geometry(torch, rotating)
        if getattr(rotating, "runtime_shape", None) is not None:
            raise ValueError("Live score capture currently requires UBmem")
        abi = getattr(rotating.library, "v9_threshold_trace_abi", None)
        if abi is None:
            raise ValueError("Native library lacks score-trace ABI; rebuild KVIP with its normal build cache")
        abi.argtypes, abi.restype = [], ctypes.c_uint32
        if abi() != 1:
            raise ValueError("Native score-trace ABI 1 is required")
        if np.any(cpu_array(torch, rotating.context)[64:72]):
            raise ValueError("Native diagnostic Context slot is already in use")
        archive.manifest["geometry"].update(
            split_q=bool(split),
            rolling_threshold=bool(rotating.shape.rolling_threshold),
            threshold_only=bool(rotating.shape.threshold_only),
        )
        real = archive.manifest["metadata"].get("capture_real_inputs", False)
        if type(real) is not bool:
            raise ValueError("capture_real_inputs must be an explicit boolean")
        archive.manifest["real_inputs_saved"] = real
        if real:
            self._real_inputs(selector, archive)
        words = b * 4 * threshold_query_words(h)
        allocation = torch.zeros(words + 32, dtype=torch.int32, device=rotating.context.device)
        offset = (-allocation.data_ptr()) % 128 // 4
        self.buffer = allocation[offset : offset + words].view(b, 4, threshold_query_words(h))
        self.pointer = torch.from_numpy(np.asarray([self.buffer.data_ptr()], dtype=np.uint64).view(np.uint8).copy()).to(
            rotating.context.device
        )
        _RETAINED_CAPTURES.append(self)
        self.armed = True
        rotating.context[64:72].copy_(self.pointer)
        rotating.lease.check_stream().synchronize()

    def _real_inputs(self, selector, archive):
        torch = self.torch
        owners = archive.array("pre.row_owners")
        g = archive.manifest["geometry"]
        b, h, heads = g["batch"], g["history"], g["heads"]
        for name, tensor, dtype, shape in (
            ("keys", selector.storage.keys, torch.uint8, (b, h, 128)),
            ("key_scales", selector.storage.scales, torch.float32, (b, h)),
        ):
            if tensor.dtype != dtype:
                raise ValueError(f"Unexpected real LI {name} dtype")
            archive.arrays["real." + name] = np.stack(
                [cpu_array(torch, tensor[int(owner)]) for owner in owners]
            ).reshape(shape)
        for name, tensor, dtype, shape in (
            ("query", selector.inputs.query, torch.float8_e4m3fn, (b, 4, heads, 128)),
            ("weights", selector.inputs.weights, torch.bfloat16, (b, 4, heads)),
            ("query_scale", selector.inputs.scale, torch.float32, (b, 4, heads)),
        ):
            if tensor.dtype != dtype:
                raise ValueError(f"Unexpected real LI {name} dtype")
            archive.arrays["real." + name] = cpu_array(torch, tensor).reshape(shape).copy()

    def after(self, archive, rows, mapping):
        a, g = archive.arrays, archive.manifest["geometry"]
        b, h, windows = g["batch"], g["history"], g["rounds"]
        a["post.score_keys"] = read_score_keys(self.torch, self.rotating)
        raw = cpu_array(self.torch, self.buffer)
        a["post.threshold_records"] = (
            raw[:, :, : threshold_capacity(h) * 4].reshape(b, 4, threshold_capacity(h), 4).copy()
        )
        for row in range(b):
            for query in range(4):
                end = int(a["pre.query_ends"][row, query])
                selected = a["post.topk"][row, query]
                a["post.threshold_records"][row, query, -1] = [
                    end,
                    h,
                    int(np.min(a["post.score_keys"][row, query, selected])),
                    4,
                ]
        a["post.window_posted"] = rows[:, 16 : 16 + windows].copy()
        a["post.window_processed_ends"] = rows[:, 80 : 80 + 4 * windows].reshape(b, windows, 4).copy()
        fetched = a["post.prefetch_records"]
        a["post.prefetch_ordinals"] = (mapping[fetched[:, 0], fetched[:, 1]] - g["resident"]).astype(np.int32)
        validate_scores(archive)
        archive.manifest["score_contract"] = dict(
            encoding="uint16 FloatToSortableKey(bfloat16 CAST_ROUND score), not raw BF16 bits",
            coordinates="packed B/Q4, pre-call physical Index position; positions >= query_ends are invalid scratch",
            predicate="strict unsigned score_key > threshold_key; equality is repaired by final TopK/Get",
            threshold_columns=["valid_prefix_end", "applies_from_physical_position", "sortable_key", "kind"],
            threshold_kinds={
                "0": "unused zero record",
                "1": "native resident TopK",
                "2": "native threshold-only",
                "3": "native rolling TopK",
                "4": "derived final Kth, not a native observation",
            },
            prefetch_order="Pair prefetch_records with prefetch_ordinals; split admission order by cumulative window_posted counts",
            timing="Diagnostic log writes/readbacks perturb capture; archived score injection excludes real LI computation",
        )
        self.disarm()

    def disarm(self):
        if not self.armed:
            return
        # Unknown completion deliberately keeps this owner in the module list.
        # Clearing a pointer is safe only after every earlier writer completes.
        stream = self.rotating.lease.check_stream()
        stream.synchronize()
        self.rotating.context[64:72].zero_()
        stream.synchronize()
        self.armed = False
        _RETAINED_CAPTURES.remove(self)
        self.buffer = self.pointer = None
