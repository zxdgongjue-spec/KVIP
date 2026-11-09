# SPDX-License-Identifier: Apache-2.0
"""Conservative selected-set equivalence across native and fused LI ordering.

The reference uses the captured FP8/scale/weight values, not a similarity or
recall threshold. It is not a bit-exact simulation of either NPU implementation.
Only replacements in the captured selection's minimum BF16 score bin qualify.
"""

import numpy as np

from kvip.diagnostics.archive import array_digest

TOPK = 2048
QUERIES = 4
REFERENCE = (
    "FP64 selected-union scores cast FP32 then BF16 CAST_ROUND (ties away); equivalence against captured selection, "
    "not full-history LI proof or bit-exact CANN arithmetic"
)


class SelectionOracle:
    def __init__(self, archive, inputs):
        self.archive, self.inputs = archive, inputs
        self.batch = archive.manifest["geometry"]["batch"]
        self.rows = self.batch * QUERIES
        self.ends = np.asarray(inputs["query_ends"]).reshape(self.rows)
        self.owners = np.repeat(np.asarray(inputs["owners"]), QUERIES)
        self.expected = self._validate(archive.array("post.logical_output"), "captured")
        self.verified_report = None
        self.score_inputs = None

    def _validate(self, logical, name):
        values = np.asarray(logical)
        if values.dtype.kind not in "iu" or values.size != self.rows * TOPK:
            raise AssertionError(f"{name} selection must contain B*4*2048 integer logical IDs")
        values = values.reshape(self.rows, TOPK)
        for row, selected in enumerate(values):
            if np.any(selected < 0) or np.any(selected >= self.ends[row]):
                raise AssertionError(
                    f"{name} selection violates causal bound: query={row} owner={self.owners[row]} "
                    f"query_end={self.ends[row]}"
                )
        ordered = np.sort(values, axis=1)
        duplicate = np.flatnonzero(np.any(np.diff(ordered, axis=1) == 0, axis=1))
        if len(duplicate):
            row = int(duplicate[0])
            raise AssertionError(f"{name} selection has duplicate IDs: query={row} owner={self.owners[row]}")
        if np.any(ordered > np.iinfo(np.int32).max):
            raise AssertionError(f"{name} selection exceeds int32 logical IDs")
        return np.ascontiguousarray(ordered, dtype="<i4")

    def _load_scores(self):
        # Exact-set matches need no Torch import, tensor decode or score work.
        if self.score_inputs is not None:
            return self.score_inputs
        import torch

        expected_types = {"query": "float8_e4m3fn", "weights": "bfloat16", "query_scale": "float32"}
        for name, dtype in expected_types.items():
            if self.archive.manifest["tensors"]["pre." + name]["dtype"] != dtype:
                raise AssertionError(f"Cannot prove score ties with unexpected captured {name} dtype")
        query, weights, scale = (
            self.archive.tensor("pre." + name, torch, "cpu").to(torch.float64).numpy() for name in expected_types
        )
        unpack = np.asarray(self.inputs["unpack"])
        if unpack.shape != (self.rows,) or not np.array_equal(np.sort(unpack), np.arange(self.rows)):
            raise AssertionError("Cannot prove score ties with invalid packed-to-TND query mapping")
        query, weights, scale = query[unpack], weights[unpack], scale[unpack]
        heads = self.archive.manifest["geometry"]["heads"]
        if (
            query.shape != (self.rows, heads, 128)
            or weights.shape != (self.rows, heads)
            or scale.shape != weights.shape
        ):
            raise AssertionError("Cannot prove score ties with inconsistent Q/weight/scale shapes")
        if not all(np.all(np.isfinite(value)) for value in (query, weights, scale)):
            raise AssertionError("Cannot prove score ties with nonfinite Q/weight/scale inputs")
        keys, key_scale = self.inputs["keys"], self.inputs["scales"]
        if keys.dtype != np.uint8 or key_scale.dtype != np.float32 or keys.shape[:2] != key_scale.shape:
            raise AssertionError("Cannot prove score ties with unexpected logical Index key/scale storage")
        self.score_inputs = torch, query, weights, scale
        return self.score_inputs

    def _scores(self, row, tokens):
        torch, query, weights, scale = self._load_scores()
        owner = int(self.owners[row])
        raw = np.ascontiguousarray(self.inputs["keys"][owner, tokens])
        keys = torch.from_numpy(raw).view(torch.float8_e4m3fn).to(torch.float64).numpy()
        key_scale = self.inputs["scales"][owner, tokens].astype(np.float64)
        if keys.shape != (len(tokens), 128) or not np.all(np.isfinite(keys)) or not np.all(np.isfinite(key_scale)):
            raise AssertionError(f"Cannot prove score ties with invalid Index values: query={row} owner={owner}")
        qk = keys @ query[row].T
        scores = (np.maximum(qk, 0) * (weights[row] * scale[row])[None, :]).sum(axis=1) * key_scale
        if not np.all(np.isfinite(scores)):
            raise AssertionError(f"Nonfinite reference scores: query={row} owner={owner}")
        # Match the established Ascend CAST_ROUND reference in the LI fixture;
        # Torch's default BF16 ties-to-even can merge distinct boundary bins.
        with np.errstate(over="ignore", invalid="ignore"):
            bits = scores.astype(np.float32).view(np.uint32)
        bf16 = ((bits + np.uint32(0x8000)) >> 16).astype(np.uint16)
        rounded = (bf16.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        if not np.all(np.isfinite(rounded)):
            raise AssertionError(f"Reference scores overflow BF16: query={row} owner={owner}")
        return scores, rounded

    def verify(self, logical):
        actual = self._validate(logical, "native")
        digest = array_digest(actual)
        if self.verified_report is not None:
            if digest != self.verified_report["selected_set_sha256"]:
                raise AssertionError("Native LI selected set changed across eager/graph replays")
            return dict(self.verified_report)
        report = dict(
            status="exact",
            selected_set_sha256=digest,
            compared_queries=self.rows,
            changed_queries=0,
            total_replacements=0,
            min_overlap_fraction=1.0,
            reference=REFERENCE,
        )
        details = []
        for row, (native, captured) in enumerate(zip(actual, self.expected)):
            if np.array_equal(native, captured):
                continue
            native_only = np.setdiff1d(native, captured, assume_unique=True)
            captured_only = np.setdiff1d(captured, native, assume_unique=True)
            union = np.union1d(native, captured)
            scores, bins = self._scores(row, union)
            native_indices, captured_indices = (
                np.searchsorted(union, native_only),
                np.searchsorted(union, captured_only),
            )
            boundary = float(np.min(bins[np.searchsorted(union, captured)]))
            replacements = len(native_only)
            overlap = (TOPK - replacements) / TOPK
            if np.any(bins[native_indices] != boundary) or np.any(bins[captured_indices] != boundary):
                changed = np.r_[native_indices, captured_indices]
                raise AssertionError(
                    "Native LI selection is not a proven BF16 boundary tie: "
                    f"query={row} owner={int(self.owners[row])} overlap={TOPK - replacements}/{TOPK} "
                    f"replacements={replacements} captured_min_bf16={boundary:.9g} "
                    f"changed_score_range=[{scores[changed].min():.9g},{scores[changed].max():.9g}] "
                    f"native_only_bf16_range=[{bins[native_indices].min():.9g},{bins[native_indices].max():.9g}] "
                    f"captured_only_bf16_range=[{bins[captured_indices].min():.9g},{bins[captured_indices].max():.9g}] "
                    f"native_only_ids={native_only[:8].tolist()} captured_only_ids={captured_only[:8].tolist()}; "
                    + REFERENCE
                )
            report["changed_queries"] += 1
            report["total_replacements"] += replacements
            report["min_overlap_fraction"] = min(report["min_overlap_fraction"], overlap)
            details.append(
                dict(query=row, owner=int(self.owners[row]), replacements=replacements, boundary_bf16=boundary)
            )
        if details:
            report.update(status="bf16_boundary_ties", boundary_details=details)
        self.verified_report = report
        return dict(report)
