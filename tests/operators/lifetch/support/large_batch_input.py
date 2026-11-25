# SPDX-License-Identifier: Apache-2.0
"""Expand a real BS16 Q4 trace, preserving each donor's scores and cache state."""

from dataclasses import fields
import hashlib
import json
from pathlib import Path

import numpy as np

from workloads import ReplayRound, ReplaySequence


BATCHES = (20, 24, 28, 32)
GROUPS = (10, 4, 17, 7, 2)


class FrozenRound:
    def __init__(self, inputs, refs):
        self.inputs = inputs
        self.rounds = (tuple(refs),)

    def __iter__(self):
        yield self.inputs


def expand(source, batch):
    if source.geometry["batch"] != 16 or batch not in BATCHES:
        raise ValueError("Expected a BS16 donor and a supported target batch")
    # Sample donor rows across the entire capture rather than concentrating the
    # additional requests on its first few DPs. Both arms use the same order.
    extra = batch - 16
    donors = list(range(16)) + [i * 16 // extra for i in range(extra)]
    arrays = {
        f.name: getattr(source, f.name)[donors]
        for f in fields(source)
        if isinstance(getattr(source, f.name), np.ndarray)
    }
    # Page aliases in the original 16 are preserved. The added bank is private
    # to these requests and preserves aliases among its own selected donor DPs.
    identities = tuple((i // 16, *source.identities[row]) for i, row in enumerate(donors))
    page_keys = tuple(
        tuple(None if p is None else (i // 16, *p) for p in source.page_keys[row]) for i, row in enumerate(donors)
    )
    inputs = ReplayRound(
        source.source_round,
        source.scheduler_round,
        identities,
        dict(source.geometry, batch=batch),
        page_keys=page_keys,
        source_paths=source.source_paths,
        **arrays,
    )
    return inputs, donors


def load(trace, group, source_round, batch):
    snapshot = Path(trace) / "manifest.json"
    manifest = json.loads(snapshot.read_text()) if snapshot.is_file() else {}
    if manifest.get("metadata", {}).get("merged_from"):
        from large_batch_merged import merged_refs, merged_round

        if source_round != 1:
            raise ValueError("Merged capture contains one replay round")
        refs = merged_refs(snapshot.resolve(), manifest, group)
        source = merged_round(refs)
    else:
        sequence = ReplaySequence(trace, group, (source_round, source_round))
        (refs,) = sequence.rounds
        source = next(iter(sequence))
    g = source.geometry
    if (
        (g["batch"], g["heads"], g["history"], g["resident"], g["token_bytes"], g["output_stride"])
        != (16, 32, 131328, 16384, 656, 16896)
        or source.query_ends.shape != (16, 4)
        or np.any(np.diff(source.query_ends, axis=1) != 1)
    ):
        raise ValueError(f"Expected the captured 128K 16-request Q4 workload, got {g}")
    inputs, donors = expand(source, batch)
    return FrozenRound(inputs, refs), donors


def input_digest(inputs, pages, donors):
    """Fingerprint operands, scoring oracle, cache/LRU, aliases and request order."""
    digest = hashlib.sha256()
    metadata = (inputs.geometry, inputs.identities, inputs.source_round, inputs.scheduler_round, donors)
    digest.update(json.dumps(metadata, sort_keys=True).encode())
    for key, array in [(f.name, getattr(inputs, f.name)) for f in fields(inputs)] + [
        ("page_indices", pages.page_indices)
    ]:
        if isinstance(array, np.ndarray):
            digest.update(json.dumps((key, array.shape, array.dtype.str)).encode())
            digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()
