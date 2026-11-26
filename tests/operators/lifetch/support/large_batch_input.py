# SPDX-License-Identifier: Apache-2.0
"""Expand a real BS16 Q4 trace, preserving each donor's scores and cache state."""

from dataclasses import fields
import hashlib
import json
from pathlib import Path

import numpy as np

from workloads import CaptureReference, ReplayRound, ReplaySequence


BATCHES = (20, 24, 28, 32)
FIXED_BATCHES = (1, 2, 4, 8, 16, *BATCHES)
GROUPS = (10, 4, 17, 7, 2)


class FrozenRound:
    def __init__(self, inputs, refs):
        self.inputs = inputs
        self.rounds = (tuple(refs),)

    def __iter__(self):
        yield self.inputs


def expand(source, batch, cohort=0):
    if source.geometry["batch"] != 16 or batch not in FIXED_BATCHES:
        raise ValueError("Expected a BS16 donor and a supported target batch")
    # Sample donor rows across the entire capture rather than concentrating the
    # additional requests on its first few DPs. Both arms use the same order.
    cohorts = max(1, 16 // batch)
    if not 0 <= cohort < cohorts:
        raise ValueError(f"BS{batch} cohort {cohort} outside 0..{cohorts - 1}")
    if batch <= 16:
        # Round-robin cohorts preserve every BS16 donor exactly once per BS.
        donors = list(range(cohort, 16, cohorts))
    else:
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


def read_source(trace, group, source_round):
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
    return source, refs


def save_source(directory, trace, group, source_round, source, refs):
    """Write one verified donor bank for the whole group, outside any timing."""
    directory.mkdir()
    attributes = {}
    arrays = []
    for field in fields(source):
        value = getattr(source, field.name)
        if isinstance(value, np.ndarray):
            np.save(directory / (field.name + ".npy"), value, allow_pickle=False)
            arrays.append(field.name)
        else:
            attributes[field.name] = value
    evidence = [dict(dp_rank=r.dp_rank, source_round=r.source_round, scheduler_round=r.scheduler_round,
                     group=r.group, manifest_path=str(r.manifest_path), request_signature=r.request_signature)
                for r in refs]
    index = dict(trace=str(trace.resolve()), group=group, source_round=source_round, attributes=attributes,
                 arrays=arrays, refs=evidence)
    (directory / "source.json").write_text(json.dumps(index))


def mapped_source(directory, trace, group, source_round):
    index = json.loads((directory / "source.json").read_text())
    if (index["trace"], index["group"], index["source_round"]) != (str(trace.resolve()), group, source_round):
        raise ValueError("Prepared bank belongs to a different trace, group or round")
    values = index["attributes"]
    values["identities"] = tuple(map(tuple, values["identities"]))
    values["page_keys"] = tuple(tuple(None if key is None else tuple(key) for key in row) for row in values["page_keys"])
    values["source_paths"] = tuple(values["source_paths"])
    for name in index["arrays"]:
        values[name] = np.load(directory / (name + ".npy"), allow_pickle=False, mmap_mode="r")
    refs = [CaptureReference(**dict(r, manifest_path=Path(r["manifest_path"]),
                                    request_signature=tuple(map(tuple, r["request_signature"]))))
            for r in index["refs"]]
    return ReplayRound(**values), refs


def load(trace, group, source_round, batch, cohort=0, prepared=None):
    if prepared is None:
        source, refs = read_source(trace, group, source_round)
    else:
        source, refs = mapped_source(prepared, trace, group, source_round)
    inputs, donors = expand(source, batch, cohort)
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare one verified read-only BS16 donor bank")
    parser.add_argument("trace", type=Path)
    parser.add_argument("group", type=int, choices=GROUPS)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    source, refs = read_source(args.trace, args.group, 1)
    save_source(args.directory, args.trace, args.group, 1, source, refs)
