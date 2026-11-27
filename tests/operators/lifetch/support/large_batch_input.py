# SPDX-License-Identifier: Apache-2.0
"""Expand one frozen BS16 Q4 donor, preserving scores and cache state."""

from dataclasses import fields
import hashlib
import json
from pathlib import Path

import numpy as np

from workloads import TOPK, TOKEN_BYTES, CaptureReference, ReplayRound, ReplaySequence


BATCHES = (20, 24, 28, 32)
FIXED_BATCHES = (1, 2, 4, 8, 16, *BATCHES)
GROUPS = (10, 4, 17, 7, 2)
SYNTHETIC_GROUP = 0


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


def load_synthetic(fixture, batch):
    """Use the compact synthetic QK operands with private DDR pages for every request."""
    manifest = fixture / "fixture.json"
    config = json.loads(manifest.read_text())
    full_chain = config["full_chain_fixture"]
    history, resident, heads = config["history"], config["resident"], full_chain["index_heads"]
    if (
        full_chain["model"] != "compact_builtin_v1"
        or config["queries"] != 4
        or config["storage"] != history
        or config["visible_lengths"] != [history] * 4
        or (history, resident, heads) != (131072, 16384, 32)
    ):
        raise ValueError("Synthetic packed input requires a compact H128K/R16K full-prefix Q4 fixture with 32 heads")

    order = np.load(fixture / "physical_to_logical.npy", allow_pickle=False)
    scores = np.load(fixture / "logical_scores.npy", allow_pickle=False)
    if (
        scores.shape != (4, history)
        or scores.dtype != np.float32
        or not np.isfinite(scores).all()
        or order.dtype.kind not in "iu"
        or not np.array_equal(np.sort(order), np.arange(history))
    ):
        raise ValueError("Invalid compact synthetic score oracle or token permutation")
    # Treat the physical fixture positions as the logical IDs of this workload,
    # as the existing single-call synthetic benchmark does.
    bf16 = (scores[:, order].view(np.uint32) >> 16).astype(np.uint16)
    scores = np.where(bf16 & 0x8000, ~bf16, bf16 ^ 0x8000)
    keys = np.fromfile(fixture / "arg1.bin", np.uint8).reshape(history, 128)
    scales = np.fromfile(fixture / "arg4.bin", np.float32).reshape(history)
    query = np.fromfile(fixture / "arg0.bin", np.uint8).reshape(4, heads, 128)
    weights = (np.fromfile(fixture / "arg2.bin", np.float32).view(np.uint32) >> 16).astype(np.uint16)
    query_scale = np.fromfile(fixture / "arg3.bin", np.float32).reshape(4, heads)

    # Begin in ascending slot order and promote the touched slots in slot
    # order, matching SlotLRU. Protect all cutoff ties so the chosen victims
    # do not depend on which tied TopK tokens a particular arm returns.
    cutoffs = np.partition(scores, -TOPK, axis=1)[:, -TOPK, None]
    touched = np.any(scores[:, :resident] >= cutoffs, axis=0)
    oldest = np.concatenate((np.flatnonzero(~touched), np.flatnonzero(touched))).astype(np.int32)

    def rows(array):
        return np.broadcast_to(array, (16, *array.shape))

    requests = tuple((f"synthetic-{row + 1}", row + 1) for row in range(16))
    ref = CaptureReference(0, 1, 1, SYNTHETIC_GROUP, manifest, requests)
    source = ReplayRound(
        source_round=1, scheduler_round=1,
        identities=tuple((ref.dp_rank, *request) for request in requests),
        geometry=dict(batch=16, history=history, resident=resident, heads=heads,
                      token_bytes=TOKEN_BYTES, output_stride=resident, sparse_mode=0),
        logical_keys=rows(keys),
        logical_scales=rows(scales),
        query=rows(query),
        weights=rows(weights.reshape(4, heads)),
        query_scale=rows(query_scale),
        query_ends=np.full((16, 4), history, dtype=np.int32),
        tail_bounds=np.zeros((16, 2), dtype=np.int32),
        logical_address_valid=rows(np.ones(history, dtype=np.uint8)),
        page_keys=((None,) * (history // 128),) * 16,
        captured_layout=rows(np.arange(history)),
        captured_hot_valid=rows(np.ones(resident, dtype=np.int32)),
        captured_lru=rows(oldest),
        source_paths=(str(manifest),),
        logical_score_keys=rows(scores),
        logical_score_valid=rows(np.ones((4, history), dtype=bool)),
    )
    inputs, donors = expand(source, batch)
    return FrozenRound(inputs, (ref,)), donors


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
