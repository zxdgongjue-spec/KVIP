# SPDX-License-Identifier: Apache-2.0
"""Replay a service LIFetch operand snapshot on a single NPU."""

import json
from types import SimpleNamespace

import numpy as np

from workloads import ReplayPagePlan, ReplayRound


class ServiceRound(ReplayRound):
    def validate_topk(self, selected):
        # The service snapshot contains operands, with no external score oracle.
        for row, queries in enumerate(selected):
            for q, ids in enumerate(queries):
                if len(np.unique(ids)) != 2048 or not self.logical_score_valid[row, q, ids].all():
                    raise AssertionError(f"Invalid logical TopK for row={row}, q={q}")


def read_round(snapshot):
    with np.load(snapshot, allow_pickle=False) as archive:
        if int(archive["schema_version"]) != 1:
            raise ValueError("Unsupported service snapshot version")
        geometry = json.loads(str(archive["geometry"]))
        arrays = {name: archive[name] for name in archive.files if name not in ("schema_version", "geometry")}

    b, h = geometry["batch"], geometry["history"]
    ends = arrays["query_ends"]
    active = geometry["active_batch"]
    if (not 1 <= active <= b or ends.shape != (b, 4) or np.any(ends[:active] < 2048)
            or np.any(ends[:active] > h) or np.any(ends[active:])):
        raise ValueError("Service snapshot replay requires Q4 active rows with at least 2048 valid tokens")
    layout = arrays["logical_ids"]
    if layout.shape != (b, h) or not np.all(np.sort(layout, axis=1) == np.arange(h)):
        raise ValueError("Invalid captured Index permutation")
    rows = np.arange(b)[:, None]

    def logical(physical):
        result = np.empty_like(physical)
        result[rows, layout] = physical
        return result

    physical_aliases = arrays["page_aliases"]
    aliases = logical(physical_aliases).reshape(b, -1, 128)
    first = aliases.max(axis=2)
    if np.any((aliases >= 0) & (aliases != first[..., None])):
        raise ValueError("Captured token addresses disagree on DDR page identity")
    pages = tuple(tuple(None if slot < 0 else (0, int(slot)) for slot in row) for row in first)
    valid = np.empty((b, 4, h), dtype=bool)
    for q in range(4):
        valid[:, q] = logical(np.arange(h) < ends[:, q, None])
    lru = arrays["lru_slots"].copy()
    lru[active:] = np.arange(lru.shape[1])
    inputs = ServiceRound(
        source_round=1, scheduler_round=1,
        identities=tuple((0, f"row{i}", i + 1) for i in range(b)),
        geometry=geometry,
        logical_keys=logical(arrays["keys"]), logical_scales=logical(arrays["scales"]),
        query=arrays["query"].reshape(b, 4, geometry["heads"], 128),
        weights=arrays["weights"].view(np.uint16).reshape(b, 4, geometry["heads"]),
        query_scale=arrays["query_scale"].reshape(b, 4, geometry["heads"]),
        query_ends=ends, tail_bounds=arrays["tail_bounds"],
        logical_address_valid=logical(physical_aliases >= 0),
        page_keys=pages, captured_layout=layout,
        captured_hot_valid=arrays["hot_valid"], captured_lru=lru,
        source_paths=(str(snapshot),), logical_score_keys=None, logical_score_valid=valid,
    )
    inputs.packed_lru_slots = arrays["lru_slots"]
    inputs.packed_reserved_hot = arrays["reserved_hot"]
    return inputs, arrays["completion"]


class SnapshotSequence:
    def __init__(self, inputs, path):
        self.inputs = inputs
        self.rounds = ((SimpleNamespace(source_round=1, manifest_path=path),),)

    def __iter__(self):
        yield self.inputs


def run_service_snapshot(args):
    from execution import run_path

    inputs, original_status = read_round(args.snapshot)
    config = SimpleNamespace(**{**vars(args), "mode": "bench"}, input="real", level=1,
                             transport=inputs.geometry["transport"],
                             snapshot_aics=inputs.geometry["aics"],
                             intermediate_fetches=inputs.geometry["windows"],
                             prefetch_tokens=inputs.geometry["prefetch_tokens"],
                             prefetch_rank_start=inputs.geometry["prefetch_rank_start"])
    sequence = SnapshotSequence(inputs, args.snapshot)
    print(f"Service completion: {original_status.tolist()}; replay uses deterministic KV payloads", flush=True)
    result = run_path(config, args.output, sequence, ReplayPagePlan(sequence), fused=True)
    print(json.dumps({"correctness": result["correctness"], "timing": result["all_rounds"]}, indent=2))
    return int(result["correctness"] != "PASS")
