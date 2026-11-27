# SPDX-License-Identifier: Apache-2.0
"""Authenticate completed four-rank cells before using an interrupted timing matrix."""
from dataclasses import fields
from types import SimpleNamespace
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def expanded_digest(source, refs, donors):
    # Match large_batch_input.input_digest byte-for-byte without copying a BS32
    # bank. The original concatenated each selected row in exactly this order.
    import numpy as np
    from large_batch_input import FrozenRound
    from workloads import ReplayPagePlan

    geometry = dict(source.geometry, batch=len(donors))
    identities = tuple((i // 16, *source.identities[row]) for i, row in enumerate(donors))
    keys = tuple(tuple(None if key is None else (i // 16, *key) for key in source.page_keys[row])
                 for i, row in enumerate(donors))
    page_input = SimpleNamespace(geometry=geometry, identities=identities, page_keys=keys)
    pages = ReplayPagePlan(FrozenRound(page_input, refs))
    digest = hashlib.sha256(json.dumps((geometry, identities, source.source_round, source.scheduler_round, donors),
                                       sort_keys=True).encode())
    for field in fields(source):
        array = getattr(source, field.name)
        if isinstance(array, np.ndarray):
            shape = (len(donors), *array.shape[1:])
            digest.update(json.dumps((field.name, shape, array.dtype.str)).encode())
            for donor in donors:
                digest.update(memoryview(np.ascontiguousarray(array[donor])).cast("B"))
    indices = pages.page_indices
    digest.update(json.dumps(("page_indices", indices.shape, indices.dtype.str)).encode())
    digest.update(memoryview(indices).cast("B"))
    return digest.hexdigest()


def validate(directory, timed_source, helpers, work, sdk_kernel):
    sys.path.insert(0, str(timed_source / "tests/operators/lifetch/support"))
    from fixed_q4_candidate_only import cell
    from large_batch_input import mapped_source, read_source

    with sdk_kernel.open("rb") as binary:
        sdk_sha = hashlib.file_digest(binary, "sha256").hexdigest()
    previous = None
    last_group = None
    source = None
    for marker in sorted(directory.glob("g*/bs*/block*/fused/placement_verified")):
        match = re.fullmatch(r"g(2|4|7|10|17)/bs(20|24|28|32)/block1/fused/placement_verified",
                             marker.relative_to(directory).as_posix())
        if not match or marker.resolve() != marker:
            raise ValueError(f"Unexpected completed cell {marker}")
        group, batch = map(int, match.groups())
        row = cell(directory, group, batch)
        record = json.loads((marker.parent / "packed_result.json").read_text())
        if record["identity"]["kvaf_get_kernel_sha256"] != sdk_sha:
            raise ValueError(f"KVAF GET binary differs at {marker}")
        if row["ranks"]["ranks"] != [1024, 1568, 1792, 2048]:
            raise ValueError(f"Unexpected ranks at {marker}")
        placement = tuple(row["measured_NUMA_UB"])
        if previous is not None and previous != placement:
            raise ValueError(f"Registered memory placement differs at {marker}")
        previous = placement
        if group != last_group:
            source = None
            trace = subprocess.check_output(["bash", str(helpers / "peer_g2_input.sh"), str(work), str(group)],
                                            text=True).strip()
            trace = Path(trace).resolve()
            print(f"{group}\t{trace}")
            prepared = directory / f"g{group}/donors"
            source, refs = (mapped_source(prepared, trace, group, 1)
                            if (prepared / "source.json").is_file() else read_source(trace, group, 1))
            last_group = group
        fingerprint = expanded_digest(source, refs, record["identity"]["donor_rows"])
        if fingerprint != row["source_sha256"]:
            raise ValueError(f"Current inputs differ from stored ten-call graph at {marker}")
        print(f"RESUME_VERIFIED G{group} BS{batch} calls=10 source_sha256={fingerprint}", file=sys.stderr, flush=True)
    if previous is not None:
        print(f"placement\t{previous[0]} {previous[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("directory", "timed_source", "helpers", "work", "sdk_kernel"):
        parser.add_argument(argument, type=Path)
    args = parser.parse_args()
    validate(args.directory.resolve(), args.timed_source, args.helpers, args.work, args.sdk_kernel)
