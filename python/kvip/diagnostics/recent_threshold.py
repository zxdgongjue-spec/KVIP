# SPDX-License-Identifier: Apache-2.0
"""Request-aligned recent Index ranges for analysis and native replay inputs.

The previous ABC beginning is a logical boundary. The native contiguous read
also requires those recent logical IDs to remain at their identity physical
Index positions; reject an archive that does not satisfy that contract.
"""

import numpy as np

QUERIES = 4


def request_identities(trace):
    metadata = trace.manifest["metadata"]
    by_owner = {item["owner"]: item for item in metadata["requests"]}
    owners = trace.array("pre.row_owners")
    if len(by_owner) != len(owners) or set(by_owner) != set(map(int, owners)):
        raise ValueError("Request identities do not cover active owners exactly")
    identities = []
    for row, owner in enumerate(owners):
        item = by_owner[int(owner)]
        if item["query_begin"] != row * QUERIES or item["query_stop"] != (row + 1) * QUERIES:
            raise ValueError("Request identity differs from its packed Q4 row")
        identities.append((metadata["dp_rank"], str(item["request_id"]), int(owner)))
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate request identity")
    return identities


def range_state(trace):
    """Keep only the previous boundaries and request lifetime, never Index/KV bytes."""
    source, geometry = trace.manifest["metadata"], trace.manifest["geometry"]
    if source["context"].get("global_decode_confirmed") is not True:
        raise ValueError("Recent Index predecessor must be a confirmed all-decode round")
    identities = request_identities(trace)
    bounds = trace.array("pre.tail_bounds")
    if bounds.dtype != np.int32 or bounds.shape != (len(identities), 2):
        raise ValueError("Previous ABC bounds must be int32[B,2]")
    if np.any(bounds < 0) or np.any(bounds[:, 0] > bounds[:, 1]) or np.any(bounds > geometry["history"]):
        raise ValueError("Previous ABC bounds are outside Index history")
    return dict(
        dp_rank=source["dp_rank"],
        group=source["group"],
        layer_names=tuple(source["layer_names"]),
        scheduler_round=source["context"]["round"],
        requests={identity: tuple(map(int, bounds[row])) for row, identity in enumerate(identities)},
    )


def threshold_bounds(trace, previous):
    """Return full [previous ABC begin,current valid query end), aligned by identity.

    A query clips this common interval at its own query_end. There is no silent
    length cap. Retired ABC entries before the current ABC begin remain included.
    """
    source, g = trace.manifest["metadata"], trace.manifest["geometry"]
    if source["context"].get("global_decode_confirmed") is not True:
        raise ValueError("Recent Index threshold requires a confirmed all-decode round")
    if previous is None or any(
        previous.get(name) != value
        for name, value in (
            ("dp_rank", source["dp_rank"]),
            ("group", source["group"]),
            ("layer_names", tuple(source["layer_names"])),
        )
    ):
        raise ValueError("Recent Index threshold requires the preceding same-DP/group/layers state")
    delta = source["context"]["round"] - previous["scheduler_round"]
    if delta != 1:
        raise ValueError(f"Recent Index predecessor scheduler rounds are not consecutive: delta={delta}")
    identities = request_identities(trace)
    if set(identities) != set(previous["requests"]):
        raise ValueError("Recent Index predecessor request/owner lifetime differs")
    ends = trace.array("pre.query_ends")
    ids = trace.array("pre.logical_ids")
    current = trace.array("pre.tail_bounds")
    if ends.shape != (g["batch"], QUERIES) or ids.shape != (g["batch"], g["history"]):
        raise ValueError("Recent Index threshold requires packed Q4 and physical Index ID rows")
    if np.any(ends <= 0) or np.any(ends > g["history"]) or np.any(np.diff(ends, axis=1) < 0):
        raise ValueError("Recent Index query ends must be ordered valid causal prefixes")
    result = np.empty((g["batch"], 2), dtype=np.int32)
    for row, identity in enumerate(identities):
        begin = previous["requests"][identity][0]
        end = int(ends[row].max())
        if begin > int(current[row, 0]) or begin > int(ends[row].min()):
            raise ValueError("Previous ABC beginning lies after current ABC/query causality")
        if not np.array_equal(ids[row, begin:end], np.arange(begin, end)):
            raise ValueError("Recent logical IDs are not at identity physical Index positions")
        # The full score prefix must have the same logical and physical causal bound.
        maxima = np.maximum.accumulate(ids[row])
        if np.any(maxima[ends[row].astype(np.int64) - 1] >= ends[row]):
            raise ValueError("Recent Index physical score prefix contains future logical IDs")
        result[row] = (begin, end)
    return result


def recent_positions(resident, bounds, query_end):
    """Union without double-counting overlap with the resident physical prefix."""
    begin, end = map(int, bounds)
    prefix = min(resident, query_end)
    return np.concatenate(
        (
            np.arange(prefix, dtype=np.int32),
            np.arange(max(prefix, begin), min(end, query_end), dtype=np.int32),
        )
    )
