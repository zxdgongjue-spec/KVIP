# SPDX-License-Identifier: Apache-2.0
"""Cross-run Index equivalence once the original complete ordered-Index check passes."""
import hashlib
from itertools import permutations

import numpy as np


def row_assignment(src, dst):
    """Canonical destination←original-source permutation, including displaced hot entries."""
    src, dst = np.asarray(src, dtype="<i4"), np.asarray(dst, dtype="<i4")
    if src.shape != dst.shape or src.ndim != 1 or len(np.unique(src)) != len(src) or len(np.unique(dst)) != len(dst):
        raise ValueError("Bad source or destination set in verified row journal")
    released = src[~np.isin(src, dst, assume_unique=True)]
    displaced = dst[~np.isin(dst, src, assume_unique=True)]
    pairs = np.column_stack((np.concatenate((dst, released)), np.concatenate((src, displaced)))).astype("<i4")
    # The quoted source and destination sets determine the hot-valid writes too.
    h = hashlib.sha256(pairs[np.argsort(pairs[:, 0])].tobytes())
    h.update(b"\0admitted\0")
    h.update(np.sort(dst).tobytes())
    return h.hexdigest()


def archived_assignments(records, batch):
    """Old manifests may lack Q admission offsets. Only claim a full map when it is uniquely determined."""
    result = []
    for r in range(batch):
        chunks = [records[f"r{r}_w{i}"] for i in range(4)]
        tail = [records[f"r{r}_q{i}"] for i in range(4) if len(records[f"r{r}_q{i}"])]
        permutations_to_try = permutations(tail)
        # When every admitted source is outside the destination set, each pair is
        # an independent swap. The original Q arrival order cannot alter the full map.
        all_jobs = np.concatenate(chunks + tail)
        if not np.intersect1d(all_jobs[:, 0], all_jobs[:, 4]).size:
            permutations_to_try = [tail]
        possible = set()
        for order in permutations_to_try:
            entries = np.concatenate([*chunks, *order])
            possible.add(row_assignment(entries[:, 0], entries[:, 4]))
            if len(possible) > 1:
                break
        result.append(next(iter(possible)) if len(possible) == 1 else None)
    return result


def compare(control, candidate, reconstructed):
    def ordered(c, doc):
        return (c["ordered_match"] and c["actual_window_sha256"] == doc["expected_window_sha256"]
                and c["actual_final_sha256"] == doc["expected_final_sha256"])

    same_counts = []
    same_lists = []
    verified_maps = []
    for a, b, ca, cb in zip(control["timed_journal_checks"], candidate["timed_journal_checks"],
                            control["traffic"], candidate["traffic"], strict=True):
        matches = ordered(a, control) and ordered(b, candidate)
        same_lists.append(matches)
        same_counts.append(all(ca[k] == cb[k] for k in ("row_window_prefetch", "row_query_final")))
        actual = b["index_assignment_by_row"]
        original = a.get("index_assignment_by_row")
        expected = original if original is not None else reconstructed if matches else [None] * len(actual)
        rows = [None if x is None else x == y for x, y in zip(expected, actual, strict=True)]
        verified_maps.append(rows)
    map_unknown = sum(row is None for call in verified_maps for row in call)
    map_failed = sum(row is False for call in verified_maps for row in call)
    passed = (len(same_lists) == 10 and all(same_counts) and all(same_lists) and map_failed == map_unknown == 0)
    delta = None
    if passed:
        delta = {k: control["mean_us"][k] - candidate["mean_us"][k] for k in ("LI", "tail")}
        delta["graph"] = control["graph_mean_us"] - candidate["graph_mean_us"]
    return dict(guarded_pass=passed, same_counts_calls=sum(same_counts), paired_ordered_calls=sum(same_lists),
                checked_map_rows=len(verified_maps) * len(reconstructed) - map_unknown, map_unknown_rows=map_unknown,
                map_mismatch_rows=map_failed, controlled_savings_us=delta)
