"""Untimed independent oracle for the last published rolling Kth."""

import numpy as np

from kvip.diagnostics.archive import cpu_array


def reference(scores, ends, resident, bounds, progress, initial, rounds, reverse):
    b, q, history = scores.shape
    expected = np.empty((b, q), dtype=np.uint16)
    exact = np.empty((b, q), dtype=np.uint16)
    triggers = [resident + ((history-resident)//128)*e//(rounds-1)*128 for e in range(rounds)]
    for row in range(b):
        begin = (resident + 2047)//2048*2048
        stop = min(history, int(ends[row].min()))//2048*2048
        for query in range(q):
            ready = int(progress[row, query, 0])
            boundary = max((t for t in triggers if t <= ready), default=resident)
            scanned = np.arange(resident, boundary, dtype=np.int64)
            if reverse:
                mask = (scanned >= begin) & (scanned < stop)
                token = scanned[mask]
                scanned[mask] = begin + stop - 2048 - (token//2048)*2048 + token%2048
            recent = np.arange(max(resident, int(bounds[row, 0])), int(bounds[row, 1]))
            positions = np.unique(np.concatenate((np.arange(resident), recent, scanned)))
            positions = positions[positions < ends[row, query]]
            values = scores[row, query, positions]
            if len(values) < 2048:
                raise ValueError("Rolling oracle requires at least 2048 seed positions")
            key = int(np.partition(values, len(values)-2048)[-2048])
            exact[row, query] = key
            expected[row, query] = min(key, int(initial[row, query])+255)
    return expected, exact


def verify(torch, bundle, archive):
    comparison = bundle.threshold_comparison
    initial = np.array(comparison["initial_keys"], dtype=np.uint16)
    final = np.array(comparison["final_keys"], dtype=np.uint16)
    progress = cpu_array(torch, bundle.rotating.progress).reshape(*initial.shape, 32).view(np.uint32)
    actual = (progress[:, :, 8] & 65535).astype(np.uint16)
    g = archive.manifest["geometry"]
    expected, exact = reference(
        archive.array("post.score_keys"), archive.array("pre.query_ends").reshape(initial.shape),
        g["resident"], archive.array("pre.threshold_bounds"), progress, initial,
        int(bundle.rotating.runtime_shape.rounds), bundle.tuning["reverse_cold"],
    )
    if not np.array_equal(actual, expected):
        row, query = np.argwhere(actual != expected)[0]
        raise AssertionError(f"Rolling Kth differs row={row} query={query}: {actual[row,query]} != {expected[row,query]}")
    if np.any(actual < initial) or np.any(actual > final):
        raise AssertionError("Rolling threshold escaped the seed/final Kth bounds")
    return dict(status="bit_exact", queries=actual.size,
                tightened_queries=int(np.count_nonzero(actual > initial)),
                capped_queries=int(np.count_nonzero(exact.astype(np.int32)-initial > 255)),
                mean_final_gap_keys=float(np.mean(final.astype(np.int32)-actual)),
                last_keys=actual.tolist(), included_in_timing=False)
