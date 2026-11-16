# SPDX-License-Identifier: Apache-2.0
"""Check window accumulation on archived score keys, without importing Torch."""

import argparse
import json
from pathlib import Path

import numpy as np

from kvip.diagnostics.archive import TensorArchive


def audit(scores, ends, resident, bounds, rounds, reverse):
    batch, queries, history = scores.shape
    if rounds < 2 or resident < 2048 or resident % 128 or history % 128:
        raise ValueError("Unsupported rolling geometry")
    cold_tiles = (history-resident)//128
    boundaries = [cold_tiles*epoch//(rounds-1)*128 for epoch in range(rounds)]
    if max(np.diff(boundaries)) > 18432:
        raise ValueError("Window exceeds native 18432-score UB allocation")
    counters = dict(queries=batch*queries, boundaries=0, capped=0,
                    histogram_tile=0, histogram_window=0,
                    cdf_tile=0, cdf_batch=0, cdf_window=0, fast_tiles=0, tiles=0, histogram_first=0)
    all_keys = []
    for row in range(batch):
        scan = np.arange(resident, history, dtype=np.int64)
        reverse_begin = (resident+2047)//2048*2048
        reverse_end = min(history, int(ends[row].min()))//2048*2048
        if reverse:
            mask = (scan >= reverse_begin) & (scan < reverse_end)
            token = scan[mask]
            scan[mask] = reverse_begin+reverse_end-2048-(token//2048)*2048+token%2048
        for query in range(queries):
            end = int(ends[row, query])
            lo, hi = max(resident, int(bounds[row, 0])), min(end, int(bounds[row, 1]))
            hi = max(lo, hi)
            seed = np.concatenate((scores[row, query, :resident], scores[row, query, lo:hi])).astype(np.int32)
            initial = int(np.partition(seed, len(seed)-2048)[-2048])
            seed_bins = 255-np.clip(seed-initial, 0, 255)
            cdf = np.minimum(np.cumsum(np.bincount(seed_bins, minlength=256)), 2048)
            truth = list(seed)
            counters["histogram_tile"] += 2*((len(seed)+127)//128+cold_tiles)
            counters["histogram_window"] += 2*((len(seed)+255)//256)
            counters["histogram_first"] += 2*((len(seed)+255)//256)
            counters["cdf_tile"] += cold_tiles
            counters["tiles"] += cold_tiles
            tiles = scan[::128]
            counters["fast_tiles"] += int(np.count_nonzero(
                (tiles+128 <= end) & ((tiles >= hi) | (tiles+128 <= lo))))
            keys = []
            for start, stop in zip(boundaries, boundaries[1:]):
                positions = scan[start:stop]
                values = scores[row, query, positions].astype(np.int32)
                eligible = (positions < end) & ((positions < lo) | (positions >= hi))
                # Independently reproduce the old unsigned subtraction/mask.
                delta = ((values-initial) & 65535).clip(0, 255)
                old = np.where(eligible & (values >= initial), 255-delta, 255)
                # New Max-before-subtraction path, with clipping only where needed.
                new = 255-np.minimum(np.maximum(values, initial)-initial, 255)
                new[~eligible] = 255
                if not np.array_equal(old, new):
                    raise AssertionError("Encoded score bins changed")
                cdf = np.minimum(cdf+np.cumsum(np.bincount(new, minlength=256)), 2048)
                actual = initial+255-int(np.flatnonzero(cdf >= 2048)[0])
                truth.extend(values[eligible])
                exact = int(np.partition(truth, len(truth)-2048)[-2048])
                expected = min(exact, initial+255)
                if actual != expected:
                    raise AssertionError(f"Window Kth changed row={row} q={query}: {actual} != {expected}")
                if not keys:
                    counters["histogram_first"] += 2*((stop-start+255)//256)
                elif keys[0] > actual:
                    raise AssertionError("Frozen first-window threshold exceeds a later prefix Kth")
                # The compact strict cutoff must retain exactly the same keys,
                # including BF16 ties at the raised Kth boundary.
                cutoff = max(initial, actual-1)
                if not np.array_equal((values > initial) & (values >= actual), values > cutoff):
                    raise AssertionError("Compact admission changed the candidate set")
                keys.append(actual)
                counters["boundaries"] += 1
                counters["capped"] += exact > initial+255
                counters["histogram_window"] += 2*((stop-start+255)//256)
                counters["cdf_batch"] += ((stop-start)//128+7)//8
                counters["cdf_window"] += 1
            all_keys.append(keys)
    counters["keys"] = all_keys
    return counters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    # The native replay verifies the whole archive. This CPU check only needs
    # three small tensors; do not read/hash synthetic KV or real Index again.
    manifest = json.loads((args.snapshot / "manifest.json").read_text())
    if manifest.get("schema_version") not in (1, 2) or manifest.get("status") != "complete":
        raise ValueError("Snapshot is incomplete or has an unsupported schema")
    archive = TensorArchive(args.snapshot, manifest)
    try:
        g = archive.manifest["geometry"]
        scores = archive.array("post.score_keys", verify=True)
        ends = archive.array("pre.query_ends", verify=True).reshape(g["batch"], 4)
        bounds = archive.array("pre.threshold_bounds", verify=True)
        rounds = min(g["requested_windows"], g["batch"]*(2 if 2*g["batch"] <= g["aic_count"] else 1))
        for reverse in (False, True):
            result = audit(scores, ends, g["resident"], bounds, rounds, reverse)
            print(f"CPU_ORACLE {'reverse' if reverse else 'forward'} PASS "
                  f"queries={result['queries']} window_checks={result['boundaries']} capped={result['capped']}", flush=True)
            print(f"CPU_WORK hist={result['histogram_tile']}->{result['histogram_window']} "
                  f"CDF_updates={result['cdf_tile']}->{result['cdf_batch']}->{result['cdf_window']} "
                  f"full_tile_fast_path={100*result['fast_tiles']/result['tiles']:.2f}% "
                  "window_UB=36KiB/AIV counts_only_not_latency", flush=True)
            print(f"CPU_LEAN compact_admission=exact first_window_bound=safe "
                  f"hist_all={result['histogram_window']} hist_first={result['histogram_first']} "
                  f"count_reduction={100*(1-result['histogram_first']/result['histogram_window']):.2f}% "
                  "not_latency_prediction", flush=True)
    finally:
        archive.close()


if __name__ == "__main__":
    main()
