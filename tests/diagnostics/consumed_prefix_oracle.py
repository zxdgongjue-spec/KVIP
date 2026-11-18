"""Replay exact Q4 posting order for frozen or independently lagging publications."""
import sys
from pathlib import Path

import numpy as np
from kvip.diagnostics.incremental_topk import scan_windows


def publications(data, b):
    ids, scores, keys = (data[k][b] for k in ("expected_ids", "expected_scores", "expected_thresholds"))
    eligible = data["eligible"][b].astype(bool)
    history = data["scores"][b]
    ends = data["query_ends"][b]
    lo, hi = data["threshold_bounds"][b]
    windows = scan_windows(history.shape[-1], 16384, ends, True)
    result = []
    for q, epochs in enumerate(ids):
        records = []
        for e, rank in enumerate(epochs):
            selected = rank[eligible[rank]].tolist()
            if not e or keys[q,e] != keys[q,e-1]:
                base = suffix = e
                head = rank[scores[q,e] > keys[q,e]]
                fixed = len(head[eligible[head]]) // 8 * 8
                prefix = 0
            else:
                w = windows[e-1]
                fresh = (w < ends[q]) & ((w < lo) | (w >= hi)) & (history[q,w] > keys[q,e-1])
                if np.any(fresh):
                    prefix, suffix = fixed, e
            if prefix and selected[:prefix] != records[base][0][:prefix]:
                raise AssertionError("Prefix identity mismatch")
            if records and records[-1][1:] == (base, suffix, prefix) and records[-1][0] != selected:
                raise AssertionError("Changed payload without a new version")
            records.append((selected, base, suffix, prefix))
        result.append(records)
    return result


def post(qs, ready, cap, skip_prefix):
    posted = set()
    order = []
    cursors = [(None, None, 0) for _ in qs]
    inspected = 0
    for versions in ready:
        room = min(cap, 512 - len(posted))
        batch = []
        for q, version in enumerate(versions):
            if room == 0 or version < 0:
                continue
            ids, base, suffix, prefix = qs[q][version]
            start = 0
            old_base, old_suffix, count = cursors[q]
            if skip_prefix and old_base == base:
                if old_suffix == suffix and count == len(ids):
                    continue
                start = (count if old_suffix == suffix else min(count, prefix)) // 8 * 8
            if not all(t in posted for t in ids[:start]):
                raise AssertionError("Skipped an ID still needing a post")
            inspected += len(ids) - start
            missing = [t for t in ids[start:] if t not in posted]
            chosen = missing[:room]
            proved = len(ids) if len(chosen) == len(missing) else start + len(chosen)
            cursors[q] = (base, suffix, proved)
            batch.extend(chosen)
            posted.update(chosen)
            room -= len(chosen)
        order.append(batch)
    need = {t for qs_q in qs for t in qs_q[-1][0]}
    return order, np.array([inspected, len(posted & need), len(posted - need), len(need - posted)])


def main(folder):
    rng = np.random.default_rng(141623820)
    for g in (7, 2, 10, 17, 4):
        with np.load(folder / f"g{g}/probe_inputs.npz") as archive:
            data = dict(archive)
        totals = {cap: np.zeros(5, dtype=np.int64) for cap in (64,128,480,512)}
        for b in range(16):
            qs = publications(data,b)
            full = np.broadcast_to(np.arange(9)[:,None], (9,4))
            lag = np.maximum.accumulate(np.maximum(-1,full-rng.integers(0,4,(9,4))),axis=0)
            for cap, stats in totals.items():
                for schedule in (full, lag):
                    expected, ref = post(qs, schedule, cap, False)
                    actual, opt = post(qs, schedule, cap, True)
                    if expected != actual:
                        raise AssertionError((g,b,cap))
                    if schedule is full:
                        stats += [ref[0], *opt]
        for cap,(old,new,useful,waste,repair) in totals.items():
            print(f"G{g} cap={cap} inspected={old}->{new} useful={useful} waste={waste} "
                  f"repair={repair} total_KiB={(useful+waste+repair)*656/1024:.2f}")
    print("PASS Q4 posting order including independently delayed publications")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
