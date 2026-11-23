"""Rebuild complete compacted window inputs from the saved score tensors."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

K = 2048
TILE = 128


def physical_order(history, resident, ends):
    order = np.arange(history, dtype=np.uint32)
    lo = (resident + 2047) // 2048 * 2048
    hi = int(min(ends)) // 2048 * 2048
    cold = order[lo:hi].copy()
    order[lo:hi] = lo + hi - 2048 - (cold // 2048 * 2048) + cold % 2048
    assert np.array_equal(order[order], np.arange(history))
    return order


def full_select(scores, ids):
    cut = int(np.partition(scores, -K)[-K])
    pos = np.concatenate((np.flatnonzero(scores > cut), np.flatnonzero(scores == cut)))[:K]
    return scores[pos], ids[pos], cut


def verified_select(old, adds, old_cut):
    n = len(adds)
    count_ties = int(np.count_nonzero(old == old_cut))
    assert np.all(old[: K - count_ties] > old_cut) and np.all(old[K - count_ties :] == old_cut)
    combined = np.concatenate((old, adds))
    if not n:
        return np.arange(K), "empty"
    if old_cut and n < count_ties:
        return np.r_[
            np.arange(K - count_ties), np.arange(K, K + n), np.arange(K - count_ties, K - n)
        ], "unchanged"
    if old_cut and old_cut < 65535 and n < K:
        prospective = old_cut + 1
        ge = np.flatnonzero(combined > prospective)
        eq = np.flatnonzero(combined == prospective)
        if len(ge) < K <= len(ge) + len(eq):
            return np.concatenate((ge, eq))[:K], "step"
    _, pos, _ = full_select(combined, np.arange(len(combined)))
    return pos, "fallback"


def write_case(old_ids, old_scores, phys, adds, old_cut, final_ids, final_scores):
    if not 0 < len(adds) < K:
        raise ValueError("Component fixture quota exceeded")
    base = int(min(phys))
    offsets = (phys - base).astype("<u2")
    assert np.array_equal(base + offsets.astype("<u4"), phys)
    record = np.zeros(5128, "<u4")
    record[:3] = len(adds), base, old_cut
    record[8:2056] = old_ids
    record[2056:3080] = old_scores.astype("<u2").view("<u4")
    record[3080:4104].view("<u2")[: len(adds)] = offsets
    record[4104:].view("<u2")[: len(adds)] = adds
    want = np.zeros(3080, "<u4")
    want[0] = K
    want[8:2056] = final_ids
    want[2056:] = final_scores.astype("<u2").view("<u4")
    return record, want


def tag(ids, order, eligible, tail):
    phys = order[ids].copy()
    missing = ~eligible[ids]
    recent = (ids >= tail[0]) & (ids < tail[1])
    phys[missing & recent] |= np.uint32(0x40000000)
    phys[missing & ~recent] |= np.uint32(0x80000000)
    return phys


def group(archive, group, rows, reps):
    src = archive / f"g{group}" / "probe_inputs.npz"
    with np.load(src) as z:
        s, ends, bounds, tail, eligible, old_id, old_v, cuts = [
            z[k]
            for k in (
                "scores",
                "query_ends",
                "threshold_bounds",
                "tail_bounds",
                "eligible",
                "expected_ids",
                "expected_scores",
                "expected_thresholds",
            )
        ]
    geom = json.loads((src.parent / "manifest.json").read_text())["geometry"]
    resident, history = geom["resident"], geom["history"]
    windows = [(history - resident) // TILE * w // 8 * TILE + resident for w in range(9)]
    for b in range(len(s)):
        order = physical_order(history, resident, ends[b])
        for q, end in enumerate(ends[b]):
            key = s[b, q]
            seed = np.r_[np.arange(resident), np.arange(max(resident, bounds[b, 0]), min(end, bounds[b, 1]))]
            seed_s, seed_i, seed_c = full_select(key[seed], seed)
            assert (
                np.array_equal(seed_i, old_id[b, q, 0])
                and np.array_equal(seed_s, old_v[b, q, 0])
                and seed_c == cuts[b, q, 0]
            )
            intervals = []
            for first, last in zip(windows, windows[1:]):
                chunk = order[first:last]
                ids = chunk[(chunk < end) & ((chunk < bounds[b, 0]) | (chunk >= bounds[b, 1]))]
                intervals.append(ids)
            # Verify all eight source steps before combining adjacent steps as a four-window fixture.
            for w, ids in enumerate(intervals):
                adds = ids[key[ids] > cuts[b, q, w]]
                check_s, check_i, check_c = full_select(
                    np.r_[old_v[b, q, w], key[adds]], np.r_[old_id[b, q, w], adds]
                )
                assert (
                    np.array_equal(check_s, old_v[b, q, w + 1])
                    and np.array_equal(check_i, old_id[b, q, w + 1])
                    and check_c == cuts[b, q, w + 1]
                )
            # Keep the candidate state resident across all four updates, with original residency tags.
            prior_id = tag(seed_i, order, eligible[b], tail[b])
            prior_s = seed_s
            for w in range(0, 8, 2):
                raw = np.concatenate(intervals[w : w + 2])
                cut = int(cuts[b, q, w])
                adds = raw[key[raw] > cut]
                phys = order[adds]
                assert np.all(eligible[b, adds]) and np.all(phys[:-1] < phys[1:])
                combined_s = np.r_[prior_s, key[adds]]
                combined_i = np.r_[prior_id, phys]
                next_s, next_i, new_cut = full_select(combined_s, combined_i)
                archived_i = tag(old_id[b, q, w + 2], order, eligible[b], tail[b])
                assert np.array_equal(archived_i, next_i) and np.array_equal(old_v[b, q, w + 2], next_s)
                assert new_cut == cuts[b, q, w + 2]
                ix, branch = verified_select(prior_s, key[adds], cut)
                assert np.array_equal(combined_s[ix], next_s) and np.array_equal(combined_i[ix], next_i)
                lost = int(len(adds) - np.count_nonzero(ix >= K))
                row = dict(
                    group=group,
                    row=b,
                    q=q,
                    window=w // 2,
                    old_cut=cut,
                    cut=new_cut,
                    delta=new_cut - cut,
                    raw=len(raw),
                    added=len(adds),
                    losers=lost,
                    old_ties=int(np.count_nonzero(prior_s == cut)),
                    ties_at_next=int(np.count_nonzero(combined_s.astype(np.uint32) == cut + 1)),
                    branch=branch,
                )
                rows.append(row)
                if 0 < len(adds) < K:
                    reps[branch].append(
                        (row, write_case(prior_id, prior_s, phys, key[adds], cut, next_i, next_s))
                    )
                prior_id, prior_s = combined_i[ix], combined_s[ix]


def synthetic():
    cut = 1000
    old = np.r_[np.full(K - 4, cut + 3, dtype="<u2"), np.full(4, cut, dtype="<u2")]
    oldids = np.arange(K, dtype="<u4") | np.uint32(0x80000000)
    adds = np.array([cut + 2, cut + 2, cut + 3, cut + 3, cut + 3, cut + 2], dtype="<u2")
    phys = 65000 + np.arange(0, 2 * len(adds), 2, dtype="<u4")
    ns, ni, c = full_select(np.r_[old, adds], np.r_[oldids, phys])
    assert c == cut + 2
    return dict(group=0, branch="synthetic-step-fail", added=len(adds), losers=2, delta=c - cut), write_case(
        oldids, old, phys, adds, cut, ni, ns
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--all-fixtures",
        action="store_true",
        help="Also emit each nonempty single-pack input for the full five-group component replay.",
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    rows = []
    summary = []
    for g in (10, 4, 17, 7, 2):
        reps = defaultdict(list)
        group(args.input, g, rows, reps)
        if args.all_fixtures:
            full = sorted(
                (entry for branch in reps.values() for entry in branch),
                key=lambda entry: tuple(entry[0][k] for k in ("window", "row", "q")),
            )
            np.asarray([v[0] for _, v in full]).tofile(args.output / f"g{g}.bin")
            np.asarray([v[1] for _, v in full]).tofile(args.output / f"g{g}.want")
            (args.output / f"g{g}.json").write_text(json.dumps([r for r, _ in full], indent=2) + "\n")
        group_rows = [r for r in rows if r["group"] == g]
        counts = {
            b: sum(r["branch"] == b for r in group_rows) for b in ("empty", "unchanged", "step", "fallback")
        }
        candidates = [r["added"] for r in group_rows]
        item = dict(
            group=g,
            windows=len(group_rows),
            counts=counts,
            added_min_median_p95_max=[
                np.min(candidates).item(),
                np.median(candidates).item(),
                np.percentile(candidates, 95).item(),
                np.max(candidates).item(),
            ],
            all_added=sum(candidates),
            all_discarded=sum(r["losers"] for r in group_rows),
            rising_steps_with_losing_additions=sum(
                r["branch"] == "step" and r["losers"] > 0 for r in group_rows
            ),
        )
        summary.append(item)
        print(json.dumps(item), flush=True)
        for branch in ("step", "fallback" if reps["fallback"] else "unchanged"):
            pool = sorted(reps[branch], key=lambda a: a[0]["added"])
            if branch == "step" and any(a[0]["losers"] for a in pool):
                pool = [a for a in pool if a[0]["losers"]]
            records.append(pool[len(pool) // 2])
    records.append(synthetic())
    for name, data in (("cases.bin", [v[0] for _, v in records]), ("cases.want", [v[1] for _, v in records])):
        np.asarray(data).tofile(args.output / name)
    for name, data in (
        ("cases.json", [r for r, _ in records]),
        ("all.json", rows),
        ("summary.json", summary),
    ):
        (args.output / name).write_text(json.dumps(data, indent=2) + "\n")
    print("cases", len(records), "~probe-input bytes", (args.output / "cases.bin").stat().st_size)


if __name__ == "__main__":
    main()
