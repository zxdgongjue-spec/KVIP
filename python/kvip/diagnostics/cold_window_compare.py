"""G7 physical AIV scan boundaries. Device clocks come from audit-only binaries."""

import argparse
from pathlib import Path

import numpy as np

from kvip.diagnostics.ranked_port import load_case


def edges(history, resident, *, reference):
    cold = (history - resident) // 128
    if (history, resident) != (131328, 16384):
        raise ValueError("Expected the G7 padded replay geometry")
    if reference:
        half = ((cold + 3) // 4 + 7) // 8 * 2  # 256-aligned tiles per shard
        return [resident // 128 + min(cold, e * 2 * half) for e in range(9)]
    return [resident // 128 + cold * e // 8 for e in range(9)]


def check_prefix(scores, ends, bounds, resident, *, skew):
    """Two independent progress cursors, a common prefix and an exact final."""
    history = scores.shape[-1]
    cuts = np.asarray(edges(history, resident, reference=True)) * 128
    half = (cuts[1] - cuts[0]) // 2
    pos = np.arange(history)
    shard = ((pos - resident) // half) % 2
    checks = 0
    for b, q in np.ndindex(scores.shape[:2]):
        valid = pos < int(ends[b, q])
        seed = valid & ((pos < resident) | ((pos >= bounds[b, 0]) & (pos < bounds[b, 1])))
        seen = seed.copy()
        values = scores[b, q]
        def best(ids):
            return ids[np.lexsort((ids, ~values[ids]))[:2048]]
        state = best(np.flatnonzero(seed))
        counts = np.zeros(history, dtype=np.uint8)
        for e, (lo, hi) in enumerate(zip(cuts, cuts[1:])):
            a = min(lo + half, hi)
            counts[lo:a] += 1
            counts[a:hi] += 1
            lag = 128 * ((b + q + e) % 4) if skew and e != 7 else 0
            end0, end1 = max(lo, a - lag), max(a, hi - 2 * lag)
            available = valid & (seed | ((shard == 0) & (pos < end0)) | ((shard == 1) & (pos < end1)))
            fresh = np.flatnonzero(available & ~seen)
            candidates = np.concatenate((state, fresh))
            # One deterministic tie rule isolates geometry/progress from BF16 ties.
            state = best(candidates)
            if not np.array_equal(state, best(np.flatnonzero(available))) or np.any(seen & ~available):
                raise ValueError(f"Lost prefix row={b} q={q} epoch={e}")
            seen = available
            checks += 1
        if not np.array_equal(seen, valid) or np.any(counts[resident:] != 1):
            raise ValueError("Final scan missed or duplicated positions")
    return checks


def decode(raw, rate, mode):
    times = np.asarray(raw, dtype=np.int64).reshape(64, 8, 8)
    if mode not in (1, 2, 3):
        raise ValueError("Unexpected cold-clock binary")
    cuts = edges(131328, 16384, reference=mode != 1)
    half = (cuts[1] - cuts[0]) // 2
    for core in range(64):
        shard = core // 2 % 2
        for epoch, (lo, hi) in enumerate(zip(cuts, cuts[1:])):
            start = lo + shard * half if mode == 3 else lo
            stop = min(hi, start + half) if mode == 3 else hi
            if tuple(times[core, epoch, 4:6]) != (start, stop):
                raise ValueError(f"Physical tile bounds disagree AIV={core} epoch={epoch} mode={mode}")
    begin, end, open_, close = times[:, :, :4].transpose(2, 0, 1)
    served = open_ > 0
    if not (np.isfinite(rate) and rate > 0 and np.all(begin > 0) and np.all(end > begin)
            and np.all(begin[:, 1:] >= end[:, :-1]) and np.all(served.sum(1) == 1)
            and np.all(served.sum(0) == 8) and np.array_equal(served, close > 0)
            and np.all(open_[served] >= end[served]) and np.all(close[served] >= open_[served])
            and np.all((close[:, :-1] <= begin[:, 1:]))):
        raise ValueError("Invalid physical per-AIV cold bounds or 8x8 Fetch ownership")
    scans = (end - begin) / rate
    first = begin[:, 0]
    last = np.maximum(end[:, -1], close[:, -1])
    fetch = (close - open_).sum(1) / rate
    outside = (last - first) / rate - fetch
    gaps = outside - scans.sum(1)
    if (gaps < -1e-7).any():
        raise ValueError("Cold wall accounting does not balance")
    return dict(scan_us=scans.tolist(), outside_us=outside.tolist(), gap_us=gaps.tolist(),
                fetch_us=fetch.tolist(), physical_aivs=64, cold_windows=8, verified_tile_ends=cuts[1:])


def checked(root, name, number, reference):
    c = load_case(root / f"g7-{name}-{number}.json")
    if (c["snapshot_sha256"] != reference["snapshot_sha256"] or c["geometry"] != reference["geometry"]
            or tuple(c["geometry"][k] for k in ("rounds", "service_lanes")) != (8, 8)
            or tuple(c["tuning"][k] for k in ("window_limit", "prefetch_limit")) != (64, 512)
            or c.get("score_comparison", {}).get("status") != "bit_exact"
            or any(c["threshold_comparison"][k] != reference["threshold_comparison"][k]
                   for k in ("initial_keys", "final_keys"))):
        raise ValueError(f"{name}: score, threshold or workload mismatch")
    return c


def graph_report(root):
    reference = load_case(root / "g7-forward-1.json")
    times, halves, phases = {}, {}, {}
    for name in ("forward", "reverse", "ranked"):
        c = checked(root, name, 1, reference)
        v = np.asarray(c["timing"]["samples_us"])
        if v.shape != (10,) or not np.isfinite(v).all() or min(v) <= 0:
            raise ValueError(f"{name}: incomplete timing")
        times[name] = np.array([np.mean(v), np.median(v)])
        halves[name] = np.array([[np.mean(h), np.median(h)] for h in np.array_split(v, 2)])
        drift = 100 * (halves[name][1] / halves[name][0] - 1)
        reads = c["reads_samples"]
        width = c["geometry"]["token_bytes"] / 1024
        traffic = [np.mean([r[k] for r in reads])*width for k in ("unused_prefetch", "final", "transfers")]
        print(f"GRAPH {name} mean/P50_us={times[name][0]:.2f}/{times[name][1]:.2f} "
              f"half_drift_pct={drift[0]:+.2f}/{drift[1]:+.2f} waste/repair/total_KiB="
              + "/".join(f"{x:.1f}" for x in traffic))
        if name == "forward":
            continue
        from kvip.diagnostics.system_profile import _csv_timeline

        prof, = c["system_profiles"]["passes"]
        t = prof.get("timeline") or {}
        n = 4 if name == "ranked" else 1
        if (t.get("li") or {}).get("count") != n:
            t = _csv_timeline(Path(prof["directory"]), prof["artifacts"])
        if ((t.get("li") or {}).get("count") == n and (t.get("streaming") or {}).get("count") == 1):
            phases[name] = np.array([t["li"]["span_us"], t["streaming"]["span_us"]])
    if len(phases) == 2:
        delta = phases["ranked"] - phases["reverse"]
        print(f"KERNEL_DELTA_US vs_reverse LI_span={delta[0]:+.2f} resolve_update={delta[1]:+.2f}")
    else:
        print("KERNEL_DELTA_UNAVAILABLE see saved profiler tables")
    passed = True
    for base in ("forward", "reverse"):
        delta = times[base] - times["ranked"]
        conservative = halves[base].min(axis=0) - halves["ranked"].max(axis=0)
        print(f"GRAPH_SAVED_US vs_{base} mean/P50={delta[0]:+.2f}/{delta[1]:+.2f} "
              f"worst_half_mean/P50={conservative[0]:+.2f}/{conservative[1]:+.2f}")
        passed &= bool((delta > 0).all() and (conservative > 0).all())
    print("GATE " + ("BOTH_BASELINES_BEATEN" if passed else "KEEP_BASELINES"))


def detail_report(root):
    reference = load_case(root / "g7-ranked-1.json")
    outside, scans, gaps, legacy = {}, {}, {}, {}
    for name in ("reverse-original", "rank-original", "rank-matched", "rank-split"):
        c = checked(root, name, 3, reference)
        if c["kvip_commit"] != reference["kvip_commit"]:
            raise ValueError(f"{name}: diagnostics use a different source")
        sample, = c["native_diagnostics"]["samples"]
        d = sample["cold_window_clock"]
        scans[name] = np.mean(d["scan_us"], axis=0)
        outside[name] = float(np.mean(d["outside_us"]))
        gaps[name] = float(np.mean(d["gap_us"]))
        if name.endswith("-original"):
            legacy[name] = sample["li_phase_audit"]["phases"]["scan_excluding_own_fetch"]["mean"]
    old = outside["rank-original"] - outside["reverse-original"]
    left = outside["rank-split"] - outside["reverse-original"]
    print(f"SCAN_GAP_US legacy_this_run={legacy['rank-original']-legacy['reverse-original']:+.2f} "
          f"common_before={old:+.2f} common_after={left:+.2f} recovered={old-left:+.2f}")
    print(f"SCAN_SAVED_US change_window_edges={outside['rank-original']-outside['rank-matched']:+.2f} "
          f"change_shard_schedule_at_same_edges={outside['rank-matched']-outside['rank-split']:+.2f}")
    print("MATCHED_SCAN_SAVED_US w0..7="
          + "/".join(f"{x:+.2f}" for x in scans["rank-matched"]-scans["rank-split"])
          + f" inter_window={gaps['rank-matched']-gaps['rank-split']:+.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    arg = parser.add_mutually_exclusive_group(required=True)
    arg.add_argument("--graph", type=Path)
    arg.add_argument("--detail", type=Path)
    arg.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    if args.graph:
        graph_report(args.graph)
    elif args.detail:
        detail_report(args.detail)
    else:
        from kvip.diagnostics.archive import load_snapshot
        a = load_snapshot(args.snapshot)
        n = [check_prefix(a.array("post.score_keys"), a.array("pre.query_ends").reshape(-1, 4),
                          a.array("pre.threshold_bounds"), 16384, skew=s) for s in (False, True)]
        print(f"CPU_G7 exact_prefixes regular/skew={n[0]}/{n[1]} cold_tiles=898")
        print("COLD_EDGES absolute_token_exclusive original=" + ",".join(str(x*128) for x in edges(131328, 16384, reference=False)[1:])
              + " reference=" + ",".join(str(x*128) for x in edges(131328, 16384, reference=True)[1:]))


if __name__ == "__main__":
    main()
