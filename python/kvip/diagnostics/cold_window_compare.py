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


def report(root):
    def case(name, repeat):
        return load_case(root / f"g7-{name}-{repeat}.json")
    controls = ("forward", "reverse", "ranked-prior", "ranked")
    first = case("forward", 1)
    def check(c):
        if (c["geometry"] != first["geometry"] or c["snapshot_sha256"] != first["snapshot_sha256"]
                or tuple(c["geometry"][k] for k in ("rounds", "service_lanes")) != (8, 8)
                or tuple(c["tuning"][k] for k in ("window_limit", "prefetch_limit")) != (64, 512)
                or any(c["threshold_comparison"][k] != first["threshold_comparison"][k]
                       for k in ("initial_keys", "final_keys"))):
            raise ValueError("Snapshot, thresholds, geometry or budget mismatch")
    timing = {}
    for name in controls:
        cases = [case(name, r) for r in (1, 2)]
        for c in cases:
            check(c)
        timing[name] = a = np.asarray([[c["timing"][k] for k in ("mean_us", "p50_us")] for c in cases])
        print(f"GRAPH {name} mean/P50_us={a[0,0]:.2f}/{a[0,1]:.2f},{a[1,0]:.2f}/{a[1,1]:.2f} "
              + "drift_pct=" + "/".join(f"{v:+.2f}" for v in 100*(a[1]/a[0]-1)))
        # Count dispatches in the actual replay. The donor path launches
        # hot, seed, cold, final in order, so report gaps as well as durations.
        from kvip.diagnostics.system_profile import _csv_timeline
        profiles = [p for p in cases[-1]["system_profiles"]["passes"] if p["metric"] == "none"]
        if len(profiles) != 1:
            raise ValueError("Expected one profiler pass")
        profile = profiles[0]
        timeline = profile.get("timeline") or {}
        if not timeline.get("kernels"):
            timeline = _csv_timeline(Path(profile["directory"]), profile["artifacts"])
        rows = sorted((r for r in timeline.get("kernels", []) if "v9_li_kernel" in r["name"]),
                      key=lambda r: r["start_us"])
        expected = 4 if name == "ranked" else 1
        if len(rows) != expected:
            raise ValueError(f"{name}: expected {expected} device LI tasks, saw {len(rows)}")
        span = rows[-1]["start_us"] + rows[-1]["duration_us"] - rows[0]["start_us"]
        phases = "hot/seed/cold/final" if expected == 4 else "single"
        times = "/".join(f"{v['duration_us']:.2f}" for v in rows)
        print(f"DEVICE_LI {name} {phases}_us={times} span_us={span:.2f}")
    outside, scans, legacy = {}, {}, {}
    expected_audit_commit = case("ranked", 1)["kvip_commit"]
    for name in ("reverse-original", "forward-original", "rank-original", "rank-matched", "rank-split"):
        c = case(name, 3)
        check(c)
        if c["kvip_commit"] != expected_audit_commit:
            raise ValueError(f"Unexpected diagnostic source for {name}")
        sample = c["native_diagnostics"]["samples"][0]
        d = sample["cold_window_clock"]
        if name in ("rank-original", "reverse-original"):
            legacy[name] = sample["li_phase_audit"]["phases"]["scan_excluding_own_fetch"]["mean"]
        scans[name] = np.mean(d["scan_us"], axis=0)
        outside[name] = float(np.mean(d["outside_us"]))
        print(f"COLD {name} outside_mean/P50_us={outside[name]:.2f}/{np.median(d['outside_us']):.2f}"
              f" inter_window_gap_mean_us={np.mean(d['gap_us']):.2f} scan_mean_us_by_window="
              + "/".join(f"{v:.2f}" for v in scans[name]))
    print("HARDWARE_WINDOW_RANGES_PASS physical_AIVs=64 epochs=8 "
          "rank-matched/rank-split cover identical Q4 positions in every epoch")
    original = outside["rank-original"] - outside["reverse-original"]
    remain = outside["rank-split"] - outside["reverse-original"]
    old_metric = legacy["rank-original"] - legacy["reverse-original"]
    print(f"LEGACY_TO_COMMON historical_old_run_us=+143.62 this_run_old_edges_us={old_metric:+.2f} "
          f"this_run_common_edges_us={original:+.2f} boundary_effect_us={old_metric-original:+.2f}")
    print(f"SCAN_GAP_US original_common={original:+.2f} remaining={remain:+.2f} recovered={original-remain:+.2f}")
    for before, after in (("reverse-original", "forward-original"), ("forward-original", "rank-original"),
                          ("rank-original", "rank-matched"), ("rank-matched", "rank-split")):
        print(f"CHANGE {before}->{after} outside_mean_us={outside[after]-outside[before]:+.2f}"
              + " scan_delta_by_window=" + "/".join(f"{v:+.2f}" for v in scans[after]-scans[before]))
    win = True
    for base in controls[:2]:
        delta = timing[base].min(axis=0) - timing["ranked"].max(axis=0)
        print(f"GRAPH_SAVED_US vs_{base} mean/P50={delta[0]:+.2f}/{delta[1]:+.2f}")
        win &= bool(np.all(delta > 0))
    print("GATE " + ("BOTH_BASELINES_BEATEN" if win else "KEEP_BASELINES"))
    print("SCOPE cold clocks by physical AIV, first cold tile to last cold tile/Fetch return; "
          "graph uses independent uninstrumented repeats including all LI launches")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    arg = parser.add_mutually_exclusive_group(required=True)
    arg.add_argument("--root", type=Path)
    arg.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    if args.root:
        report(args.root)
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
