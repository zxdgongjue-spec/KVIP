# SPDX-License-Identifier: Apache-2.0
"""Frozen-input adapter and independent diagnostics for the dynamic TopK port.

Native code is loaded from its pinned checkout, never from this reader. CPU
selection checks are functional proofs, not device timing estimates.
"""

import argparse
import ctypes
import json
from pathlib import Path

import numpy as np

from kvip.diagnostics.archive import cpu_array, load_snapshot


def prefix_oracle(scores, ends, bounds, resident=16384, *, skew=False):
    """Incremental strict-above-Kth selection must retain every required rank."""
    checks = candidates = delayed = 0
    unchanged = boundary = reselect = 0
    history = scores.shape[-1]
    split = False  # Keep the current H131328 split-Q schedule.
    for row, q in np.ndindex(scores.shape[:2]):
        end = int(ends[row, q])
        seed = np.zeros(history, dtype=bool)
        seed[:min(resident, end)] = True
        seed[max(resident, int(bounds[row, 0])):min(end, int(bounds[row, 1]))] = True
        seen = seed.copy()
        values = scores[row, q]
        ids = np.flatnonzero(seed)
        if len(ids) < 2048:
            raise ValueError("The frozen test requires at least 2048 seed scores")
        def native_list(population):
            scores = values[population]
            key = np.partition(scores, len(scores)-2048)[-2048]
            return np.concatenate((population[scores > key], population[scores == key]))[:2048]

        selected = native_list(ids)
        for epoch in range(1, 9):
            cap = resident + (history-resident)//128*epoch//8*128
            stop = min(end, cap)
            ready = seed.copy()
            if skew and epoch != 8:
                # A delayed query keeps its seed and catches up on its own;
                # no whole-Q4 readiness assumption or overwritten seed state.
                stop = 0 if epoch == 1 and q == 3 else min(
                    end, max(resident, cap-128*((row+q)%3)))
                delayed += int(stop < min(end, cap))
            ready[:stop] = True
            fresh = np.flatnonzero(ready & ~seen)
            cutoff = values[selected].min()
            offers = fresh[values[fresh] > cutoff]
            merged = np.concatenate((selected, offers))
            reference = native_list(merged)
            strict = np.count_nonzero(values[selected] > cutoff)
            if not len(offers):
                unchanged += 1
            elif len(offers) < 2048-strict:
                boundary += 1
                selected = np.concatenate((selected[:strict], offers, selected[strict:2048-len(offers)]))
            else:
                reselect += 1
                selected = reference
            if not np.array_equal(selected, reference):
                raise AssertionError(f"Boundary shortcut changed ordered IDs row={row} q={q} epoch={epoch}")
            population = values[ready]
            kth = np.partition(population, len(population) - 2048)[-2048]
            if (len(np.unique(selected)) != 2048 or np.any(values[selected] < kth)
                    or np.count_nonzero(values[selected] > kth) != np.count_nonzero(population > kth)):
                raise AssertionError(f"Incremental rank mismatch row={row} q={q} epoch={epoch}")
            seen = ready
            candidates += len(offers)
            checks += 1
    return dict(checks=checks, candidates=candidates, sequence_split=split, skew=skew, delayed_windows=delayed,
                unchanged=unchanged, boundary_only=boundary, full_rank=reselect, timing="NOT_SIMULATED")


def score_keys(torch, rotating):
    """Locate real LI score rows through the completed native headers."""
    shape = rotating.runtime_shape
    pointers = cpu_array(torch, rotating.ranked_state).view(np.uint64)[:, :, 0]
    workspace = rotating.arguments[12]
    raw = cpu_array(torch, workspace).reshape(-1).view(np.uint16)
    base = workspace.data_ptr()
    result = np.empty((shape.batch, 4, shape.history), dtype=np.uint16)
    for row, q in np.ndindex(pointers.shape):
        delta = int(pointers[row, q]) - base
        if delta < 0 or delta % 2 or delta // 2 + shape.history > len(raw):
            raise ValueError(f"Ranked score pointer outside compute storage: row={row} q={q}")
        result[row, q] = raw[delta // 2:delta // 2 + shape.history]
    return result


def summarize(values):
    return dict(n=len(values), mean=float(np.mean(values)), p50=float(np.median(values))) if values else dict(n=0)


def decode_windows(raw, rounds, lanes, rate, *, split=False):
    events = np.ascontiguousarray(raw).view("<u8").reshape(rounds, lanes, 16)
    buckets = {name: [] for name in ("predecessor_wait", "rank_update", "open_to_first_post", "open_to_return")}
    if split:
        buckets.update(post_wait=[], combined_wait=[], post_work=[])
    records = []
    for epoch in range(rounds):
        for lane in range(lanes):
            event = events[epoch, lane]
            end, opened, first, begin_rank, end_rank = map(int, event[[1, 4, 11, 13, 14]])
            if not opened:
                continue
            if not opened <= begin_rank <= end_rank <= end:
                raise ValueError(f"Invalid ranked window clocks epoch={epoch} lane={lane}")
            intervals = dict(predecessor_wait=begin_rank-opened, rank_update=end_rank-begin_rank,
                             open_to_return=end-opened)
            if split:
                # Ranked ABI 140 stores ordered-post wait cycles in word 8.
                post_wait = int(event[8])
                if post_wait > end-end_rank:
                    raise ValueError("Post wait exceeds time after rank")
                intervals.update(post_wait=post_wait, combined_wait=begin_rank-opened+post_wait,
                                 post_work=end-end_rank-post_wait)
            if first:
                if not end_rank <= first <= end:
                    raise ValueError("First doorbell is outside the completed ranked service")
                intervals["open_to_first_post"] = first-opened
            for name, cycles in intervals.items():
                buckets[name].append(cycles/rate)
            records.append(dict(epoch=epoch, lane=lane, **{k: v/rate for k, v in intervals.items()}))
    if split and len(records) != rounds*lanes:
        raise ValueError(f"Missing ranked window markers: {len(records)} of {rounds*lanes}")
    if not records:
        raise ValueError("No ranked cold-window timestamps were captured")
    return dict(units="us", ticks_per_us=rate, windows={k: summarize(v) for k, v in buckets.items()},
                records=records, scope="Per-lane cold service; rank excludes predecessor wait and Post. First Post is not necessarily useful or completed. Intervals overlap across lanes.")


def window_diagnostics(torch, rotating, stream):
    # Keep calibration after the instrumented replay, outside headline timing.
    raw = cpu_array(torch, rotating.events)
    output = torch.empty(4, dtype=torch.int64, device=rotating.device)
    function = rotating.library.kvip_counter_calibration
    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    function.restype = ctypes.c_uint32
    rates = []
    for _ in range(3):
        start, stop = [torch.npu.Event(enable_timing=True) for _ in range(2)]
        start.record(stream)
        rc = function(output.data_ptr(), stream.npu_stream)
        if rc:
            raise RuntimeError(f"Ranked counter calibration launch failed: {rc}")
        stop.record(stream)
        stop.synchronize()
        clocks = cpu_array(torch, output).view(np.uint64)
        elapsed = start.elapsed_time(stop)*1000
        cycles = int(clocks[1])-int(clocks[0])
        if elapsed <= 0 or cycles < 20_000_000:
            raise ValueError("Invalid ranked counter calibration")
        rates.append(cycles/elapsed)
    if np.ptp(rates)/np.median(rates) > .05:
        raise ValueError(f"Unstable ranked clock calibration: {rates}")
    result = decode_windows(raw, rotating.rounds, rotating.service_lanes, float(np.median(rates)),
                            split=hasattr(rotating, "ranked_offers"))
    headers = cpu_array(torch, rotating.ranked_state).view(np.uint64)
    result["pipeline"] = dict(
        queries=int(headers.shape[0]*headers.shape[1]),
        next_query_dma=int(np.count_nonzero(headers[:, :, 8] & 1)),
        staged_metadata=int(np.count_nonzero(headers[:, :, 8] & 2)),
        lanes=rotating.service_lanes,
    )
    return result


def build(root, revision, build_dir, soc, baseline):
    from kvip.diagnostics.upstream import activate

    activate(root, revision)
    from kvip.runtime.build import build_libraries, kvaf_build_paths

    cann, headers = kvaf_build_paths(build_dir)
    options = [dict(ranked_port=True)]
    if baseline:
        from kvip.diagnostics.scan_options import options_for

        options = [options_for(name) for name in ("forward", "reverse")]
    from kvip.diagnostics.scan_options import verify_library

    for flags in options:
        libraries = build_libraries(cache_dir=build_dir / "lifetch", cann=cann, headers=headers, soc=soc, **flags)
        verify_library(libraries[0], dict(flags, ranked_port=not baseline, cold_only_windows=True))
    print("BUILD_PASS backend=" + ("controls" if baseline else "ranked"), flush=True)


def load_case(path):
    data = json.loads(path.read_text())
    if data.get("status") != "pass":
        raise ValueError(f"Replay failed: {path}")
    if "cases" in data:
        if len(data["cases"]) != 1:
            raise ValueError("One fresh process must own exactly one variant")
        return next(iter(data["cases"].values()))
    return data


def report(root, group):
    names = ("forward", "reverse", "ranked")
    cases = {name: [load_case(root / f"g{group}-{name}-{run}.json") for run in (1, 2)] for name in names}
    first = cases["forward"][0]
    for pair in cases.values():
        for case in pair:
            geometry, tuning = case["geometry"], case["tuning"]
            if (geometry["rounds"], geometry["service_lanes"], tuning["window_limit"], tuning["prefetch_limit"]) != (8, 8, 64, 512):
                raise ValueError("Replay must use 8 Fetch windows x 8 AIVs and budgets 64/512")
            if case["snapshot_sha256"] != first["snapshot_sha256"] or geometry != first["geometry"]:
                raise ValueError("Comparison inputs or budgets differ")
            for field in ("initial_keys", "final_keys"):
                if case["threshold_comparison"][field] != first["threshold_comparison"][field]:
                    raise ValueError("Comparison seed/final thresholds differ")
    print(f"RESULT_BEGIN G{group}")
    print("CHECK full_scores_TopK_KV_Index=PASS seed_final_thresholds=IDENTICAL; BF16-tied IDs may differ")
    print("variant mean_us(first/last) P50_us(first/last) drift_mean/P50% miss% useful/waste/repair/total_KiB")
    kernel_times = {}
    for name, pair in cases.items():
        times = [c["timing"] for c in pair]
        reads = [r for c in pair for r in c["reads_samples"]]
        width = first["geometry"]["token_bytes"] / 1024
        traffic = [np.mean([r["prefetch"]-r["unused_prefetch"] for r in reads])*width]
        traffic += [np.mean([r[k] for r in reads])*width for k in ("unused_prefetch", "final", "transfers")]
        miss = 100*sum(r["cold_demand"] for r in reads)/sum(r["selected_unique"] for r in reads)
        drift = [100*(times[1][k]/times[0][k]-1) for k in ("mean_us", "p50_us")]
        print(f"{name:12} {times[0]['mean_us']:.2f}/{times[1]['mean_us']:.2f} "
              f"{times[0]['p50_us']:.2f}/{times[1]['p50_us']:.2f} {drift[0]:+.2f}/{drift[1]:+.2f} {miss:.4f} "
              + "/".join(f"{x:.1f}" for x in traffic))
        for profile in pair[-1].get("system_profiles", {}).get("passes", []):
            if profile.get("metric") != "none":
                continue
            timeline = profile.get("timeline") or {}
            kernels = [timeline.get(key) for key in ("init", "li", "streaming")]
            if all(stage and stage.get("count") == 1 for stage in kernels):
                kernel_times[name] = [stage["sum_us"] for stage in kernels]
                print(f"KERNEL_US {name} init/LI/resolve_update="
                      + "/".join(f"{value:.2f}" for value in kernel_times[name])
                      + " separate_profile=1")
            else:
                print(f"KERNEL_US {name} unavailable_or_multiple_kernels")
        samples = pair[0].get("native_diagnostics", {}).get("samples", [])
        if name == "ranked" and samples:
            windows = samples[0]["windows"]
            pipeline = samples[0]["pipeline"]
            print(f"PIPELINE next_query_dma={pipeline['next_query_dma']}/{pipeline['queries']} "
                  f"staged_metadata={pipeline['staged_metadata']}/{pipeline['queries']} lanes={pipeline['lanes']}")
            fields = []
            for key in ("rank_update", "predecessor_wait", "post_wait", "combined_wait", "post_work", "open_to_first_post", "open_to_return"):
                if key not in windows:
                    continue
                value = windows[key]
                fields.append(f"{key}={value['mean']:.2f}/{value['p50']:.2f}" if value["n"] else f"{key}=NA")
            print("WINDOW mean/P50_us " + " ".join(fields))
    if "ranked" in kernel_times and "reverse" in kernel_times:
        deltas = [a-b for a, b in zip(kernel_times["ranked"], kernel_times["reverse"])]
        print("KERNEL_DELTA_US ranked-minus-reverse init/LI/resolve_update="
              + "/".join(f"{value:+.2f}" for value in deltas) + " separate_profile=1")
    for name in names[2:]:
        gains = []
        for baseline in names[:2]:
            gain = [100*(1-max(c["timing"][k] for c in cases[name])/min(c["timing"][k] for c in cases[baseline]))
                    for k in ("mean_us", "p50_us")]
            gains.extend(gain)
            print(f"GAIN {name} vs_{baseline} conservative_mean/P50={gain[0]:+.2f}/{gain[1]:+.2f}%")
        print(f"GATE {name} " + ("CANDIDATE" if min(gains) > 1 else "KEEP_BASELINES"))
    print("NOTE gain uses slowest candidate vs fastest control repeats; window markers are separate instrumented replays, not graph deltas")
    print(f"RESULT_END G{group}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    cpu = sub.add_parser("cpu")
    cpu.add_argument("--snapshot", type=Path, required=True)
    prebuild = sub.add_parser("build")
    prebuild.add_argument("--root", type=Path, required=True)
    prebuild.add_argument("--revision", required=True)
    prebuild.add_argument("--build-dir", type=Path, required=True)
    prebuild.add_argument("--soc", required=True)
    prebuild.add_argument("--baseline", action="store_true")
    summary = sub.add_parser("report")
    summary.add_argument("--root", type=Path, required=True)
    summary.add_argument("--group", type=int, required=True)
    args = parser.parse_args()
    if args.action == "build":
        build(args.root, args.revision, args.build_dir, args.soc, args.baseline)
    elif args.action == "report":
        report(args.root, args.group)
    else:
        archive = load_snapshot(args.snapshot)
        for skew in (False, True):
            result = prefix_oracle(archive.array("post.score_keys"), archive.array("pre.query_ends").reshape(-1, 4),
                                   archive.array("pre.threshold_bounds"), archive.manifest["geometry"]["resident"], skew=skew)
            print("CPU_PREFIX_PASS " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
