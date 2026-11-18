"""Report verified unprofiled controls separately from device profiles and instrumented ticks."""
import hashlib
import json
import sys
from pathlib import Path

from kvip.diagnostics.prefetch_admission import mean_admission
from kvip.diagnostics.scan_economy_report import control, gain, useful_prefetch
from kvip.diagnostics.scan_options import budgets_for, options_for
from kvip.diagnostics.scan_report import traffic

CANDIDATES = ("reverse-consumed", "reverse-consumed-cap128", "reverse-consumed-cap480")
CONTROLS = ("forward", "reverse")
METRICS = ("mean_us", "p50_us")


def verdict(cases, name):
    timing = cases[name]["timing"]
    margins = []
    for base in CONTROLS:
        best, drift = control(cases, base)
        gains = gain(timing, best)
        margins.extend(gains[k] - abs(drift[k]) for k in METRICS)
        if min(gains.values()) <= 0:
            return "SLOWER"
    if "-cap" in name:
        same_budget = cases["reverse-cap" + name.rsplit("-cap", 1)[1]]["timing"]
        if any(timing[k] >= same_budget[k] for k in METRICS):
            return "BUDGET_ONLY_OR_SLOWER"
    if min(margins) <= 0:
        return "WITHIN_DRIFT"
    return "AHEAD"


def report(path, phase, revision, manifest):
    data = json.loads(path.read_text())
    if data.get("status") != "pass":
        raise ValueError(f"Incomplete {path}")
    cases = data["cases"]
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    for name, case in cases.items():
        tuning = case["tuning"]
        if (case["snapshot_sha256"] != digest or case["production_source"]["commit"] != revision
                or case["geometry"]["rounds"] != 9 or case["geometry"]["token_bytes"] != 656
                or (tuning["prefetch_limit"], tuning["window_limit"]) != budgets_for(name)):
            raise ValueError(f"Source, snapshot, geometry or budget mismatch in {name}")
        if "reverse-native-trunk" in cases:
            expected = options_for(name)
            for flag in ("native_trunk", "trunk_pruned", "seed_parallel"):
                if tuning.get(flag, False) != expected.get(flag, False):
                    raise ValueError(f"{name} did not use the requested {flag} setting")
        if tuning.get("incremental_topk"):
            check = case["incremental_check"]
            if check["status"] != "exact_prefix_ranks_and_pending":
                raise ValueError(f"Unverified prefix selection in {name}")
            reference = next(v for v in cases.values() if v["tuning"].get("incremental_topk"))
            for section, key in (("incremental_check", "prefix_lists_sha256"),
                                 ("selection_comparison", "ordered_output_sha256")):
                if case[section][key] != reference[section][key]:
                    raise ValueError(f"{name} changed {key}")

    if phase == "kernels":
        baseline = None
        for name in data["execution_order"]:
            entry, = [p for p in cases[name]["system_profiles"]["passes"] if p["metric"] == "none"]
            t = {k: entry["timeline"][k]["span_us"] for k in ("device", "init", "li", "streaming")}
            if baseline is None:
                baseline = t
            print(f"KERNEL {name} LI={t['li']:.3f} extra={t['li']-baseline['li']:+.3f} "
                  f"stream={t['streaming']:.3f} saved={baseline['streaming']-t['streaming']:+.3f} "
                  f"init={t['init']:.3f} init_plus_LI_extra={t['li']+t['init']-baseline['li']-baseline['init']:+.3f} "
                  f"span={t['device']:.3f} us")
        return
    if phase == "markers":
        for name in data["execution_order"]:
            a, = cases[name]["native_diagnostics"]["samples"]
            w = a["window_probe"]
            print(f"TICKS {name} seed_max={w['seed_max_ticks']} "
                  f"select/remap/tag/save_meta={w.get('seed_slowest_parts', [])} "
                  f"merge8={w['merge_max_by_cold_window']} publish8={w['publish_max_by_cold_window']}")
            print(f"FETCH {name} wait9={w['wait_max_by_epoch']} candidate9={w['candidate_max_by_epoch']} "
                  f"post9={w['post_max_by_epoch']} service9={w['service_max_by_epoch']} "
                  f"inspected={sum(w['inspected_by_epoch'])}")
        return
    if "reverse-native-trunk" in cases:
        gates = (
            ("reverse-native-trunk", "DENSE_GATE", ("reverse-seed-parallel",)),
            ("reverse-trunk-pruned", "PRUNED_GATE", ("reverse-seed-parallel",)),
        )
    else:
        gates = (
            ("reverse-seed-mask", "SEED_GATE", ("reverse-consumed",)),
            ("reverse-seed-parallel", "PARALLEL_GATE", ("reverse-consumed", "reverse-seed-mask")),
        )
    for repeat, label, other_controls in gates:
        if repeat not in cases:
            continue
        marg = []
        # Slower of the two candidate runs against faster controls per metric;
        # require residual gain above the larger observed repeat drift.
        _, candidate_drift = control(cases, repeat)
        comparisons = []
        references = CONTROLS + other_controls
        if "reverse-cap128-end" in cases:
            references += ("reverse-cap128",)
        for base in references:
            best, drift = control(cases, base)
            for k in METRICS:
                slow = max(cases[n]["timing"][k] for n in (repeat, repeat + "-end"))
                marg.append(100 * (1 - slow / best[k]) - max(abs(drift[k]), abs(candidate_drift[k])))
            comparisons.append(f"{base}={marg[-2]:+.2f}/{marg[-1]:+.2f}")
        gate = "PASS" if min(marg) > 0 else "KEEP_BASELINES"
        print(f"{label} {gate} net_mean/P50_pp={' '.join(comparisons)}")
    for base in CONTROLS:
        best, drift = control(cases, base)
        print(f"CONTROL {base} mean/p50={best['mean_us']:.3f}/{best['p50_us']:.3f} us "
              f"drift={drift['mean_us']:+.2f}/{drift['p50_us']:+.2f}%")
    print("case                       mean/P50 us      vsF mean/P50%   vsR mean/P50%   "
          " useful_pre  waste/repair/total MiB   limited/watched   gate")
    for name in data["execution_order"]:
        if name.endswith("-end"):
            continue
        c = cases[name]
        t, b, a = c["timing"], traffic(c), mean_admission(c["reads_samples"])
        gf, gr = (gain(t, control(cases, n)[0]) for n in CONTROLS)
        print(f"{name:27} {t['mean_us']:7.2f}/{t['p50_us']:7.2f} "
              f"{gf['mean_us']:+7.2f}/{gf['p50_us']:+6.2f} {gr['mean_us']:+7.2f}/{gr['p50_us']:+6.2f} "
              f"{useful_prefetch(c):9.1f} {b['waste']/2**20:.3f}/{b['repair']/2**20:.3f}/{b['total']/2**20:.3f} "
              f"{a['window_limited_epochs']:.1f}/{a['visited_epochs']:.0f} "
              f"{verdict(cases,name) if name in CANDIDATES else '-'}")


if __name__ == "__main__":
    if sys.argv[1] == "--prior":
        root = Path(sys.argv[2])
        for group in (7, 2, 10, 17, 4):
            f = root / f"g{group}-graph.json"
            if not f.is_file():
                print(f"PRIOR G{group} unavailable")
                continue
            cases = json.loads(f.read_text())["cases"]
            for name in ("reverse", "reverse-inc-register", "reverse-inc-segments"):
                a = mean_admission(cases[name]["reads_samples"])
                if a is None:
                    print(f"PRIOR G{group} {name} no_saved_admission_counters")
                    continue
                print(f"PRIOR G{group} {name} window_only={a['window_limited_epochs']:.1f}/"
                      f"{a['visited_epochs']:.0f} total_exhausted={a['budget_exhausted_epochs']:.1f} "
                      f"unused_room={a['remaining_admissible_budget']:.1f}")
    elif sys.argv[1] == "--combined":
        folder = Path(sys.argv[2])
        for name in CANDIDATES:
            groups = {g: verdict(json.loads((folder / f"g{g}-graph.json").read_text())["cases"], name)
                      for g in (2, 7, 10, 17, 4)}
            ok = all(v == "AHEAD" for v in groups.values())
            print("FIVE_GROUP_GATE", name, "PASS" if ok else "KEEP_BASELINES",
                  " ".join(f"G{g}={v}" for g, v in groups.items()))
    else:
        report(Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4]))
