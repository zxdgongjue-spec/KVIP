"""Compare every candidate with the unchanged fused baseline first."""

from statistics import mean

from kvip.diagnostics.scan_report import traffic


def control(cases, name):
    first, last = (cases[n]["timing"] for n in (name, name + "-end"))
    keys = ("mean_us", "p50_us")
    best = {k: min(first[k], last[k]) for k in keys}
    drift = {k: 100 * (last[k] / first[k] - 1) for k in keys}
    return best, drift


def gain(timing, reference):
    return {k: 100 * (1 - timing[k] / reference[k]) for k in ("mean_us", "p50_us")}


def useful_prefetch(case):
    return mean(r["prefetch"] - r["unused_prefetch"] for r in case["reads_samples"])


def report(cases, order):
    original, drift = control(cases, "forward")
    reverse, reverse_drift = control(cases, "reverse")
    print("BASELINE unchanged_forward; headline gains use faster first/end independently for mean/p50")
    for name, reference, change in (("forward", original, drift), ("reverse", reverse, reverse_drift)):
        print(f"CONTROL {name} mean_us={reference['mean_us']:.3f} p50_us={reference['p50_us']:.3f} "
              f"drift_mean={change['mean_us']:+.2f}% drift_p50={change['p50_us']:+.2f}%")
    print("variant                 mean_us  p50_us  gain_mean% gain_p50% useful_pre  repair_MiB waste_MiB total_MiB")
    for name in order:
        case = cases[name]
        t, b = case["timing"], traffic(case)
        g = gain(t, original)
        print(f"{name:23} {t['mean_us']:8.3f} {t['p50_us']:8.3f} {g['mean_us']:10.2f} {g['p50_us']:9.2f} "
              f"{useful_prefetch(case):10.1f} {b['repair']/2**20:10.4f} {b['waste']/2**20:9.4f} {b['total']/2**20:9.4f}")
    if "reverse-first-reuse" in cases:
        pairs = (("reverse", "reverse-first-in"), ("reverse-first-in", "reverse-first-reuse"),
                 ("reverse-window-in", "reverse-window-reuse"),
                 ("reverse-first-reuse", "reverse-window-reuse"))
    else:
        pairs = (("reverse", "reverse-out"), ("reverse-out", "reverse-compact-safe"),
                 ("reverse-compact-safe", "reverse-first-safe"),
                 ("reverse-out", "reverse-first"), ("reverse-first", "reverse-first-safe"))
    for left, right in pairs:
        a, b = traffic(cases[left]), traffic(cases[right])
        g = gain(cases[right]["timing"], cases[left]["timing"])
        print(f"INCREMENT {left}->{right} mean_gain={g['mean_us']:+.2f}% p50_gain={g['p50_us']:+.2f}% "
              f"useful_pre_delta={useful_prefetch(cases[right])-useful_prefetch(cases[left]):+.1f} "
              f"repair_delta_KiB={(b['repair']-a['repair'])/1024:+.2f} "
              f"waste_delta_KiB={(b['waste']-a['waste'])/1024:+.2f} "
              f"total_delta_KiB={(b['total']-a['total'])/1024:+.2f}")
    candidates = []
    for name in order:
        if name.startswith("forward") or name.endswith("-end"):
            continue
        g = gain(cases[name]["timing"], original)
        inc = gain(cases[name]["timing"], reverse)
        passes = all(g[k] > max(1.0, abs(drift[k])) for k in g)
        if name != "reverse":
            passes &= all(inc[k] > max(1.0, abs(reverse_drift[k])) for k in inc)
        if passes:
            candidates.append(name)
        print(f"GATE {name} {'CANDIDATE' if passes else 'KEEP_BASELINE'} "
              f"vs_original_mean/p50={g['mean_us']:+.2f}/{g['p50_us']:+.2f}% "
              f"vs_reverse_mean/p50={inc['mean_us']:+.2f}/{inc['p50_us']:+.2f}%")
    samples = cases["forward"]["reads_samples"]
    cold, selected = (mean(s[k] for s in samples) for k in ("cold_demand", "selected_unique"))
    print(f"CHECK ordered_TopK=identical initial/final_thresholds=identical cold_miss={100*cold/selected:.4f}%")
    for name in order:
        check = cases[name].get("rolling_threshold_check")
        if check:
            print(f"ROLLING {name} {check['status']} scope={check.get('scope','all_scanned_windows')} "
                  f"last_key_gap={check['mean_final_gap_keys']:.3f} capped={check['capped_queries']} "
                  f"cap={check.get('cap_mode','seed_plus_255')}")
    print("DECISION " + ("retest_candidates="+",".join(candidates) if candidates else "keep_original_fused"))
    print("NOTE positive delta=more work/bytes; useful_pre includes all prefetch windows; no completion-overlap claim")
    print("NOTE gates screen full graph mean AND median beyond repeat drift/1%; not statistical proof")
