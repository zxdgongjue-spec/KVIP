#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Post-hoc diagnostic for an existing P0-A2.3 Active-K OFF/ON ABBA run.

Do not confuse apparent reduction in V9 latency with an optimization effect
when the unrelated native_chain control moves by an even larger amount.
Ratio-of-ratios and excess-latency DiD are sensitivity analyses, NOT exact
corrections for hardware noise or causal proof. Uses existing reports only.
"""
import argparse
import json
import math
from pathlib import Path
import statistics


PAIRS = ((0, 1), (3, 2))
METRICS = ("round_init", "prototype_core", "prototype_with_update", "aiv_cache_update")


def trace_init_p50(report_path):
    path = Path(report_path)
    if not path.is_file():
        return None
    samples = json.loads(path.read_text()).get("core_trace", {}).get("samples", [])
    values = []
    for sample in samples:
        stage = sample.get("accounting", {}).get("stages_us", {})
        init = stage.get("init")
        if isinstance(init, (int, float)) and math.isfinite(init) and init > 0:
            values.append(float(init))
    return statistics.median(values) if values else None


def analyze(report):
    runs = report.get("runs", [])
    if len(runs) != 4 or [r.get("switch") for r in runs] != ["OFF", "ON", "ON", "OFF"]:
        raise ValueError("expected four complete ABBA runs in OFF/ON/ON/OFF order")
    reference = runs[0]
    for r in runs[1:]:
        if r.get("fixture_sha256") != reference.get("fixture_sha256"):
            raise ValueError("fixture hashes must match across all four runs")
        if r.get("native_unique_misses") != reference.get("native_unique_misses"):
            raise ValueError("Native miss counts must match across all four runs")

    results = []
    for pair_id, (off_index, on_index) in enumerate(PAIRS, 1):
        off, on = runs[off_index], runs[on_index]
        a, b = off["metrics_p50_us"], on["metrics_p50_us"]
        native_off, native_on = float(a["native_chain"]), float(b["native_chain"])
        if min(native_off, native_on) <= 0:
            raise ValueError("nonpositive Native baseline")
        drift = 100 * (native_off - native_on) / native_off
        entry = dict(
            pair=pair_id,
            run_indices=[off_index, on_index],
            native_off_us=native_off,
            native_on_us=native_on,
            native_apparent_saving_pct=drift,
            metrics={},
        )
        for name in METRICS:
            off_us, on_us = float(a[name]), float(b[name])
            if min(off_us, on_us) <= 0:
                raise ValueError(f"nonpositive metric {name}")
            raw_saved = off_us - on_us
            metric = dict(
                off_us=off_us, on_us=on_us,
                raw_saved_us=raw_saved,
                raw_saved_pct=100 * raw_saved / off_us,
                native_control_drift_pct=drift,
            )
            if name in ("prototype_core", "prototype_with_update"):
                # Alternative (not causally privileged) noise adjustments.
                off_norm = off_us / native_off
                on_norm = on_us / native_on
                metric.update(
                    ratio_of_ratios_saved_pct=100 * (1 - on_norm / off_norm),
                    excess_latency_saved_us=(off_us - native_off) - (on_us - native_on),
                )
            entry["metrics"][name] = metric
        entry["device_core_trace_init_off_us"] = trace_init_p50(off.get("report_path", ""))
        entry["device_core_trace_init_on_us"] = trace_init_p50(on.get("report_path", ""))
        results.append(entry)

    aggregate = {}
    for name in METRICS:
        vals = [p["metrics"][name] for p in results]
        agg = {
            "raw_saved_us": statistics.median(v["raw_saved_us"] for v in vals),
            "raw_saved_pct": statistics.median(v["raw_saved_pct"] for v in vals),
            "both_raw_pairs_positive": all(v["raw_saved_us"] > 0 for v in vals),
        }
        if name in ("prototype_core", "prototype_with_update"):
            agg.update(
                native_normalized_saved_pct=statistics.median(
                    v["ratio_of_ratios_saved_pct"] for v in vals
                ),
                excess_latency_saved_us=statistics.median(
                    v["excess_latency_saved_us"] for v in vals
                ),
                both_normalized_pairs_positive=all(
                    v["ratio_of_ratios_saved_pct"] > 0 for v in vals
                ),
            )
        aggregate[name] = agg
    native_drift = [p["native_apparent_saving_pct"] for p in results]
    return {
        "source_order": ["OFF", "ON", "ON", "OFF"],
        "native_control_apparent_saving_pct": native_drift,
        "pairs": results,
        "aggregate": aggregate,
        "interpretation": (
            "If native_chain (not affected by Active-K) drifts by more than "
            "V9's apparent benefit, raw ON/OFF attribution is unreliable. "
            "Native-normalized ratios and excess-latency DiD are diagnostics "
            "only; device trace uses separate profiled replay. Two pairs "
            "do not establish statistical significance."
        ),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    report = json.loads(args.report.read_text())
    result = analyze(report)
    output = args.output or args.report.with_name("abba_attribution.json")
    output.write_text(json.dumps(result, indent=2))
    for item in result["pairs"]:
        print(
            f"PAIR {item['pair']}: Native control apparent saving "
            f"{item['native_apparent_saving_pct']:+.3f}%",
            flush=True,
        )
        for name, v in item["metrics"].items():
            more = ""
            if "ratio_of_ratios_saved_pct" in v:
                more = (
                    f", native-normalized={v['ratio_of_ratios_saved_pct']:+.3f}%"
                    f", excess-latency-saved={v['excess_latency_saved_us']:+.3f} us"
                )
            print(f"  {name}: raw={v['raw_saved_pct']:+.3f}%{more}")
        print(
            "  profiled core init device-us: "
            f"OFF={item['device_core_trace_init_off_us']} "
            f"ON={item['device_core_trace_init_on_us']}"
        )
    print("ATTRIBUTION_REPORT", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
