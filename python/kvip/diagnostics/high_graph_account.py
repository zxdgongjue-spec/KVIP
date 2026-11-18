# SPDX-License-Identifier: Apache-2.0
"""One event and one exported kernel timeline from the very same HIGH replay."""
import json
import math
from decimal import Decimal
from pathlib import Path

from kvip.diagnostics.graph_accounting import timeline
from kvip.diagnostics.read_through_report import require


def print_account(arm, case, headline):
    for key in ("kvip_commit", "library_sha256", "libraries", "snapshot_sha256", "geometry", "selection_comparison",
                "graph_backend", "tuning", "high_post_index", "device", "kvaf_build", "soc", "diagnostics_commit"):
        require(case.get(key) is not None and case[key] == headline[key], f"{arm}: Level0 mismatch in {key}")
    (profile,) = case["system_profiles"]["passes"]
    exp = profile["experiment"]
    route = exp["auto_route"]
    require(profile["correctness"] == "passed" and profile["graph_replays"] == 1 and profile["metric"] == "none"
            and profile["included_in_headline_timing"] is False and exp["snapshot_sha256"] == case["snapshot_sha256"]
            and exp["production_source"]["commit"] == case["kvip_commit"] and exp["libraries"] == case["libraries"]
            and route["mode"] == "high" and route["full_aiv"] == (arm == "after") == case["high_post_index"]
            and route["ordered_index_verified"], f"{arm}: invalid source, graph or HIGH oracle")
    elapsed = profile["graph_event_us"]
    require(math.isfinite(elapsed) and elapsed > 0, f"{arm}: missing same-replay Event")
    (trace,) = Path(profile["directory"]).rglob("trace_view.json")
    with trace.open() as src:
        doc = json.load(src, parse_float=Decimal)
    values, durations = timeline(doc.get("traceEvents", []) if isinstance(doc, dict) else doc, elapsed)
    totals = []
    for sig in ("v9_init_kernel", "v9_li_kernel", "v9_streaming_resolve_update"):
        found = [us for name, us in durations.items() if sig in name]
        require(len(found) == 1, f"{arm}: expected one {sig} kernel name")
        totals.append(found[0])
    other = sum(durations.values()) - sum(totals)
    print(f"G7 KERNEL_ACCOUNT {arm} profiled_replays=1 units=us event={elapsed:.3f} span={values['kernel_span']:.3f}"
          f" init/LI/streaming={totals[0]:.3f}/{totals[1]:.3f}/{totals[2]:.3f} other={other:.3f}"
          f" gap/overlap={values['kernel_gap']:.3f}/{values['kernel_overlap']:.3f}"
          f" event_minus_span={values['outside_kernel']:.3f} prefetch/final={route['prefetch']}/{route['final_get']}",
          flush=True)
