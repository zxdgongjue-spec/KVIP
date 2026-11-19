"""Account for work outside the rotating Fetch interval on the same query.

Diagnostic timestamps never replace uninstrumented graph timings. In particular,
maxima on different queries are not additive and window durations are not summed
across concurrently executing cores.
"""

import argparse
from pathlib import Path

import numpy as np

from kvip.diagnostics.ranked_port import kernel_costs, load_case, summarize


def decode(times, events, batch, rate, *, ranked):
    li = np.asarray(times).view(np.uint64).reshape(-1)[64*batch:128*batch].reshape(batch*4, 16)
    if not np.any(li[:, 12:16]):
        return dict(status="unavailable", reason="Native source predates LI phase markers")
    if batch != 16 or not np.isfinite(rate) or rate <= 0:
        raise ValueError("LI phase audit requires the BS16/Q4 split-Q workload and a valid clock")
    ev = np.ascontiguousarray(events).view(np.uint64).reshape(8, 8, 16)
    windows = {}
    for epoch, lane in np.ndindex(8, 8):
        e = ev[epoch, lane]
        core = int(e[2] & np.uint64(0xffffffff))
        query = core*2 + lane % 2
        opened = int(e[4] if ranked else e[13])
        end = int(e[1] if ranked else e[14])
        if query in windows or not 0 <= query < batch*4 or not 0 < opened <= end:
            raise ValueError("Missing, duplicate or invalid per-core Fetch interval")
        windows[query] = dict(epoch=epoch, lane=lane, begin_tick=opened, end_tick=end)
    origin = int(li[:, 12].min())
    rows = []
    for query, ticks in enumerate(li):
        seed0, seed1, final0, rank_ready = map(int, ticks[12:16])
        rank_end, map_end, end = map(int, ticks[[0, 1, 7]])
        if not 0 < seed0 <= seed1 <= final0 <= rank_end <= map_end <= end:
            raise ValueError(f"Missing or reversed LI phase timestamps query={query}")
        if ranked and not final0 <= rank_ready <= rank_end:
            raise ValueError(f"Invalid final predecessor wait query={query}")
        w = windows[query]
        if not seed1 <= w["begin_tick"] <= w["end_tick"] <= final0:
            raise ValueError(f"Fetch interval lies outside this query's cold scan query={query}")
        service = w["end_tick"] - w["begin_tick"]
        parts = dict(seed=seed1-seed0, scan_excluding_own_fetch=final0-seed1-service,
                     own_fetch=service, final_rank_wait=rank_ready-final0 if ranked else 0,
                     final_rank_work=rank_end-(rank_ready if ranked else final0),
                     mapping_wait=map_end-rank_end, classification=end-map_end)
        if sum(parts.values()) != end-seed0 or min(parts.values()) < 0:
            raise ValueError(f"LI query interval does not balance query={query}")
        rows.append(dict(query=query, core=query, **w,
                         seed_begin_us=(seed0-origin)/rate, end_us=(end-origin)/rate,
                         parts_us={k: v/rate for k, v in parts.items()},
                         total_us=(end-seed0)/rate))
    fields = rows[0]["parts_us"]
    stats = {k: dict(summarize([r["parts_us"][k] for r in rows]),
                     max=max(r["parts_us"][k] for r in rows)) for k in fields}
    latest = max(rows, key=lambda r: r["end_us"])
    return dict(status="available", origin_tick=origin, ticks_per_us=rate,
                phases=stats, latest_query=latest, records=rows,
                scope="seed entry through classification exit; excludes initial LI scoring and kernel epilogue; scan includes producer stalls and memory contention; per-query intervals are additive, phase maxima are not")


def report(root):
    kernels = {}
    for name in ("reverse", "ranked"):
        case = load_case(root / f"g7-{name}-2.json")
        samples = case.get("native_diagnostics", {}).get("samples", [])
        if not samples:
            raise ValueError(f"Missing diagnostic replay: {name}")
        print(f"AUDIT_BEGIN G7 {name}")
        cost = kernel_costs(case)
        if cost:
            kernels[name] = cost
            print(f"KERNEL_US {name} init/LI/resolve_update=" + "/".join(f"{v:.2f}" for v in cost)
                  + " separate_profile=1")
        for sample_id, sample in enumerate(samples):
            audit = sample.get("li_phase_audit", {})
            if audit.get("status") != "available":
                raise ValueError(f"Missing native LI audit: {name}: {audit}")
            print(f"SAMPLE={sample_id} seed_is_inside_LI; windows=8; each_AIV_serves_once")
            for field, values in audit["phases"].items():
                print(f"PHASE {field} mean/P50/max_us={values['mean']:.2f}/{values['p50']:.2f}/{values['max']:.2f}")
            row = audit["latest_query"]
            print(f"LATEST_QUERY q={row['query']} own_window={row['epoch']} seed_begin_us={row['seed_begin_us']:.2f} "
                  f"end_us={row['end_us']:.2f} total_us={row['total_us']:.2f} "
                  + " ".join(f"{k}={v:.2f}" for k, v in row["parts_us"].items()))
            if sample.get("records"):
                for epoch in range(8):
                    rows = [r for r in sample["records"] if r["epoch"] == epoch]
                    slow = max(rows, key=lambda r: r["open_to_return"])
                    print(f"WINDOW_MAX epoch={epoch} lane={slow['lane']} total_us={slow['open_to_return']:.2f} "
                          f"rank_us={slow['rank_update']:.2f} wait_us={slow['combined_wait']:.2f}")
        print("AUDIT_END")
    if len(kernels) == 2:
        delta = [v-b for v, b in zip(kernels["ranked"], kernels["reverse"])]
        print("KERNEL_DELTA_US ranked-minus-reverse init/LI/resolve_update="
              + "/".join(f"{v:+.2f}" for v in delta) + " separate_profile=1")
    print("AUDIT_PASS timings=separate_instrumented_replay; no_speedup_claim; raw_per_query_and_window_records_in_JSON")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    report(parser.parse_args().root)
