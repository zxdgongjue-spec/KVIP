# SPDX-License-Identifier: Apache-2.0
"""Account for the G2 graph against strong serial using matched Level0 replays."""

import argparse
import json
import math
import re
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from statistics import mean, median

from kvip.diagnostics.post_index_order_report import _low_traffic, _validate
from kvip.diagnostics.read_through_report import load_case, require


ROLES = ("serial", "before", "after")
EDGE_KEYS = ("event_to_execute", "execute_to_kernel", "kernel_to_join", "join_to_event", "event_alignment")


def timestamp(event):
    # Profiler timestamps are epoch microseconds, often encoded as strings.
    return Decimal(str(event["ts"]))


def finish(event):
    return timestamp(event) + Decimal(str(event["dur"]))


def timeline(events, elapsed):
    """Only use device task rows. Analysis tracks duplicate the kernels."""
    tasks = [e for e in events if e.get("ph") == "X" and e.get("args", {}).get("Task Type")]
    launches = [e for e in tasks if e["args"]["Task Type"] == "MODEL_EXECUTE"]
    joins = [e for e in tasks if e["args"]["Task Type"] == "MODEL_WAIT_COMPLETE"]
    kernels = sorted((e for e in tasks if e["args"]["Task Type"].startswith("KERNEL")), key=timestamp)
    require(
        kernels and len({(e["pid"], e["args"].get("Model Id")) for e in kernels}) == 1, "Mixed or missing model kernels"
    )
    first = timestamp(kernels[0])
    last = first
    gaps = Decimal(0)
    kernel_us = defaultdict(Decimal)
    for e in kernels:
        start, end = timestamp(e), finish(e)
        require(end > start, f"Invalid kernel duration for {e['name']}")
        gaps += max(Decimal(0), start - last)
        last = max(last, end)
        kernel_us[e["name"]] += end - start
    span = last - first
    values = dict(
        profile_event=elapsed,
        kernel_span=float(span),
        kernel_gap=float(gaps),
        kernel_overlap=float(sum(kernel_us.values()) + gaps - span),
        outside_kernel=elapsed - float(span),
    )
    if len(launches) == len(joins) == 1:
        launch, join = launches[0], joins[0]
        if (
            (launch["pid"], launch["tid"], launch["args"].get("Model Id"))
            == (join["pid"], join["tid"], join["args"].get("Model Id"))
            and launch["pid"] == kernels[0]["pid"]
            and launch["args"].get("Model Id") == kernels[0]["args"].get("Model Id")
        ):
            execute, joined = timestamp(launch), finish(join)
            values.update(execute_to_kernel=float(first - execute), kernel_to_join=float(joined - last))
            tags = sorted(
                (
                    e
                    for e in tasks
                    if e["args"]["Task Type"].replace("_", " ").upper() == "EVENT RECORD"
                    and e["pid"] == launch["pid"]
                    and e["tid"] == launch["tid"]
                ),
                key=timestamp,
            )
            if len(tags) == 2 and timestamp(tags[0]) <= execute and joined <= finish(tags[1]):
                begin, end = (timestamp(e) for e in tags)
                values.update(
                    event_to_execute=float(execute - begin),
                    join_to_event=float(end - joined),
                    event_alignment=elapsed - float(end - begin),
                )
    return values, {name: float(duration) for name, duration in kernel_us.items()}


def sample(path, role):
    case = load_case(str(path), role)
    if role != "serial":
        low = _validate(case, role)
        require(
            len(low) == len(case["_samples"]) - 1
            and case["auto_route"]["samples"][0]["mode"] == "high"
            and all(not row["fallback"] for row in low),
            f"{role}: expected cold HIGH then learned LOW",
        )
    (profile,) = case["system_profiles"]["passes"]
    exp = profile["experiment"]
    require(
        profile.get("correctness") == "passed"
        and profile.get("graph_replays") == 1
        and profile.get("metric") == "none"
        and profile.get("included_in_headline_timing") is False,
        f"{role}: expected one separate verified Level0 replay",
    )
    require(
        exp["snapshot_sha256"] == case["snapshot_sha256"] and exp["production_source"]["commit"] == case["kvip_commit"],
        f"{role}: trace source/input mismatch",
    )
    if role != "serial":
        route = exp["auto_route"]
        require(
            route["mode"] == "low"
            and not route["fallback"]
            and route["prefetch"] == 0
            and route["final_get"] == case["reads_samples"][-1]["final"],
            f"{role}: wrong trace route/traffic",
        )
    elapsed = profile.get("graph_event_us")
    require(
        isinstance(elapsed, (float, int)) and math.isfinite(elapsed) and elapsed > 0,
        f"{role}: missing graph Event on the measured profiler replay",
    )
    (trace,) = Path(profile["directory"]).rglob("trace_view.json")
    with trace.open() as source:
        doc = json.load(source, parse_float=Decimal)
    values, kernels = timeline(doc.get("traceEvents", []) if isinstance(doc, dict) else doc, elapsed)
    headline = case["_samples"][1:]
    values.update(headline=mean(headline), p50=median(headline), between_samples_unassigned=mean(headline) - elapsed)
    li = [us for k, us in kernels.items() if "lightningindexer" in k.lower() or "v9_li_kernel" in k.lower()]
    require(len(li) == 1, "Missing or ambiguous LI kernel")
    values["li"] = li[0]
    values["init"] = sum(us for k, us in kernels.items() if "v9_init_kernel" in k)
    values["other_kernels"] = sum(kernels.values()) - values["li"] - values["init"]
    probe = case["queue_probe"]
    if probe["status"] != "unavailable":
        samples = probe["samples"]
        require(
            [r["fenced"] for r in samples] == [True, False, False, True]
            and all(
                r["final_get"] == case["reads_samples"][-1]["final"]
                and math.isfinite(r["elapsed_us"])
                and r["elapsed_us"] > 0
                for r in samples
            ),
            f"{role}: incomplete queue control",
        )
        values["queue_fenced"] = mean(r["elapsed_us"] for r in samples if r["fenced"])
        values["queue_pending"] = sum(r["begin_pending_after_submit"] for r in samples if not r["fenced"])
        values["idle_started"] = sum(not r["begin_pending_after_submit"] for r in samples if r["fenced"])
        if values["queue_pending"] == values["idle_started"] == 2:
            values["queue_ahead"] = mean(r["elapsed_us"] for r in samples if not r["fenced"])
            values["queue_sensitivity"] = values["queue_fenced"] - values["queue_ahead"]
    return case, values, kernels


def summarize(directory):
    files = defaultdict(dict)
    for path in directory.glob("r*-*.json"):
        match = re.fullmatch(r"r(\d+)-(serial|before|after)\.json", path.name)
        if match:
            number, role = int(match[1]), match[2]
            require(role not in files[number], f"Duplicate role {role} in round {number}")
            files[number][role] = path
    rounds = sorted(files)
    require(len(rounds) >= 2 and rounds == list(range(1, len(rounds) + 1)), "Need consecutive paired rounds from 1")
    require(all(set(files[i]) == set(ROLES) for i in rounds), "An interleaved round is incomplete")
    rows = {r: [sample(files[i][r], r) for i in rounds] for r in ROLES}
    base = rows["serial"][0][0]
    fused_ref = rows["before"][0][0]
    for role, measurements in rows.items():
        for case, _, _ in measurements:
            for field in ("snapshot_sha256", "geometry", "device", "soc", "kvaf_build", "warmup", "diagnostics_commit"):
                require(case.get(field) is not None and case[field] == base[field], f"{role}: different {field}")
            require(len(case["_samples"]) == len(base["_samples"]), f"{role}: different iteration count")
            require(
                case["kvip_commit"] == measurements[0][0]["kvip_commit"]
                and case["libraries"] == measurements[0][0]["libraries"],
                f"{role}: binary changed between rounds",
            )
            if role != "serial":
                require(_low_traffic(case) == _low_traffic(fused_ref), f"{role}: different LOW transfers")
                for key in ("ordered_output_sha256", "selected_set_sha256"):
                    require(
                        case["selection_comparison"][key] == fused_ref["selection_comparison"][key],
                        f"{role}: different fused outputs",
                    )
    print(
        f"MATCHED rounds={len(rounds)} units=us unprofiled_samples_per_role={len(rounds) * (len(base['_samples']) - 1)} "
        "profiled_replays_per_role=" + str(len(rounds))
    )
    for role, measurements in rows.items():
        headline = [s for case, _, _ in measurements for s in case["_samples"][1:]]
        first, last = measurements[0][1], measurements[-1][1]
        print(
            f"GRAPH {role} steady_mean/p50={mean(headline):.3f}/{median(headline):.3f} "
            f"last_minus_first_mean/p50={last['headline'] - first['headline']:+.3f}/{last['p50'] - first['p50']:+.3f} "
            f"profile_event/span={mean(v['profile_event'] for _, v, _ in measurements):.3f}/"
            f"{mean(v['kernel_span'] for _, v, _ in measurements):.3f}"
        )
        names = list(measurements[0][2])
        require(all(set(k) == set(names) for _, _, k in measurements), f"{role}: different kernel lists")
        print(f"KERNEL_MEAN {role} " + " ".join(f"{n}={mean(k[n] for _, _, k in measurements):.3f}" for n in names))
        pending = sum(v.get("queue_pending", 0) for _, v, _ in measurements)
        started = sum(v.get("idle_started", 0) for _, v, _ in measurements)
        observed = f"idle_begin_reached={started}/{len(rounds)*2} queued_begin_pending={pending}/{len(rounds)*2}"
        if all("queue_ahead" in v for _, v, _ in measurements):
            print(
                f"QUEUE {role} idle/queued_mean_us={mean(v['queue_fenced'] for _, v, _ in measurements):.3f}/"
                f"{mean(v['queue_ahead'] for _, v, _ in measurements):.3f} "
                f"idle_minus_queued_us={mean(v['queue_sensitivity'] for _, v, _ in measurements):+.3f} "
                f"{observed}"
            )
        else:
            print(f"QUEUE {role} status=unresolved {observed}")
    for candidate, control in (("before", "serial"), ("after", "serial"), ("after", "before")):
        pairs = [(a[1], b[1]) for a, b in zip(rows[candidate], rows[control])]

        def delta(key):
            return mean(a[key] - b[key] for a, b in pairs)

        signed = {
            k: delta(k)
            for k in (
                "headline",
                "profile_event",
                "kernel_span",
                "outside_kernel",
                "between_samples_unassigned",
                "init",
                "li",
                "other_kernels",
                "kernel_gap",
                "kernel_overlap",
            )
        }
        # headline = span + outside + (unprofiled minus profiled). This last term includes sampling noise.
        require(
            math.isclose(
                signed["headline"],
                signed["kernel_span"] + signed["outside_kernel"] + signed["between_samples_unassigned"],
                abs_tol=1e-7,
            ),
            "Unclosed mean budget",
        )
        print(f"DELTA {candidate}-{control} mean_us " + " ".join(f"{k}={v:+.3f}" for k, v in signed.items()))
        if all("queue_sensitivity" in a and "queue_sensitivity" in b for a, b in pairs):
            print(
                f"QUEUE_DELTA {candidate}-{control} idle_gap={delta('queue_fenced'):+.3f} "
                f"queued_gap={delta('queue_ahead'):+.3f} gap_change_idle_minus_queued={delta('queue_sensitivity'):+.3f}"
            )
        print(
            f"EDGE_DELTA {candidate}-{control} mean_us "
            + " ".join(
                f"{key}={delta(key):+.3f}" if all(key in a and key in b for a, b in pairs) else f"{key}=unresolved"
                for key in EDGE_KEYS
            )
        )
        print(
            f"PAIRED {candidate}-{control} round_steady_mean/p50_us="
            + ",".join(f"{a['headline'] - b['headline']:+.3f}/{a['p50'] - b['p50']:+.3f}" for a, b in pairs)
        )
    print(
        "CHECK snapshot=identical correctness=pass learned_LOW=pass lifecycle=pass same_profile_event_and_kernels=true"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    args = p.parse_args()
    try:
        summarize(args.directory)
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise SystemExit(f"ACCOUNTING_FAILED: {error}") from None


if __name__ == "__main__":
    main()
