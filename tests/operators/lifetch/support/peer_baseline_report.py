"""Compare the same three-kernel G2 measurement with the supplied reference."""
import argparse
import hashlib
import json
from pathlib import Path


def report(directory):
    matches = list(directory.glob("**/serial/path_result.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one serial result under {directory}, found {len(matches)}")
    result = json.loads(matches[0].read_text())
    inputs = json.loads((matches[0].parents[1] / "replay_input.json").read_text())
    geometry = dict(history=131328, resident=16384, heads=32, token_bytes=656,
                    output_stride=16896, batch=16)
    ids = json.dumps(inputs["identities"], separators=(",", ":")).encode()
    if (hashlib.sha256(ids).hexdigest() !=
            "83bef8d67299723cbd9e04eaf14b9e270bbfdb2609bca771a8aee9c1eabd92bc"
            or inputs["page_count"] != 16416 or inputs["rounds"] != 1
            or any(inputs["geometry"].get(k) != v for k, v in geometry.items())):
        raise ValueError("different G2 inputs, page layout or request order")
    timing = result["all_rounds"]
    if (result["correctness"] != "PASS" or result["path"] != "serial"
            or result["runtime_workers"] != 2 or result["transport"] != "urma"
            or timing["timing_protocol"] != "one_profiled_graph"
            or timing["graph_calls"] != 20 or timing["measured_graphs"] != 1
            or timing["samples"] != 20):
        raise ValueError("serial correctness or measurement protocol mismatch")
    reference = {"event_mean": 298.718547821, "span_mean": 294.4984125,
                 "span_p50": 292.69750000000045, "li": 224.149475,
                 "lidu_prepare": 35.776175, "fetch": 34.5287625}
    stages = timing["stage_mean_us"]
    if set(stages) != {"li", "lidu_prepare", "fetch"}:
        raise ValueError(f"unexpected serial stages: {sorted(stages)}")
    profile = json.loads((matches[0].parent / "diagnose/round-1/graph_profile.json").read_text())
    if (len(profile["samples"]) != 20
            or any(set(row["intervals_us"]) != set(stages)
                   or any(len(intervals) != 1 for intervals in row["intervals_us"].values())
                   for row in profile["samples"])
            or profile["accounting"].get("unclassified")):
        raise ValueError("invocations do not match the three reference kernel intervals")
    values = {"event_mean": timing["event_mean_us"],
              "span_mean": timing["profiler_mean_us"], "span_p50": timing["p50_us"], **stages}
    print("INPUT_CHECK PASS G2 BS16 Q4 656B pages=16416 request_order=matched workers=2")
    for k, ref in reference.items():
        print(f"BASELINE {k} us={values[k]:.3f} peer={ref:.3f} "
              f"delta_us={values[k]-ref:+.3f} delta_pct={(values[k]/ref-1)*100:+.2f}")
    # A screening threshold for this one graph; subsequent replicates can estimate drift.
    inside = all(abs(values[k] / reference[k] - 1) <= .05 for k in ("event_mean", "span_p50"))
    print("INITIAL_SCREEN event_mean_and_span_p50_within_5pct=" + str(inside).lower())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    report(parser.parse_args().directory)
