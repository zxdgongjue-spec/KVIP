"""Summarize archived B24 terminal and window clocks and export the new evidence."""
import argparse
import json
from pathlib import Path
from statistics import mean, median
import tarfile
import tempfile

GROUPS = (10, 4, 17, 7, 2)
MAIN = "609fa26fd715f9409bbe74e1b2d3685c58dfb0cf"
CANDIDATE = "afe227c98ed17633f757163e3077fa090a743414"
STAGES = (
    "arrival", "initialize", "TopK", "publish.TopK", "deduplicate_and_slots", "output",
    "WQE_and_publish", "release_scratch", "control_gap", "diagnostic_snapshot",
    "index_update_and_join", "final_CQ_join", "prefetch_CQ_join", "retire",
)


def read(path):
    return json.loads(path.read_text())


def summarize_call(detail):
    head = detail["streaming_cycles"][0]
    timeline = head["terminal_timeline"]
    cores = timeline["cores"]
    blocks = {c["block"]: c for c in cores}
    if len(cores) != 64 or set(blocks) != set(range(64)):
        raise ValueError("Expected 16 rows with four terminal cores each")
    latest = blocks[timeline["last_block"]]
    if latest["end_us"] != max(c["end_us"] for c in cores):
        raise ValueError("Last retiring block does not match the clock record")
    stages = dict.fromkeys(STAGES, 0.0)
    stages["arrival"] = latest["start_us"]
    for stage in latest["stages"]:
        stages[stage["name"]] += stage["duration_us"]
    if abs(sum(stages.values()) - timeline["observed_span_us"]) > 1e-4:
        raise ValueError("Recorded phases do not cover the last retiring core")
    hz = timeline["clock_mhz"]
    origin = min(c["stages"][0]["start_cycles"] for c in cores)
    phase = {s["name"]: s for s in latest["stages"]}
    final = head["final_get_bandwidth"]
    posts = final["groups"]
    if not posts or sum(p["entries"] for p in posts) != final["entries"]:
        raise ValueError("Missing final submission records")
    events = {(e["epoch"], e["lane"]): e for e in head["prefetch_services"]}
    requests = detail["algorithm"]["requests"]
    if len(events) != 64 or len(requests) != 16:
        raise ValueError("Incomplete window or admission records")
    posted_blocks = {p["block"] for p in posts}
    remaining = dict(
        span=timeline["observed_span_us"],
        final_post_first=(min(p["first_cycles"] for p in posts) - origin) / hz,
        final_wqe_publish_last=max(s["end_us"] for c in cores for s in c["stages"]
                                   if s["name"] == "WQE_and_publish" and c["block"] in posted_blocks),
        final_cq_observed_last=(max(p["last_cycles"] for p in posts) - origin) / hz,
        critical_core_enter_final_cq=phase["final_CQ_join"]["start_us"],
        critical_core_final_entries=next((p["entries"] for p in posts if p["block"] == latest["block"]), 0),
        prefetch_scope_begin_relative_to_terminal=(head["prefetch_bandwidth"]["first_cycles"] - origin) / hz,
    )
    for e in range(4):
        rows = [events[e, b] for b in range(16)]
        remaining[f"window{e}_end_relative_to_terminal"] = (max(x["end_cycles"] for x in rows) - origin) / hz
        remaining[f"window{e}_entries"] = sum(x["posted_count"] for x in rows)
        for b in range(16):
            if events[e, b]["posted_count"] != requests[b]["windows"][e]["posted"]:
                raise ValueError("Window service and posted identity counts disagree")
    return {"stages": stages, "diagnostic": remaining, "last_row_q": divmod(latest["block"], 4)}


def summarize(root, group, arm, revision):
    dirs = list((root / f"g{group}" / arm / "results").glob("*/cases/diagnose/fused"))
    if len(dirs) != 1:
        raise ValueError(f"Expected one G{group} {arm} archive under {root}, found {len(dirs)}")
    fused = dirs[0]
    receipt = read(fused / "mainline_checks.json")
    if (receipt["production_revision"] != revision or receipt["measured_calls"] != 20 or
            receipt["ordered_diagnostic_rows"] != 320 or receipt["profiled_terminal_kernel"] != "v9_parallel_terminal"):
        raise ValueError(f"Unexpected worker provenance in {fused}")
    details = read(fused / "diagnose/round-1/detail_samples.json")
    profile = read(fused / "diagnose/round-1/full_attribution.json")
    if (len(details) != 20 or len(profile["samples"]) != 20 or
            [c["round"] for c in details] != list(range(20))):
        raise ValueError(f"Missing complete 20 calls in {fused}")
    calls = [summarize_call(d) for d in details]
    def summary(values):
        return dict(mean=mean(values), p50=median(values))
    groups = {k: {field: summary([c[k][field] for c in calls]) for field in calls[0][k]}
              for k in ("stages", "diagnostic")}
    groups["official_tail"] = summary([s["accounting"]["stages_us"]["topk_resolve_update"] for s in profile["samples"]])
    groups["last_row_q_by_call"] = [c["last_row_q"] for c in calls]
    return fused, groups, receipt


def run(work, candidate):
    candidate = candidate.resolve()
    main = Path((candidate / "reference.txt").read_text().strip()).resolve()
    if (candidate.joinpath("candidate.txt").read_text().strip() != CANDIDATE or
            any(x.joinpath("main.txt").read_text().strip() != MAIN or
                x.joinpath("physical_npu.txt").read_text().strip() != "4" for x in (candidate, main)) or
            main.joinpath("candidate.txt").exists()):
        raise ValueError("Expected the isolated candidate and archived upstream on physical NPU4")
    report = {}
    files = []
    for group in GROUPS:
        _, old, old_proof = summarize(main, group, "mainline", MAIN)
        fused, new, new_proof = summarize(candidate, group, "li-window", CANDIDATE)
        if (old_proof["capture_operand_hashes"] != new_proof["capture_operand_hashes"] or
                old_proof["runtime"] != new_proof["runtime"]):
            raise ValueError(f"G{group} runtime or replay operands differ")
        comparison_path = candidate / f"g{group}/comparison.json"
        comparison = read(comparison_path)
        if (comparison["production_revision"] != CANDIDATE or comparison["group"] != group or
                Path(comparison["archive"]).resolve() != main):
            raise ValueError(f"G{group} comparison names a different candidate or mainline archive")
        for name, data in (("mainline", old), ("candidate", new)):
            if abs(data["official_tail"]["mean"] - comparison[name]["final_topk_through_tail"][0]) > 0.001:
                raise ValueError(f"G{group} profiler and official comparison tail disagree")
        fields = {s: new["stages"][s]["mean"] - old["stages"][s]["mean"] for s in STAGES}
        delta = new["diagnostic"]["span"]["mean"] - old["diagnostic"]["span"]["mean"]
        official = new["official_tail"]["mean"] - old["official_tail"]["mean"]
        print(f"G{group} candidate_minus_main_us official_tail={official:+.2f} diagnostic_tail={delta:+.2f}"
              f" index_and_join={fields['index_update_and_join']:+.2f} final_CQ_join={fields['final_CQ_join']:+.2f}"
              f" prefetch_CQ_join={fields['prefetch_CQ_join']:+.2f} other={delta-fields['index_update_and_join']-fields['final_CQ_join']-fields['prefetch_CQ_join']:+.2f}", flush=True)
        report[f"G{group}"] = dict(mainline=old, candidate=new, delta_mean_us=fields)
        files += [comparison_path, fused.parent / "replay_input.json", fused.parent / "config.json",
                  candidate / f"g{group}/li-window/thp.json"]
        files += [fused / name for name in ("mainline_checks.json", "path_result.json",
                    "score_topk_check.json", "score_topk_ids.npz", "window_workers.json",
                    "diagnose/round-1/detail_samples.json", "diagnose/round-1/full_attribution.json")]
    files += [candidate / f for f in ("run.log", "candidate.txt", "main.txt", "reader.txt",
                                      "reference.txt", "host.txt", "physical_npu.txt", "kvaf.txt")]
    for file in files:
        if not file.is_file() or file.stat().st_size == 0:
            raise ValueError(f"Missing or empty archive member {file}")
    destination = Path(tempfile.mkdtemp(prefix="isolated-li-clocks.", dir=work))
    summary = destination / "summary.json"
    summary.write_text(json.dumps(dict(mainline=str(main), candidate=str(candidate),
        stage_scope="diagnostic replay; stages of the last retiring core, candidate minus mainline unpaired means",
        last_row_q_encoding="zero based row and Q",
        clock_scope="CQ timestamps mark the core observing completion. Prefetch scope begins at first doorbell or archived service-entry fallback.",
        groups=report), indent=2) + "\n")
    package = destination / "isolated-li-clocks.tar.gz"
    partial = package.with_suffix(".partial")
    with tarfile.open(partial, "w:gz", compresslevel=3, dereference=True) as archive:
        archive.add(summary, arcname=summary.name)
        for file in files:
            archive.add(file, arcname=str(Path(candidate.name) / file.relative_to(candidate)))
    partial.replace(package)
    print(f"RETURN_FILE={package}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    run(args.work, args.candidate or args.work / "peer-baseline/li-window-five.Pjd3Ub")
