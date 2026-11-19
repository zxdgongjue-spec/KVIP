#!/usr/bin/env python3
"""Compare SDK LI/LIDU/Get with P2 and staged LIFetch, using the public harness."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CASES = {
    "legacy": ("legacy", "separate", "wave", "separate"),
    "history_before": ("history", "separate", "wave", "separate"),
    "history_direct": ("history", "direct", "phase", "inline"),
}
METRICS = ("mean_us", "p50_us", "p99_us")


def csv_numbers(value, converter):
    values = tuple(converter(item) for item in value.split(","))
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Expected distinct comma-separated numbers")
    return values


def arguments(argv=None, *, serial=False):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kvaf-server", required=True, help="Existing URMA CPU service HOST:PORT")
    p.add_argument("--kvaf-root", type=Path, default=ROOT / "third_party/KVAF")
    p.add_argument("--device", type=int, default=5)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--batches", type=lambda s: csv_numbers(s, int), default=(1, 16))
    p.add_argument("--miss-percent", type=lambda s: csv_numbers(s, float),
                   default=(0, 3, 5, 10, 15, 20) if serial else (0, 3, 10, 50, 90, 100))
    p.add_argument("--prefetch", type=lambda s: csv_numbers(s, int), default=(0, 512))
    p.add_argument("--rounds", type=int, default=3, help="Measured rounds, in addition to discarded r0")
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--aic-budget", type=int, default=32)
    p.add_argument("--aiv-budget", type=int, default=64)
    p.add_argument("--q-workers", type=int, choices=(1, 2, 4), default=2)
    p.add_argument("--intermediate-fetches", type=int, choices=(2, 8), default=2)
    p.add_argument("--timeout-seconds", type=int, default=3600, help="Timeout per public harness worker")
    p.add_argument("--output-root", type=Path, default=Path("/home/w00568585"))
    p.add_argument("--plan", action="store_true", help="Print commands without executing or creating files")
    if serial:
        p.description = "Compare original serial, numeric-direct optimized serial, and legacy fused full chains"
        p.add_argument("--serial-history-parts", type=int, choices=(0, 1, 2, 4, 8, 16, 32), default=0,
                       help="0=largest power of two <= min(16, paired score capacity/batch)")
    a = p.parse_args(argv)
    if (a.device < 0 or a.rounds < 1 or a.warmup < 1 or a.iterations < 1 or a.timeout_seconds <= 0
            or not 1 <= a.aic_budget <= 32 or not 2 <= a.aiv_budget <= 64):
        p.error("Invalid device, sampling, timeout, or hardware budget")
    if (any(not 1 <= b <= 16 for b in a.batches)
            or any(not math.isfinite(m) or not 0 <= m <= 100 for m in a.miss_percent)
            or any(pf < 0 for pf in a.prefetch)):
        p.error("Batches must be 1..16; miss 0..100; prefetch nonnegative")
    # Match PrepareShape/ValidateStreamingBudget, including P1 fallback.
    li_cores = max(b * (1 + (2 * b <= a.aic_budget)) for b in a.batches)
    if max(a.batches) > a.aic_budget or max(2 * li_cores, 3 * max(a.batches) + 4 * a.q_workers) > a.aiv_budget:
        p.error("Budget too small for the matched legacy control; reduce batch/Fetch workers or increase budget")
    a.kvaf_root = a.kvaf_root.resolve()
    return a


def plan(a, output):
    rng = random.Random(1417)
    for round_no in range(a.rounds + 1):
        combinations = [(pf, case) for pf in a.prefetch for case in CASES]
        rng.shuffle(combinations)
        for pf, case in combinations:
            route, post, prepare, service = CASES[case]
            name = f"r{round_no}_pf{pf}_{case}"
            command = [a.python, str(ROOT / "tests/operators/lifetch/test_lifetch.py"), "bench",
                       "--transport", "urma", "--device", str(a.device),
                       "--kvaf-root", str(a.kvaf_root), "--kvaf-server", a.kvaf_server,
                       "--batches", ",".join(map(str, a.batches)),
                       "--miss-percent", ",".join(map(str, a.miss_percent)),
                       "--queries", "4", "--history", "131072", "--resident", "16384",
                       "--index-heads", "32", "--seed", "43",
                       "--aic-budget", str(a.aic_budget), "--aiv-budget", str(a.aiv_budget),
                       "--q-workers", str(a.q_workers), "--intermediate-fetches", str(a.intermediate_fetches),
                       "--prefetch-tokens", str(pf), "--score-route", route, "--score-post", post,
                       "--score-prepare", prepare, "--score-service", service,
                       "--warmup", str(a.warmup), "--iterations", str(a.iterations),
                       "--timeout-seconds", str(a.timeout_seconds), "--output", str(output / name)]
            yield dict(name=name, round=round_no, discard=round_no == 0, prefetch=pf, case=case, argv=command)


def read_run(output, job, a):
    tables = list((output / job["name"]).rglob("results.csv"))
    if len(tables) != 1:
        raise RuntimeError(f"Expected one public harness results.csv for {job['name']}, found {len(tables)}")
    with tables[0].open(newline="") as stream:
        raw = list(csv.DictReader(stream))
    expected = {(b, float(m)) for b in a.batches for m in a.miss_percent}
    observed, rows = set(), []
    for item in raw:
        if (item.get("accepted") != "True" or item.get("execution") != "COMPLETED"
                or item.get("correctness") != "PASS"):
            raise RuntimeError(f"Run is incomplete or not PASS: {job['name']} {item.get('case')}")
        point = (int(item["batch"]), float(item["miss_percent"]))
        if point not in expected or point in observed:
            raise RuntimeError(f"Unexpected/duplicate point {point} in {job['name']}")
        observed.add(point)
        row = {k: job[k] for k in ("name", "round", "discard", "prefetch", "case")}
        row.update(batch=point[0], miss_percent=point[1], report=item["report"])
        for prefix in ("serial", "fused"):
            for metric in METRICS:
                value = float(item[f"{prefix}_{metric}"])
                if not math.isfinite(value) or value <= 0:
                    raise RuntimeError(f"Invalid timing in {job['name']}")
                row[f"{prefix}_{metric}"] = value
        report = json.loads(Path(item["report"]).read_text())
        row["score_geometry"] = json.dumps(report.get("score_geometry"), sort_keys=True)
        row["resources"] = json.dumps(report.get("resources"), sort_keys=True)
        row["binary_sha256"] = report.get("binary_sha256")
        row["native_misses_per_request"] = json.dumps(report.get("native_misses_per_request"))
        row["read_samples"] = json.dumps(report.get("read_samples"), sort_keys=True)
        rows.append(row)
    if observed != expected:
        raise RuntimeError(f"Missing benchmark points in {job['name']}: {expected - observed}")
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(output, rows):
    write_csv(output / "full_chain.csv", rows)
    measured = [r for r in rows if not r["discard"]]
    lookup = {(r["round"], r["prefetch"], r["batch"], r["miss_percent"], r["case"]): r for r in measured}
    if len(lookup) != len(measured):
        raise RuntimeError("Duplicate measured rows")
    groups = {}
    for row in measured:
        groups.setdefault((row["prefetch"], row["batch"], row["miss_percent"], row["case"]), []).append(row)
    summary = []
    for key, members in sorted(groups.items()):
        item = dict(zip(("prefetch", "batch", "miss_percent", "case"), key))
        item["rounds"] = len(members)
        for prefix in ("serial", "fused"):
            for metric in METRICS:
                item[f"{prefix}_{metric}_round_mean"] = statistics.mean(r[f"{prefix}_{metric}"] for r in members)
        ratios = {"fused_over_serial": [], "fused_over_legacy": [], "fused_over_history_before": []}
        for r in members:
            pairing = (r["round"], r["prefetch"], r["batch"], r["miss_percent"])
            ratios["fused_over_serial"].append(r["fused_p50_us"] / r["serial_p50_us"])
            for name, control in (("fused_over_legacy", "legacy"), ("fused_over_history_before", "history_before")):
                ratios[name].append(r["fused_p50_us"] / lookup[(*pairing, control)]["fused_p50_us"])
        for name, values in ratios.items():
            item[name + "_p50_mean"] = statistics.mean(values)
            item[name + "_p50_per_round"] = json.dumps(values)
        item["fused_p50_round_stdev_us"] = statistics.stdev(r["fused_p50_us"] for r in members) if len(members) > 1 else 0
        summary.append(item)
    write_csv(output / "summary.csv", summary)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["Full chain | us | mean of measured-round p50 | ratios < 1 are faster | r0 discarded",
             "pf  batch miss% case            Serial    Fused   /Serial /legacy /history_before"]
    for r in summary:
        lines.append(f"{r['prefetch']:3} {r['batch']:5} {r['miss_percent']:5g} {r['case']:15} "
                     f"{r['serial_p50_us_round_mean']:8.2f} {r['fused_p50_us_round_mean']:8.2f} "
                     f"{r['fused_over_serial_p50_mean']:7.3f} {r['fused_over_legacy_p50_mean']:7.3f} "
                     f"{r['fused_over_history_before_p50_mean']:14.3f}")
    (output / "summary.log").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def run_experiment(a, *, make_plan=plan, read=read_run, summary=summarize, prefix="lifetch_full_chain", note=None):
    if a.plan:
        for job in make_plan(a, a.output_root / "PLAN_ONLY"):
            print(job["name"] + ": " + shlex.join(job["argv"]))
        return 0
    if any(s in os.environ.get("LD_PRELOAD", "").lower() for s in ("camodel", "cmodel")):
        raise RuntimeError("Run on physical A5 without simulator LD_PRELOAD")
    for command in ("npu-smi", a.python):
        if not shutil.which(command):
            raise RuntimeError(f"Command unavailable: {command}; source CANN and scripts/env.sh first")
    if not (a.kvaf_root / "build-npu/python/kvaf").is_dir():
        raise RuntimeError(f"Missing built KVAF package: {a.kvaf_root}/build-npu/python/kvaf")
    a.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=time.strftime(prefix + "_%Y%m%d_%H%M%S_"), dir=a.output_root))
    print(f"Results directory: {output}\nSource checkout: {ROOT}", flush=True)
    jobs = list(make_plan(a, output))
    metadata = dict(config={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, commands=jobs,
                    note=note or "Each run pairs SDK LI/LIDU/Get with one LIFetch route. r0 discarded. Same requested budgets; actual score geometry recorded. These paths do different cache/prefetch work.")
    for command, label in ((["git", "rev-parse", "HEAD"], "head"), (["git", "status", "--porcelain"], "source_status"), (["npu-smi", "info"], "npu_smi")):
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        metadata[label] = dict(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    try:
        for index, job in enumerate(jobs, 1):
            log = output / (job["name"] + ".log")
            print(f"[{time.strftime('%H:%M:%S')}] {index}/{len(jobs)} {job['name']} {'(discard)' if job['discard'] else ''}\nLog: {log}", flush=True)
            start = time.time()
            with log.open("w") as stream:
                result = subprocess.run(job["argv"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            receipt = dict(**job, exit_code=result.returncode, start_unix=start, end_unix=time.time(),
                           log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
            (output / (job["name"] + ".receipt.json")).write_text(json.dumps(receipt, indent=2) + "\n")
            if result.returncode:
                print("\n".join(log.read_text(errors="replace").splitlines()[-40:]), file=sys.stderr)
                raise RuntimeError(f"{job['name']} failed (exit {result.returncode}); see {log}")
            rows.extend(read(output, job, a))
            write_csv(output / "full_chain.csv", rows)
        summary(output, rows)
    except Exception:
        (output / "exit-status.txt").write_text("1\n")
        print(f"Experiment stopped. Completed rows/logs retained in {output}", file=sys.stderr)
        raise
    (output / "exit-status.txt").write_text("0\n")
    print(f"Completed. Summary: {output / 'summary.csv'}", flush=True)
    return 0


def main(argv=None):
    return run_experiment(arguments(argv))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
