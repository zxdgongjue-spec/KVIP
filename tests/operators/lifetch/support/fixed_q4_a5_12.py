# SPDX-License-Identifier: Apache-2.0
"""Run one captured or synthetic cohort at every fixed-Q4 batch size on A5."""

import argparse
from contextlib import ExitStack
import csv
import json
import os
from pathlib import Path
import shutil
import socket
from statistics import mean
import subprocess
import sys
import time

from fixed_q4_packed_archive import NATIVE, Q1_BASE, Q1_BATCHES, same_production, verify_q1_production
from fixed_q4_packed_report import (
    LARGE_WINDOW_MODES, LARGE_WINDOW_STAGES, Q1_STAGES, STAGES,
    read_large_q1_block, read_pair, read_q1_pair, report_cell,
)
from large_batch_input import BATCHES as NATIVE_LARGE_BATCHES
from large_batch_input import FIXED_BATCHES, SYNTHETIC_GROUP, read_source, save_source
from large_batch_pair import CALLS, WARMUP
from large_batch_placement import placement
from reporting import atomic_json
from workloads import prepare_compact_fixture

SUPPORT = Path(__file__).resolve().parent
REPO = SUPPORT.parents[3]
KVAF_REVISION = "813f1750a50b7e6c92d5f5397f2fbc05befef6da"
GROUP, COHORT = 2, 0
BLOCKS = (("serial", "fused"), ("fused", "serial"))
Q1_BLOCKS = (("off", "on"), ("on", "off"))
LARGE_Q1_BASE = "4e4ffd5f6ad7232d6e90a83f76dc2c627d51657f"
LARGE_Q1_BLOCKS = (tuple(LARGE_WINDOW_MODES), tuple(reversed(LARGE_WINDOW_MODES)))
LARGE_Q1_COMPARISONS = (
    ("deferred_pair", "native"), ("deferred_q1", "deferred_pair"), ("deferred_q1", "native"),
)
SYNTHETIC_FIXTURE = dict(history=131072, resident=16384, miss_percent=10, index_heads=32, seed=43)


def plan(synthetic=False, window_q1_pair=False, large_window_q1_pair=False):
    batches = NATIVE_LARGE_BATCHES if large_window_q1_pair else FIXED_BATCHES
    blocks = LARGE_Q1_BLOCKS if large_window_q1_pair else BLOCKS
    data = dict(
        protocol="ten_independent_calls_one_device_event",
        production_revision=NATIVE,
        input_kind="synthetic" if synthetic else "real",
        group=SYNTHETIC_GROUP if synthetic else GROUP,
        cohort=COHORT,
        batch_sizes=batches,
        blocks=blocks,
        oracles=len(batches),
        measured_graphs=len(batches) * sum(map(len, blocks)),
        calls_per_graph=CALLS,
        graph_warmups=WARMUP,
        prefetch=dict(tokens=1024, windows_requested=4, rank_start=1024),
    )
    if synthetic:
        data["synthetic_fixture_config"] = SYNTHETIC_FIXTURE
    if window_q1_pair:
        q1_graphs = len(Q1_BATCHES) * sum(map(len, Q1_BLOCKS))
        data["measured_graphs"] += q1_graphs
        data["window_q1_pair"] = dict(
            baseline_revision=Q1_BASE, enabled_batches=list(Q1_BATCHES),
            unsupported_batches=[batch for batch in FIXED_BATCHES if batch not in Q1_BATCHES],
            blocks=Q1_BLOCKS, measured_graphs=q1_graphs, scope="last_window_only",
        )
    if large_window_q1_pair:
        data["large_window_q1_pair"] = dict(
            baseline_revision=LARGE_Q1_BASE, enabled_batches=list(batches),
            queries_per_worker=LARGE_WINDOW_MODES, scope="deferred_last_window_only",
        )
    return data


def cpu_affinity(requested):
    available = os.sched_getaffinity(0)
    nodes = sorted(Path("/sys/devices/system/node").glob("node[0-9]*"))
    if requested is not None:
        nodes = [p for p in nodes if p.name == f"node{requested}"]
    for node in nodes:
        cpu_list = (node / "cpulist").read_text().strip()
        if not cpu_list:
            continue
        local = set()
        for item in cpu_list.split(","):
            first, _, last = item.partition("-")
            local.update(range(int(first), int(last or first) + 1))
        cpus = sorted(available & local)
        if cpus:
            return node.name.removeprefix("node"), ",".join(map(str, cpus))
    raise ValueError(f"No allowed CPUs on requested NUMA node {requested!r}")


def free_tcp_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def wait_for_service(process, log, port):
    deadline = time.monotonic() + 20
    marker = f"LISTEN address=127.0.0.1 port={port}"
    while time.monotonic() < deadline and process.poll() is None:
        if marker in log.read_text(errors="replace"):
            return
        time.sleep(0.1)
    raise RuntimeError(f"KVAF CPU service did not listen on port {port}; see {log}")


def run_arm(args, directory, batch, arm, source_options, group, node, cpus, *,
            oracle=None, adapter=None, window_q1=False, large_window_queries_per_worker=None):
    directory.mkdir(parents=True)
    port = free_tcp_port()
    cpu_log = directory / "cpu.log"
    thp = directory / "thp.json"
    command = [
        args.taskset, "-c", cpus, str(args.cpu_service),
        "--listen", "127.0.0.1", "--port", str(port), "--memory-page-mode", "thp",
    ]
    if adapter:
        command += ["--ub-device", adapter]
    worker = [
        sys.executable, "-B", str(SUPPORT / "large_batch_pair.py"),
        *source_options, "--group", str(group), "--round", "1",
        "--batch", str(batch), "--cohort", str(COHORT),
        "--arm", arm, "--device", str(args.device), "--physical-npu", str(args.physical_npu),
        "--kvaf-root", str(args.kvaf_root), "--kvaf-server", f"127.0.0.1:{port}",
        "--output", str(directory),
    ]
    if oracle is not None:
        worker += ["--packed-ten", "--serial-result", str(oracle)]
    if args.keep_trace:
        worker.append("--keep-trace")
    if args.window_q1_pair or args.large_window_q1_pair:
        worker.append("--q1-study")
    if window_q1:
        worker.append("--window-q1")
    if large_window_queries_per_worker is not None:
        worker += ["--large-window-queries-per-worker", str(large_window_queries_per_worker)]

    label = "synthetic" if group == SYNTHETIC_GROUP else f"G{group}"
    switch = f" window_q1={int(window_q1)}" if args.window_q1_pair and arm == "fused" else ""
    if args.large_window_q1_pair and arm == "fused":
        switch = f" window_route={directory.name}"
    print(f"CASE_BEGIN {label} BS{batch} cohort={COHORT} arm={arm}{switch} output={directory}", flush=True)
    with ExitStack() as stack:
        def launch(command, log):
            output = stack.enter_context(log.open("w"))
            process = subprocess.Popen(command, cwd=directory, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT)
            stack.callback(stop, process)
            return process

        cpu = launch(command, cpu_log)
        wait_for_service(cpu, cpu_log, port)
        monitor_log = directory / "placement.log"
        monitor = launch(
            [sys.executable, "-B", str(SUPPORT / "peer_thp.py"), "--unified-sdk",
             "--pid", str(cpu.pid), "--log", str(cpu_log), "--output", str(thp),
             "--timeout", str(args.timeout_seconds)],
            monitor_log,
        )
        worker_log = directory / "worker.log"
        job = launch(worker, worker_log)
        if job.wait(timeout=args.timeout_seconds):
            raise RuntimeError(f"{label} BS{batch} {arm} worker failed; see {worker_log}")
        if monitor.wait(timeout=10):
            raise RuntimeError(f"{label} BS{batch} {arm} placement probe failed; see {monitor_log}")
        actual_node, actual_adapter = placement(thp)
        if actual_node != node or (adapter and actual_adapter != adapter):
            raise ValueError(f"Requested NUMA/UB {node}/{adapter}, measured {actual_node}/{actual_adapter}: {thp}")
        (directory / "placement_verified").touch()
    return actual_adapter


def summarize_cell(out, cell, batch, group=GROUP):
    observations = [
        dict(order=order, mean_us=read_pair(cell / f"block{index}", group, batch, COHORT, out))
        for index, order in enumerate(BLOCKS, 1)
    ]
    times = {
        arm: {stage: mean(row["mean_us"][arm][stage] for row in observations) for stage in STAGES}
        for arm in BLOCKS[0]
    }
    saved = {stage: times["serial"][stage] - times["fused"][stage] for stage in STAGES}
    return dict(input_kind="synthetic" if group == SYNTHETIC_GROUP else "real",
                group=group, batch=batch, cohort=COHORT, cohorts=1, blocks=len(BLOCKS),
                calls_per_graph=CALLS, mean_us=times, saved_us=saved,
                graph_gain_pct=100 * saved["graph"] / times["serial"]["graph"],
                reached_20pct=times["fused"]["graph"] <= 0.8 * times["serial"]["graph"],
                observations=observations)


def q1_cell(args, out, cell, batch, oracle, source_options, group, node, cpus, adapter):
    if batch not in Q1_BATCHES:
        return dict(status="unsupported")
    observations = []
    for index, order in enumerate(Q1_BLOCKS, 1):
        block = cell / "q1" / f"block{index}"
        for arm in order:
            run_arm(args, block / arm, batch, "fused", source_options, group, node, cpus,
                    oracle=oracle, adapter=adapter, window_q1=arm == "on")
        observations.append(dict(order=order, **read_q1_pair(block, group, batch, COHORT, out)))
    left, right = observations
    if (left["context"], left["fused_bridge_sha256"]) != (right["context"], right["fused_bridge_sha256"]):
        raise ValueError(f"Q1 AB and BA used different paired contexts: {cell}")

    times = {}
    traffic = {}
    for arm in ("off", "on"):
        runs = [row["arms"][arm] for row in observations]
        times[arm] = {stage: mean(run["mean_us"][stage] for run in runs) for stage in Q1_STAGES}
        calls = [call for run in runs for call in run["traffic"]]
        traffic[arm] = {key: mean(call[key] for call in calls) for key in ("prefetch", "final")}
        traffic[arm]["total"] = sum(traffic[arm].values())
    changed_request_records, changed_totals = changed_get_counts(observations, "off", "on")
    saved = {stage: times["off"][stage] - times["on"][stage] for stage in Q1_STAGES}
    gain = 100 * saved["graph"] / times["off"]["graph"]
    print(f"BS{batch} window Q1 graph_off_us={times['off']['graph']:.2f} "
          f"graph_on_us={times['on']['graph']:.2f} graph_gain_pct={gain:+.2f} "
          f"changed_request_records={changed_request_records} changed_total_records={changed_totals}")
    return dict(status="completed", mean_us=times, saved_us=saved, graph_gain_pct=gain, traffic_mean=traffic,
                changed_request_records=changed_request_records, changed_total_records=changed_totals,
                observations=observations)


def changed_get_counts(observations, before_arm, after_arm):
    changed_request_records = changed_totals = 0
    for obs in observations:
        before_calls = obs["arms"][before_arm]["traffic"]
        after_calls = obs["arms"][after_arm]["traffic"]
        for before, after in zip(before_calls, after_calls, strict=True):
            for p0, p1, f0, f1 in zip(before["prefetch_by_request"], after["prefetch_by_request"],
                                      before["final_by_request"], after["final_by_request"], strict=True):
                changed_request_records += p0 != p1 or f0 != f1
                changed_totals += p0 + f0 != p1 + f1
    return changed_request_records, changed_totals


def large_q1_cell(args, out, cell, batch, oracle, source_options, group, node, cpus, adapter):
    observations = []
    for index, order in enumerate(LARGE_Q1_BLOCKS, 1):
        block = cell / "large_window" / f"block{index}"
        for arm in order:
            run_arm(args, block / arm, batch, "fused", source_options, group, node, cpus, oracle=oracle,
                    adapter=adapter, large_window_queries_per_worker=LARGE_WINDOW_MODES[arm])
        observations.append(dict(order=order, **read_large_q1_block(block, group, batch, COHORT, out)))
    left, right = observations
    if (left["context"], left["fused_bridge_sha256"]) != (right["context"], right["fused_bridge_sha256"]):
        raise ValueError(f"Native large forward and reverse groups used different paired contexts: {cell}")

    times, traffic, comparisons = {}, {}, {}
    for arm in LARGE_WINDOW_MODES:
        runs = [row["arms"][arm] for row in observations]
        times[arm] = {stage: mean(run["mean_us"][stage] for run in runs) for stage in LARGE_WINDOW_STAGES}
        calls = [call for run in runs for call in run["traffic"]]
        traffic[arm] = {key: mean(call[key] for call in calls) for key in ("prefetch", "final")}
        traffic[arm]["total"] = sum(traffic[arm].values())
    for arm, reference in LARGE_Q1_COMPARISONS:
        changed, changed_totals = changed_get_counts(observations, reference, arm)
        saved = {stage: times[reference][stage] - times[arm][stage] for stage in LARGE_WINDOW_STAGES}
        comparisons[f"{arm}_vs_{reference}"] = dict(
            graph_gain_pct=100 * saved["graph"] / times[reference]["graph"], saved_us=saved,
            changed_request_records=changed, changed_total_records=changed_totals,
        )
    print(f"BS{batch} native large " + " ".join(f"{arm}_graph_us={t['graph']:.2f}" for arm, t in times.items()))
    for pairing, data in comparisons.items():
        print(f"BS{batch} {pairing} graph_gain_pct={data['graph_gain_pct']:+.2f} "
              f"changed_request_records={data['changed_request_records']} "
              f"changed_total_records={data['changed_total_records']}")
    return dict(batch=batch, cohort=COHORT, group=group, mean_us=times, traffic_mean=traffic,
                comparisons=comparisons, observations=observations)


def write_large_q1_summary(out, rows, metadata):
    graphs_per_cell = sum(map(len, LARGE_Q1_BLOCKS))
    atomic_json(out / "packed_summary.json", {
        **metadata, "complete": len(rows) == len(NATIVE_LARGE_BATCHES),
        "completed_graphs": len(rows) * graphs_per_cell, "cells": rows,
    })
    fields = [
        "input_kind", "group", "batch", "cohort",
        *[f"{arm}_{stage}_us" for arm in LARGE_WINDOW_MODES for stage in LARGE_WINDOW_STAGES],
        *[f"{arm}_{counter}_requests" for arm in LARGE_WINDOW_MODES for counter in ("prefetch", "final", "total")],
        *[f"{arm}_vs_{ref}_{key}" for arm, ref in LARGE_Q1_COMPARISONS
          for key in ("graph_gain_pct", "changed_request_records", "changed_total_records")],
    ]
    with (out / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values = {"input_kind": metadata["input_kind"], **{key: row[key] for key in ("group", "batch", "cohort")}}
            for arm in LARGE_WINDOW_MODES:
                values.update({f"{arm}_{stage}_us": value for stage, value in row["mean_us"][arm].items()})
                values.update({f"{arm}_{key}_requests": value for key, value in row["traffic_mean"][arm].items()})
            for pairing, comparison in row["comparisons"].items():
                for key in ("graph_gain_pct", "changed_request_records", "changed_total_records"):
                    values[f"{pairing}_{key}"] = comparison[key]
            writer.writerow(values)


def write_summary(out, rows, metadata):
    q1_study = "window_q1_pair" in metadata
    q1_completed = sum(row.get("window_q1", {}).get("status") == "completed" for row in rows)
    complete = len(rows) == len(FIXED_BATCHES) and (not q1_study or q1_completed == len(Q1_BATCHES))
    atomic_json(out / "packed_summary.json", {
        **metadata, "complete": complete,
        "completed_graphs": len(rows) * sum(map(len, BLOCKS)) + q1_completed * sum(map(len, Q1_BLOCKS)),
        "cells": rows,
    })
    fields = [
        "input_kind", "group", "batch", "cohort", *[
            f"{arm}_{stage}_us" for arm in BLOCKS[0] for stage in STAGES
        ], "graph_gain_pct",
    ]
    if q1_study:
        fields += [
            "window_q1_status", "window_q1_graph_gain_pct", "window_q1_changed_request_records",
            "window_q1_changed_total_records",
            *[f"q1_{arm}_{stage}_us" for arm in ("off", "on") for stage in Q1_STAGES],
            *[f"q1_{arm}_{stage}_requests" for arm in ("off", "on") for stage in ("prefetch", "final", "total")],
        ]
    with (out / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values = {key: row[key] for key in ("input_kind", "group", "batch", "cohort", "graph_gain_pct")}
            for arm in BLOCKS[0]:
                for stage in STAGES:
                    values[f"{arm}_{stage}_us"] = row["mean_us"][arm][stage]
            if q1_study:
                trial = row["window_q1"]
                values["window_q1_status"] = trial["status"]
                if trial["status"] == "completed":
                    for key in ("graph_gain_pct", "changed_request_records", "changed_total_records"):
                        values[f"window_q1_{key}"] = trial[key]
                    for arm in ("off", "on"):
                        for stage, value in trial["mean_us"][arm].items():
                            values[f"q1_{arm}_{stage}_us"] = value
                        for stage, value in trial["traffic_mean"][arm].items():
                            values[f"q1_{arm}_{stage}_requests"] = value
            writer.writerow(values)


def run(args):
    revision = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", REPO, "status", "--porcelain", "-uno"], text=True).strip():
        raise ValueError("The checked-out measurement revision has tracked modifications")
    q1_study = args.window_q1_pair or args.large_window_q1_pair
    if q1_study:
        verify_q1_production(REPO, revision)
    else:
        same_production(REPO, NATIVE, revision)
    args.kvaf_root = args.kvaf_root.resolve(strict=True)
    kvaf_revision = subprocess.check_output(
        ["git", "-C", args.kvaf_root, "rev-parse", "HEAD"], text=True
    ).strip()
    if kvaf_revision != KVAF_REVISION:
        raise ValueError(f"Expected KVAF {KVAF_REVISION}, found {kvaf_revision}")
    args.cpu_service = args.kvaf_root / "build-cpu/runtime/kvaf_cpu_service"
    args.taskset = shutil.which("taskset")
    if not os.access(args.cpu_service, os.X_OK) or not args.taskset:
        raise ValueError(f"Required CPU-service executable and taskset are unavailable: {args.cpu_service}")
    node, cpus = cpu_affinity(args.cpu_numa)

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f"Choose an empty output directory: {out}")
    (out / "native.txt").write_text(revision + "\n")
    (out / "physical_npu.txt").write_text(f"{args.physical_npu}\n")
    (out / "host.txt").write_text(os.uname().nodename + "\n")
    print(f"RESULT_ROOT={out}", flush=True)

    metadata = {
        **plan(args.synthetic, args.window_q1_pair, args.large_window_q1_pair),
        "test_revision": revision, "kvaf_revision": kvaf_revision,
        "physical_npu": args.physical_npu, "device": args.device, "cpu_numa": node,
    }
    group = metadata["group"]
    metadata["production_revision"] = revision if q1_study else NATIVE
    if args.synthetic:
        fixture = out / "fixture"
        print(f"PREPARE_SYNTHETIC_FIXTURE={fixture}", flush=True)
        metadata["synthetic_fixture"] = prepare_compact_fixture(fixture, **SYNTHETIC_FIXTURE)
        metadata["fixture"] = str(fixture)
        source_options = ["--synthetic-fixture", str(fixture)]
        group_root = out / "synthetic"
    else:
        trace = args.trace.resolve(strict=True)
        group_root = out / f"g{group}"
        source, refs = read_source(trace, group, 1)
        prepared = group_root / "donors"
        prepared.parent.mkdir()
        save_source(prepared, trace, group, 1, source, refs)
        metadata["trace"] = str(trace)
        source_options = ["--trace", str(trace), "--prepared", str(prepared)]
    cells = {batch: group_root / f"bs{batch}/cohort{COHORT}" for batch in metadata["batch_sizes"]}
    adapter = args.cpu_ub_device
    for batch, cell in cells.items():
        oracle = cell / "oracle" / "serial"
        adapter = run_arm(args, oracle, batch, "serial", source_options, group, node, cpus, adapter=adapter)
    metadata["cpu_ub_device"] = adapter

    if args.large_window_q1_pair:
        rows = []
        for batch, cell in cells.items():
            oracle = cell / "oracle" / "serial"
            rows.append(large_q1_cell(args, out, cell, batch, oracle, source_options, group, node, cpus, adapter))
            write_large_q1_summary(out, rows, metadata)
        print(f"PACKED_COMPLETE cells={len(rows)} SUMMARY={out / 'packed_summary.json'}", flush=True)
        return

    rows = []
    for batch, cell in cells.items():
        oracle = cell / "oracle" / "serial"
        for index, order in enumerate(BLOCKS, 1):
            for arm in order:
                run_arm(args, cell / f"block{index}" / arm, batch, arm, source_options, group, node, cpus,
                        oracle=oracle, adapter=adapter)
        row = summarize_cell(out, cell, batch, group)
        if args.window_q1_pair:
            row["window_q1"] = dict(status="pending" if batch in Q1_BATCHES else "unsupported")
        rows.append(row)
        report_cell(row)
        write_summary(out, rows, metadata)
    if args.window_q1_pair:
        for row in rows:
            batch = row["batch"]
            if batch in Q1_BATCHES:
                cell = cells[batch]
                oracle = cell / "oracle" / "serial"
                row["window_q1"] = q1_cell(args, out, cell, batch, oracle, source_options, group, node, cpus, adapter)
                write_summary(out, rows, metadata)
    print(f"PACKED_COMPLETE cells={len(rows)} SUMMARY={out / 'packed_summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--trace", type=Path, help="real Q4 capture containing G2, round 1")
    inputs.add_argument("--synthetic", action="store_true", help="generate deterministic Q4 input without a replay directory")
    parser.add_argument("--device", type=int, help="ACL user ordinal corresponding to the selected physical card")
    parser.add_argument("--physical-npu", type=int, default=2)
    parser.add_argument("--kvaf-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cpu-numa", type=int, help="CPU memory placement node; defaults to the first allowed node")
    parser.add_argument("--cpu-ub-device", default=os.environ.get("KVAF_CPU_UB_DEVICE"))
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--keep-trace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--window-q1-pair", action="store_true",
                       help="Add same-binary fused off/on AB/BA for supported final-window Q1 shapes")
    modes.add_argument("--large-window-q1-pair", action="store_true",
                       help="Compare native and deferred two-Q / one-Q last windows on BS20, 24, 28, 32")
    args = parser.parse_args()
    if args.window_q1_pair and not args.synthetic:
        parser.error("--window-q1-pair requires --synthetic")
    if args.large_window_q1_pair and not args.synthetic:
        parser.error("--large-window-q1-pair requires --synthetic")
    if args.dry_run:
        print(json.dumps(plan(args.synthetic, args.window_q1_pair, args.large_window_q1_pair), indent=2))
        return
    if args.trace is None and not args.synthetic:
        parser.error("--trace or --synthetic is required for live measurement")
    for name in ("device", "kvaf_root", "output"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required for live measurement")
    if args.device < 0 or args.physical_npu < 0 or args.timeout_seconds <= 0:
        parser.error("device and physical card must be nonnegative and timeout must be positive")
    run(args)


if __name__ == "__main__":
    main()
