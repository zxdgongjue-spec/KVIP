# SPDX-License-Identifier: Apache-2.0
"""Run one real G2 cohort at every fixed-Q4 batch size on a prepared A5 container."""

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

from fixed_q4_packed_archive import NATIVE, same_production
from fixed_q4_packed_report import STAGES, read_pair, report_cell
from large_batch_input import FIXED_BATCHES, read_source, save_source
from large_batch_pair import CALLS, WARMUP
from large_batch_placement import placement
from reporting import atomic_json

SUPPORT = Path(__file__).resolve().parent
REPO = SUPPORT.parents[3]
KVAF_REVISION = "813f1750a50b7e6c92d5f5397f2fbc05befef6da"
GROUP, COHORT = 2, 0
BLOCKS = (("serial", "fused"), ("fused", "serial"))


def plan():
    return dict(
        protocol="ten_independent_calls_one_device_event",
        production_revision=NATIVE,
        group=GROUP,
        cohort=COHORT,
        batch_sizes=FIXED_BATCHES,
        blocks=BLOCKS,
        oracles=len(FIXED_BATCHES),
        measured_graphs=len(FIXED_BATCHES) * sum(map(len, BLOCKS)),
        calls_per_graph=CALLS,
        graph_warmups=WARMUP,
        prefetch=dict(tokens=1024, windows_requested=4, rank_start=1024),
    )


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


def run_arm(args, directory, batch, arm, prepared, node, cpus, *, oracle=None, adapter=None):
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
        "--trace", str(args.trace), "--group", str(GROUP), "--round", "1",
        "--batch", str(batch), "--cohort", str(COHORT), "--prepared", str(prepared),
        "--arm", arm, "--device", str(args.device), "--physical-npu", str(args.physical_npu),
        "--kvaf-root", str(args.kvaf_root), "--kvaf-server", f"127.0.0.1:{port}",
        "--output", str(directory),
    ]
    if oracle is not None:
        worker += ["--packed-ten", "--serial-result", str(oracle)]
    if args.keep_trace:
        worker += ["--keep-trace"]

    print(f"CASE_BEGIN G{GROUP} BS{batch} cohort={COHORT} arm={arm} output={directory}", flush=True)
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
            raise RuntimeError(f"G{GROUP} BS{batch} {arm} worker failed; see {worker_log}")
        if monitor.wait(timeout=10):
            raise RuntimeError(f"G{GROUP} BS{batch} {arm} placement probe failed; see {monitor_log}")
        actual_node, actual_adapter = placement(thp)
        if actual_node != node or (adapter and actual_adapter != adapter):
            raise ValueError(f"Requested NUMA/UB {node}/{adapter}, measured {actual_node}/{actual_adapter}: {thp}")
        (directory / "placement_verified").touch()
    return actual_adapter


def summarize_cell(out, cell, batch):
    observations = [
        dict(order=order, mean_us=read_pair(cell / f"block{index}", GROUP, batch, COHORT, out))
        for index, order in enumerate(BLOCKS, 1)
    ]
    times = {
        arm: {stage: mean(row["mean_us"][arm][stage] for row in observations) for stage in STAGES}
        for arm in BLOCKS[0]
    }
    saved = {stage: times["serial"][stage] - times["fused"][stage] for stage in STAGES}
    return dict(group=GROUP, batch=batch, cohort=COHORT, cohorts=1, blocks=len(BLOCKS),
                calls_per_graph=CALLS, mean_us=times, saved_us=saved,
                graph_gain_pct=100 * saved["graph"] / times["serial"]["graph"],
                reached_20pct=times["fused"]["graph"] <= 0.8 * times["serial"]["graph"],
                observations=observations)


def write_summary(out, rows, metadata):
    atomic_json(out / "packed_summary.json", {
        **metadata, "complete": len(rows) == len(FIXED_BATCHES),
        "completed_graphs": len(rows) * sum(map(len, BLOCKS)), "cells": rows,
    })
    fields = [
        "group", "batch", "cohort", *[
            f"{arm}_{stage}_us" for arm in BLOCKS[0] for stage in STAGES
        ], "graph_gain_pct",
    ]
    with (out / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values = {key: row[key] for key in ("group", "batch", "cohort", "graph_gain_pct")}
            for arm in BLOCKS[0]:
                for stage in STAGES:
                    values[f"{arm}_{stage}_us"] = row["mean_us"][arm][stage]
            writer.writerow(values)


def run(args):
    revision = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", REPO, "status", "--porcelain", "-uno"], text=True).strip():
        raise ValueError("The checked-out measurement revision has tracked modifications")
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
    args.trace = args.trace.resolve(strict=True)
    source, refs = read_source(args.trace, GROUP, 1)
    node, cpus = cpu_affinity(args.cpu_numa)

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f"Choose an empty output directory: {out}")
    (out / "native.txt").write_text(revision + "\n")
    (out / "physical_npu.txt").write_text(f"{args.physical_npu}\n")
    (out / "host.txt").write_text(os.uname().nodename + "\n")
    prepared = out / f"g{GROUP}" / "donors"
    prepared.parent.mkdir()
    save_source(prepared, args.trace, GROUP, 1, source, refs)
    print(f"RESULT_ROOT={out}", flush=True)

    metadata = {**plan(), "test_revision": revision, "kvaf_revision": kvaf_revision,
                "physical_npu": args.physical_npu, "device": args.device, "trace": str(args.trace), "cpu_numa": node}
    cells = {batch: out / f"g{GROUP}/bs{batch}/cohort{COHORT}" for batch in FIXED_BATCHES}
    adapter = args.cpu_ub_device
    for batch, cell in cells.items():
        oracle = cell / "oracle" / "serial"
        adapter = run_arm(args, oracle, batch, "serial", prepared, node, cpus, adapter=adapter)
    metadata["cpu_ub_device"] = adapter

    rows = []
    for batch, cell in cells.items():
        oracle = cell / "oracle" / "serial"
        for index, order in enumerate(BLOCKS, 1):
            for arm in order:
                run_arm(args, cell / f"block{index}" / arm, batch, arm, prepared, node, cpus,
                        oracle=oracle, adapter=adapter)
        row = summarize_cell(out, cell, batch)
        rows.append(row)
        report_cell(row)
        write_summary(out, rows, metadata)
    print(f"PACKED_COMPLETE cells={len(rows)} SUMMARY={out / 'packed_summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, help="real Q4 capture containing G2, round 1")
    parser.add_argument("--device", type=int, help="ACL user ordinal corresponding to the selected physical card")
    parser.add_argument("--physical-npu", type=int, default=2)
    parser.add_argument("--kvaf-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cpu-numa", type=int, help="CPU memory placement node; defaults to the first allowed node")
    parser.add_argument("--cpu-ub-device", default=os.environ.get("KVAF_CPU_UB_DEVICE"))
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--keep-trace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(plan(), indent=2))
        return
    for name in ("trace", "device", "kvaf_root", "output"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required for live measurement")
    if args.device < 0 or args.physical_npu < 0 or args.timeout_seconds <= 0:
        parser.error("device and physical card must be nonnegative and timeout must be positive")
    run(args)


if __name__ == "__main__":
    main()
