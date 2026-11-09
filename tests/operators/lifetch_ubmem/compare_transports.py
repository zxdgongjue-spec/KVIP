#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the same controlled Q4 LI fixture against URMA and UBmem on an NPU.

Each case runs in a fresh process; no model, service startup, or performance
claim beyond the captured graph events is implied. Long logs stay on disk.
"""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]


def execute(command, log, timeout, label):
    start = time.monotonic()
    print(f"START {label}; log={log}", flush=True)
    with log.open("w", buffering=1) as output:
        process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        while True:
            try:
                code = process.wait(timeout=20)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - start
                print(f"RUNNING {label} +{elapsed:.0f}s log_bytes={log.stat().st_size}", flush=True)
                if elapsed >= timeout:
                    # The URMA supervisor owns a separate worker process group;
                    # SIGINT lets it terminate and reap that group itself.
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    return 124
    return code


def extract(log, prefixes):
    rows = log.read_text(errors="replace").splitlines()
    return [line for line in rows if any(line.startswith(prefix) for prefix in prefixes)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--kvaf-root", type=Path, required=True)
    parser.add_argument("--ubmem-server", required=True)
    parser.add_argument("--urma-server", required=True)
    parser.add_argument("--soc", required=True)
    parser.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/cann"))
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--history", type=int, default=131072)
    parser.add_argument("--resident", type=int, default=16384)
    parser.add_argument("--index-heads", type=int, choices=(32, 64), default=32)
    parser.add_argument("--miss-percent", type=int, nargs="+", default=[10, 30, 50])
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--output", type=Path, default=ROOT / "tests/results/ubmem-compare")
    args = parser.parse_args(argv)
    if (
        not 1 <= args.batch <= 32
        or args.iterations < 1
        or args.warmup < 0
        or args.timeout_seconds < 20
        or any(not 0 <= value <= 100 for value in args.miss_percent)
    ):
        parser.error("invalid batch, miss ratio, timing count or timeout")
    root = args.kvaf_root.resolve()
    build = root / "build-npu"
    if not (build / "CMakeCache.txt").is_file():
        parser.error(f"KVAF NPU build missing: {build}; run scripts/build.sh --transport ubmem first")
    out = args.output.resolve() / (time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}")
    out.mkdir(parents=True)
    summary = out / "summary.log"
    lines = [
        "URMA/UBmem controlled fixture: same LI bytes, hot permutation and service LRU candidates; "
        "656-byte payload contents match; output strides and capture APIs may differ; "
        "timing includes graph scheduling; not a paired replay or model TPOT.",
        f"batch={args.batch} history={args.history} resident={args.resident} heads={args.index_heads} "
        f"Q=4 topk=2048 windows=8 budget=512 graph=True warmup={args.warmup} iterations={args.iterations}",
    ]
    summary.write_text("\n".join(lines) + "\n")
    print(f"Summary: {summary}", flush=True)
    for miss in args.miss_percent:
        for transport in ("urma", "ubmem"):
            label = f"{transport}-miss{miss}"
            log = out / f"{label}.log"
            common = [
                "--device",
                str(args.device),
                "--soc",
                args.soc,
                "--history",
                str(args.history),
                "--resident",
                str(args.resident),
                "--index-heads",
                str(args.index_heads),
                "--miss-percent",
                str(miss),
                "--warmup",
                str(args.warmup),
                "--iterations",
                str(args.iterations),
            ]
            if transport == "urma":
                command = [
                    sys.executable,
                    "-u",
                    str(ROOT / "tests/operators/lifetch/test_lifetch.py"),
                    *common,
                    "--kvaf-server",
                    args.urma_server,
                    "--kvaf-root",
                    str(root),
                    "--cann",
                    str(args.cann),
                    "--batches",
                    str(args.batch),
                    "--intermediate-fetches",
                    "8",
                    "--prefetch-tokens",
                    "512",
                    "--trace-rounds",
                    "3",
                    "--output",
                    str(out / label),
                ]
                prefixes = (
                    "PASS: exit_code=",
                    "BS=",
                    "graph_p50_us=",
                    "actual_reads native:",
                    "actual_reads prototype_with_update:",
                    "SFA_max_abs=",
                )
                required = ("PASS: exit_code=0", "graph_p50_us=", "SFA_max_abs=")
            else:
                command = [
                    sys.executable,
                    "-u",
                    str(ROOT / "tests/operators/lifetch_ubmem/smoke.py"),
                    *common,
                    "--build-dir",
                    str(build),
                    "--kvaf-server",
                    args.ubmem_server,
                    "--batches",
                    str(args.batch),
                    "--token-bytes",
                    "656",
                    "--prefetch-tokens",
                    "512",
                ]
                prefixes = ("UBMEM_LIFETCH_PASS", "UBMEM_GRAPH_EVENT_US", "UBMEM_SFA_PASS", "UBMEM_FETCH_COUNTS",
                            "UBMEM_WINDOW_TRIGGERS")
                required = ("UBMEM_LIFETCH_PASS", "UBMEM_GRAPH_EVENT_US", "UBMEM_SFA_PASS", "UBMEM_WINDOW_TRIGGERS")
            code = execute(command, log, args.timeout_seconds, label)
            selected = extract(log, prefixes)
            success = code == 0 and all(any(line.startswith(prefix) for line in selected) for prefix in required)
            record = [f"{label}: {'PASS' if success else 'FAIL'} exit={code}", *selected]
            if not success:
                record += ["last log lines:", *log.read_text(errors="replace").splitlines()[-12:]]
            with summary.open("a") as result:
                result.write("\n".join(record) + "\n")
            print("\n".join(record), flush=True)
            if not success:
                print(f"STOP: {label}; full log={log}; summary={summary}", flush=True)
                return 1
    print(f"PASS: all requested cases; summary={summary}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
