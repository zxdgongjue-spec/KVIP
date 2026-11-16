"""Execute paired score routes on a prepared CANN host; preserve per-case failure and provenance."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from score_routes import boundary_matrix, make_fixture, plan


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    output = Path(args.output)
    driver = Path(args.driver).resolve()
    if not driver.is_file() or args.iterations < 1 or args.iterations > 100:
        raise ValueError("existing built driver and iterations 1..100 required")
    for b in args.batches:
        plan(b, args.history, args.heads, args.cube, args.vector)
    rows = [
        r
        for r in boundary_matrix(args.history)
        if r["version"] == args.version
        and r["heads"] == args.heads
        and r["cube"] == args.cube
        and r["vector"] == args.vector
        and r["batch"] in args.batches
    ]
    if args.producers:
        if any(p not in (2, 4, 8, 16, args.cube) for p in args.producers):
            raise ValueError("producer filter must select an implemented matrix candidate")
        rows = [r for r in rows if r["mode"] == "batch_query" or r["producers"] in args.producers]
    if not rows:
        raise ValueError("no matching score cases; only C32/V64 and C28/V56 matrix budgets are supported")
    config_path = Path(args.config).resolve()
    configs = {str(config_path): sha(config_path)}
    if args.run_mode == "sim":
        if config_path.name != "config_stars.json":
            raise ValueError("sim --config must be the actually loaded config_stars.json")
        configs[str(config_path.with_name("config.json"))] = sha(config_path.with_name("config.json"))
    config = hashlib.sha256(json.dumps(configs, sort_keys=True).encode()).hexdigest()
    # A shared lease is required for Model/native runs. It must be shared across
    # containers when they access the same simulator/device bandwidth domain.
    lease = Path(args.lease_file)
    lease.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lease, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(fd, json.dumps(dict(pid=os.getpid(), output=str(output), started=time.time())).encode())
        os.close(fd)
        output.mkdir(parents=True, exist_ok=False)
        binary = sha(driver)
        for b in sorted({r["batch"] for r in rows}):
            fixture = output / ("fixture-b" + str(b))
            meta = make_fixture(fixture, b, args.history, args.heads)
            input_sha = hashlib.sha256(json.dumps(meta["hashes"], sort_keys=True).encode()).hexdigest()
            for r in (r for r in rows if r["batch"] == b):
                name = f"b{b}-p{r['producers']}-q{r['query_group']}"
                case = output / name
                argv = [
                    str(driver),
                    str(fixture),
                    str(case),
                    str(b),
                    str(args.history),
                    str(args.heads),
                    str(args.cube),
                    str(args.vector),
                    str(r["producers"]),
                    str(r["query_group"]),
                    str(args.iterations),
                    str(args.device),
                ]
                r.update(
                    input_sha=input_sha,
                    binary_sha=binary,
                    device=str(args.device),
                    run_mode=args.run_mode,
                    config_sha=config,
                    iterations=args.iterations,
                )
                command = output / (name + "-command.json")
                command.write_text(
                    json.dumps(dict(argv=argv, exit_code=None, status="RUNNING", configs=configs), indent=2) + "\n"
                )
                try:
                    proc = subprocess.run(argv, capture_output=True, text=True, timeout=args.timeout)
                    (output / (name + ".log")).write_text(proc.stdout + proc.stderr)
                    command.write_text(
                        json.dumps(
                            dict(argv=argv, exit_code=proc.returncode, status="RETURNED", configs=configs), indent=2
                        )
                        + "\n"
                    )
                    check = json.loads((case / "score-check.json").read_text())
                    if (
                        proc.returncode
                        or check["status"] != "PASSED"
                        or check["version"] != args.version
                        or check["run_mode"] != args.run_mode
                    ):
                        raise ValueError("driver failed/version or backend mismatch")
                    if sha(case / "score-keys.bin") != meta["hashes"]["oracle"]:
                        raise ValueError("parent oracle SHA mismatch")
                    if args.run_mode == "sim":
                        if check["score_ms"] is not None:
                            raise ValueError("Model must not report physical score_ms")
                        loaded = {
                            str(Path(p).resolve())
                            for p in re.findall(r"(?:path is|Path:)\s*([^\s]+config(?:_stars)?\.json)", proc.stdout)
                        }
                        if not set(configs) <= loaded:
                            raise ValueError("Model did not report the declared configuration paths")
                        if len(re.findall(r"\[INFO\] Total tick: \d+", proc.stdout)) != 1:
                            raise ValueError("Model normal completion footer missing or ambiguous")
                    # Host wall time under CA Model is never physical score_ms.
                    r.update(status="PASSED", oracle_passed=True, score_ms=check["score_ms"])
                except subprocess.TimeoutExpired as error:

                    def decoded(value):
                        return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")

                    (output / (name + ".log")).write_text(decoded(error.stdout) + decoded(error.stderr))
                    command.write_text(
                        json.dumps(
                            dict(argv=argv, exit_code=None, status="TIMED_OUT", timeout=args.timeout, configs=configs),
                            indent=2,
                        )
                        + "\n"
                    )
                    r["status"] = "FAILED"
                    (output / (name + "-failure.txt")).write_text(str(error) + "\n")
                except (OSError, ValueError, KeyError) as error:
                    r["status"] = "FAILED"
                    (output / (name + "-failure.txt")).write_text(str(error) + "\n")
                write_rows(output / "measurements.csv", rows)
        return rows
    finally:
        # This process created the lease; do not remove someone else's lease.
        lease.unlink()


def write_rows(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--version", choices=("current",), default="current")
    p.add_argument("--run-mode", required=True, choices=("sim", "npu"))
    p.add_argument("--config", required=True, help="actual Model config or native environment manifest")
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 24, 32])
    p.add_argument("--producers", type=int, nargs="+")
    p.add_argument("--history", type=int, default=131072)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--cube", type=int, default=32)
    p.add_argument("--vector", type=int, default=64)
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--lease-file", default="/tmp/kvip-cann91-simulator.lease")
    args = p.parse_args()
    if not args.batches or any(b < 1 or b > 32 for b in args.batches):
        p.error("batches 1..32 required")
    rows = run(args)
    failed = sum(r["status"] == "FAILED" for r in rows)
    print(json.dumps(dict(cases=len(rows), failed=failed, scope="score-only")))
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
