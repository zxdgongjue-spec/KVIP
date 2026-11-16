"""All 42 defaults: plan, CPU oracle, or real Model full-chain driver execution."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from staged_routes import cases, fixture, stage_tasks, owner_waves
from staged_oracle import check_case
from run_split_core_cases import lease


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("action", choices=("plan", "cpu", "run"))
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--driver", type=Path)
    p.add_argument("--case", action="append")
    p.add_argument("--route", choices=("auto", "history", "bsq"), default="auto")
    p.add_argument("--version", choices=("current", "historical"), default="current")
    p.add_argument("--timeout", type=int, default=10800)
    p.add_argument("--lease-file", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    specs = list(cases())
    names = {x["case"] for x in specs}
    if a.case and not set(a.case) <= names:
        p.error("unknown case; no default case is removed")
    if a.action == "run" and (
        not a.driver or not a.driver.is_file() or "libruntime_camodel.so" not in os.environ.get("LD_PRELOAD", "")
    ):
        p.error("Model execution needs a built driver and explicit CA Model runtime")
    rows = [
        dict(
            **c,
            producer=a.version,
            status="NOT_RUN",
            device_status="NOT_RUN",
            physical_npu=False,
            score_route=stage_tasks(
                c["batch"], c["resident"], c["history"], force=None if a.route == "auto" else a.route
            )["route"],
            owner_waves=owner_waves(c["batch"]),
            metric=None,
        )
        for c in specs
    ]

    def save():
        (a.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")

    save()
    if a.action == "plan":
        return 0
    cache = {}
    failed = False
    manager = lease("model", 0, a.lease_file) if a.action == "run" else __import__("contextlib").nullcontext()
    with manager:
        for row in rows:
            if a.case and row["case"] not in a.case:
                continue
            try:
                m = row["miss_percent"]
                if m not in cache:
                    folder = a.output / f"fixture-m{m}"
                    fixture(folder, m)
                    cache[m] = folder
                folder = cache[m]
                row["fixture_sha256"] = {x.name: sha(x) for x in folder.iterdir() if x.is_file()}
                row["cpu"] = check_case(folder, row)
                if a.action == "cpu":
                    row["status"] = "CPU_PASSED"
                    save()
                    continue
                dest = a.output / row["case"]
                argv = [
                    str(a.driver.resolve()),
                    str(folder.resolve()),
                    str(dest.resolve()),
                    str(row["batch"]),
                    str(row["history"]),
                    str(row["resident"]),
                    "28",
                    "56",
                    str(row["miss_per_query"]),
                    a.route,
                    "0",
                ]
                row.update(status="RUNNING", command=argv, driver_sha256=sha(a.driver))
                save()
                log = a.output / (row["case"] + ".log")
                start = time.monotonic()
                with log.open("w") as f:
                    result = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, timeout=a.timeout, cwd=a.output)
                text = log.read_text(errors="replace")
                ticks = re.findall(r"Total tick:\s*(\d+)", text)
                proof = (
                    json.loads((dest / "correctness.json").read_text()) if (dest / "correctness.json").exists() else {}
                )
                row.update(
                    exit_code=result.returncode,
                    log_sha256=sha(log),
                    host_wall_seconds=time.monotonic() - start,
                    proof=proof,
                )
                if (
                    result.returncode
                    or proof.get("status") != "PASSED"
                    or proof.get("producer") != a.version
                    or len(ticks) != 1
                    or "Model stopped successfully." not in text
                ):
                    raise RuntimeError("full device oracle/normal unique Model footer missing")
                row.update(
                    status="MODEL_PASSED",
                    device_status="PASSED",
                    metric=dict(name="whole Model ticks", value=int(ticks[0])),
                    measurement_scope="real QK/current streaming full algorithm; final GM adapter; prefetch0",
                )
            except Exception as e:
                row.update(status="FAILED", error=str(e))
                failed = True
            save()
            print(json.dumps(dict(case=row["case"], status=row["status"])), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
