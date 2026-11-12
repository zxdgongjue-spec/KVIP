#!/usr/bin/env python3
"""Plan or execute the 60 BS/miss/sequence cases, retaining unrun rows explicitly."""

import argparse
import contextlib
import csv
import hashlib
import itertools
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "tests/operators/lifetch/support"
sys.path[:0] = [str(ROOT / "python"), str(SUPPORT)]


def cases(group="all"):
    batches = {"small": (1, 2), "medium": (4, 8), "large": (16,), "all": (1, 2, 4, 8, 16)}[group]
    for batch, percent, miss in itertools.product(batches, (30, 50, 100), (3, 5, 10, 15)):
        valid = {30: 39322, 50: 65536, 100: 131072}[percent]
        yield dict(
            case=f"b{batch}-h{percent}-m{miss}",
            batch=batch,
            history_percent=percent,
            valid_history=valid,
            history=(valid + 255) // 256 * 256,
            resident=16384,
            miss_percent=miss,
            miss_per_query=round(2048 * miss / 100),
        )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(case, directory, heads):
    import numpy as np
    from real_fetch_inputs import prepare_full_chain_fixture

    directory.mkdir(parents=True, exist_ok=False)
    fixture = directory / "fixture"
    info = prepare_full_chain_fixture(
        fixture,
        case["history"],
        case["resident"],
        case["miss_percent"],
        index_heads=heads,
        exact_miss=True,
        valid_history=case["valid_history"],
    )
    order = np.load(fixture / "physical_to_logical.npy")
    bits = np.load(fixture / "logical_scores.npy")[:, order]
    assert (bits[:, case["valid_history"] :] == 0).all()
    selected = []
    for row in bits:
        threshold = np.partition(row[: case["valid_history"]], -2048)[-2048]
        ids = np.flatnonzero(row >= threshold)
        assert len(ids) == 2048 and np.count_nonzero(ids >= 16384) == case["miss_per_query"]
        selected.append(set(ids[ids >= 16384].tolist()))
    assert len(set.union(*selected)) == info["unique_misses_per_request"] == 2 * case["miss_per_query"]
    keys = bits.copy()
    keys[:, : case["valid_history"]] ^= np.uint16(0x8000)
    keys.tofile(directory / "score.bin")
    return dict(
        fixture=info,
        files={p.name: sha(p) for p in fixture.iterdir() if p.is_file()},
        score_sha256=sha(directory / "score.bin"),
        scope="CPU real FP8 fixture and score oracle only",
    )


@contextlib.contextmanager
def lease(backend, device):
    path = Path("/tmp/kvip-cann91-simulator.lease" if backend == "model" else f"/tmp/kvip-npu-{device}.lease")
    owner = dict(task="split-core-matrix", pid=os.getpid(), lease=uuid.uuid4().hex)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(owner, f)
    try:
        yield
    finally:
        if path.exists() and json.loads(path.read_text()).get("lease") == owner["lease"]:
            path.unlink()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("plan", "cpu", "run"))
    p.add_argument("--group", choices=("all", "small", "medium", "large"), default="all")
    p.add_argument("--backend", choices=("model", "native"), default="model")
    p.add_argument("--transport", choices=("urma", "ubmem"), default="ubmem")
    p.add_argument("--budget", choices=("32/64", "28/56"), default="32/64")
    p.add_argument(
        "--shards", type=int, choices=(1, 4, 8), default=8, help="AIV TopK component only; not Fetch q-workers"
    )
    p.add_argument("--heads", type=int, choices=(32, 64), default=32)
    p.add_argument("--q-workers", type=int, choices=(1, 2, 4), default=2)
    p.add_argument("--model-driver", type=Path)
    p.add_argument("--soc")
    p.add_argument("--kvaf-root", type=Path)
    p.add_argument("--cann", default="/usr/local/Ascend/cann")
    p.add_argument("--case", action="append", help="Execute only named rows; other planned rows remain NOT_RUN")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--server", default="127.0.0.1:28880")
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--timeout", type=int, default=10800)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    if a.timeout <= 0 or a.warmup < 0 or a.iterations < 1:
        p.error("Invalid timeout, warmup or iterations")
    if a.action == "run" and a.backend == "model" and not (a.model_driver and a.model_driver.is_file()):
        p.error("Model execution requires the built --model-driver; prepare CANN environment first")
    if a.action == "run" and a.backend == "model" and "libruntime_camodel.so" not in os.environ.get("LD_PRELOAD", ""):
        p.error("Model mode requires LD_PRELOAD=libruntime_camodel.so; refusing an unverified runtime")
    if a.action == "run" and a.backend == "native" and not a.soc:
        p.error("Native execution requires the actual --soc")
    if a.case and not set(a.case) <= {x["case"] for x in cases(a.group)}:
        p.error("Unknown case in the selected group")
    a.output.mkdir(parents=True, exist_ok=False)
    c, v = map(int, a.budget.split("/"))
    rows = [
        {
            **x,
            "aic_budget": c,
            "aiv_budget": v,
            "heads": a.heads,
            "backend": a.backend,
            "status": "NOT_RUN",
            "metric": None,
        }
        for x in cases(a.group)
    ]

    def save():
        (a.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
        with (a.output / "results.csv").open("w", newline="") as f:
            fields = [k for k in rows[0] if k not in ("command", "fixture_proof")]
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    save()
    if a.action == "plan":
        print(json.dumps(dict(cases=len(rows), status="NOT_RUN", output=str(a.output))))
        return 0
    identity = dict(
        head=subprocess.check_output(
            ["git", "-c", "safe.directory=" + str(ROOT), "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        tracked_diff_sha256=hashlib.sha256(
            subprocess.check_output(["git", "-c", "safe.directory=" + str(ROOT), "-C", str(ROOT), "diff", "HEAD"])
        ).hexdigest(),
        script_sha256=sha(__file__),
        physical_npu=a.backend == "native" and a.action == "run",
    )
    (a.output / "source.json").write_text(json.dumps(identity, indent=2) + "\n")
    cache = {}
    manager = lease(a.backend, a.device) if a.action == "run" else contextlib.nullcontext()
    failed = False
    with manager:
        for row in rows:
            if a.case and row["case"] not in a.case:
                continue
            key = (row["history"], row["miss_percent"], a.heads)
            if key not in cache:
                folder = a.output / f"fixture-h{row['history_percent']}-m{row['miss_percent']}"
                cache[key] = (folder, prepare(row, folder, a.heads))
            folder, proof = cache[key]
            row["fixture_proof"] = proof
            row["status"] = "PASSED_CPU"
            if a.action == "run":
                run = a.output / row["case"]
                run.mkdir()
                if a.backend == "model":
                    command = [
                        str(a.model_driver.resolve()),
                        str((folder / "score.bin").resolve()),
                        str((run / "oracle.json").resolve()),
                        str(row["batch"]),
                        str(row["history"]),
                        str(row["valid_history"]),
                        str(row["miss_per_query"]),
                        str(a.shards),
                        str(v),
                    ]
                    row["driver_sha256"] = sha(a.model_driver)
                    row["scope"] = "AIV-only TopK component; no QK/Fetch or physical NPU"
                else:
                    command = [
                        sys.executable,
                        str(ROOT / "tests/operators/lifetch/test_lifetch.py"),
                        "--transport",
                        a.transport,
                        "--device",
                        str(a.device),
                        "--kvaf-server",
                        a.server,
                        "--batches",
                        str(row["batch"]),
                        "--history",
                        str(row["history"]),
                        "--valid-history",
                        str(row["valid_history"]),
                        "--exact-miss",
                        "--miss-percent",
                        str(row["miss_percent"]),
                        "--resident",
                        "16384",
                        "--index-heads",
                        str(a.heads),
                        "--aic-budget",
                        str(c),
                        "--aiv-budget",
                        str(v),
                        "--q-workers",
                        str(a.q_workers),
                        "--warmup",
                        str(a.warmup),
                        "--iterations",
                        str(a.iterations),
                        "--comparison-output",
                        str(run / "checked-output.npz"),
                        "--output",
                        str(run),
                        "--timeout-seconds",
                        str(a.timeout),
                    ]
                    command.extend(["--soc", a.soc, "--cann", a.cann, "--trace-rounds", "0"])
                    if a.kvaf_root:
                        command.extend(["--kvaf-root", str(a.kvaf_root)])
                    row["scope"] = "Current public streaming LI and native transport; requires Ascend/KVAF"
                row["command"] = command
                row["status"] = "RUNNING"
                save()
                try:
                    with (run / "worker.log").open("w") as log:
                        child = subprocess.Popen(
                            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                        )
                        try:
                            code = child.wait(timeout=a.timeout)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGTERM)
                            try:
                                child.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid, signal.SIGKILL)
                                child.wait()
                            raise
                    if code:
                        raise RuntimeError(f"worker exit {code}")
                    if a.backend == "model":
                        text = (run / "worker.log").read_text()
                        import re

                        ticks = re.findall(r"Total tick:\s*(\d+)", text)
                        assert len(ticks) == 1 and "Model stopped successfully." in text
                        oracle = json.loads((run / "oracle.json").read_text())
                        assert oracle["status"] == "passed" and oracle["guards"]
                        row["metric"] = int(ticks[0])
                        row["status"] = "PASSED_MODEL"
                    else:
                        import numpy as np

                        snapshot = run / "checked-output.npz"
                        with np.load(snapshot) as captured:
                            ids = captured["ids"].reshape(row["batch"], 4, 2048)
                            assert ((ids >= 0) & (ids < row["valid_history"])).all()
                            assert all(len(set(q.tolist())) == 2048 for b in ids for q in b)
                            assert captured["payload"].size > 0
                        row["output_sha256"] = sha(snapshot)
                        row["status"] = "PASSED_NPU"
                    row["log_sha256"] = sha(run / "worker.log")
                except Exception as error:
                    row["status"] = "FAILED"
                    row["error"] = str(error)
                    failed = True
            save()
    print(json.dumps(dict(cases=len(rows), failed=failed, output=str(a.output))))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
