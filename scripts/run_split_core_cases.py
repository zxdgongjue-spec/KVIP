#!/usr/bin/env python3
"""Plan or execute the fixed128K/Q4 42-case smoke/customer matrix."""

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


MATRIX_VERSION = "128k-q4-v2"
BATCHES = (1, 2, 4, 8, 16, 24, 32)
MISSES = (3, 5, 10, 15, 20, 50)


def cases(group="all"):
    batches = {"small": (1, 2), "medium": (4, 8), "large": (16, 24, 32), "all": BATCHES}[group]
    for batch, miss in itertools.product(batches, MISSES):
        count = round(2048 * miss / 100)
        yield dict(
            case=f"b{batch}-h100-m{miss}",
            batch=batch,
            queries_per_request=4,
            total_queries=4 * batch,
            history_percent=100,
            valid_history=131072,
            history=131072,
            resident=16384,
            miss_percent=miss,
            miss_per_query=count,
            actual_miss_percent=100 * count / 2048,
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
def lease(backend, device, lease_file=None):
    path = Path(
        lease_file or ("/tmp/kvip-cann91-simulator.lease" if backend == "model" else f"/tmp/kvip-npu-{device}.lease")
    )
    owner = dict(task="split-core-matrix", pid=os.getpid(), lease=uuid.uuid4().hex)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(owner, f)
    try:
        yield
    finally:
        if path.exists() and json.loads(path.read_text()).get("lease") == owner["lease"]:
            path.unlink()


def driver_capabilities(driver):
    # This metadata query does not call ACL. Avoid Model preload constructors;
    # some SDKs still print loader notices, so require one capability JSON row.
    env = dict(os.environ)
    env.pop("LD_PRELOAD", None)
    result = subprocess.run(
        [str(driver.resolve()), "--capabilities"], check=True, text=True, capture_output=True, timeout=30, env=env
    )
    candidates = []
    for line in result.stdout.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and value.get("scope") == "AIV-only TopK":
            candidates.append(value)
    if len(candidates) != 1:
        raise ValueError("expected exactly one TopK capability JSON record")
    return candidates[0]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("action", choices=("plan", "cpu", "run"))
    p.add_argument(
        "--suite",
        choices=("smoke", "customer"),
        default="customer",
        help="Both suites cover all 42 cases; no silent smoke subset",
    )
    p.add_argument(
        "--lease-file", type=Path, help="Shared exclusive lease path; mount the same path across Model containers"
    )
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
    p.add_argument("--warmup", type=int, default=4, help="Native only; Model runs once without warmup")
    p.add_argument("--iterations", type=int, default=20, help="Native only; Model runs once per condition")
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
    selected = [x for x in cases(a.group) if not a.case or x["case"] in a.case]
    if a.action == "run" and a.backend == "native" and any(x["batch"] > 16 for x in selected):
        p.error(
            "BS24/32 are TopK component Model only; current native streaming supports BS1..16. Select supported --case or --group."
        )
    capabilities = None
    if a.action == "run" and a.backend == "model":
        try:
            capabilities = driver_capabilities(a.model_driver)
            assert capabilities["queries_per_request"] == 4
            assert capabilities["max_batch"] >= max(x["batch"] for x in selected)
            assert capabilities["scope"] == "AIV-only TopK"
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, AssertionError) as error:
            p.error(f"Model driver capability check failed; rebuild current csrc/operators/split_topk target: {error}")
    a.output.mkdir(parents=True, exist_ok=False)
    c, v = map(int, a.budget.split("/"))
    rows = [
        {
            **x,
            "suite": a.suite,
            "matrix_version": MATRIX_VERSION,
            "topk_shards": a.shards if a.backend == "model" else None,
            "q_workers": a.q_workers if a.backend == "native" else None,
            "warmup": 0 if a.backend == "model" else a.warmup,
            "iterations": 1 if a.backend == "model" else a.iterations,
            "aic_workers": 0 if a.backend == "model" else None,
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
        matrix_version=MATRIX_VERSION,
        suite=a.suite,
        topk_shards=a.shards if a.backend == "model" else None,
        model_driver_capabilities=capabilities,
        physical_npu=a.backend == "native" and a.action == "run",
    )
    (a.output / "source.json").write_text(json.dumps(identity, indent=2) + "\n")
    cache = {}
    manager = lease(a.backend, a.device, a.lease_file) if a.action == "run" else contextlib.nullcontext()
    failed = False
    with manager:
        for row in rows:
            if a.case and row["case"] not in a.case:
                continue
            key = (row["history"], row["miss_percent"], a.heads)
            try:
                if key not in cache:
                    folder = a.output / f"fixture-h{row['history_percent']}-m{row['miss_percent']}"
                    cache[key] = (folder, prepare(row, folder, a.heads))
                folder, proof = cache[key]
            except Exception as error:
                row["status"] = "FAILED"
                row["error"] = f"fixture preparation: {error}"
                failed = True
                save()
                continue
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
                        assert oracle["batch"] == row["batch"] and oracle["shards"] == a.shards
                        assert oracle["history"] == row["history"] and oracle["valid"] == row["valid_history"]
                        assert oracle["aiv_budget"] == v and oracle["aic_workers"] == 0
                        assert oracle["miss_per_query"] == row["miss_per_query"]
                        assert all(
                            oracle[k] for k in ("selected_set_and_scores", "all_local_candidates", "input_unchanged")
                        )
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
