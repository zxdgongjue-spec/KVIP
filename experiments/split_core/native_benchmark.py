#!/usr/bin/env python3
"""Plan/run the current native streaming matrix, grouped by miss to reuse builds.

Not a serial6/Fusion3 runner. Child harnesses own timeouts and numerical checks.
Uses fresh output directories; no implicit resume or cross-run build cache.
"""

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("customer_cases", ROOT / "scripts/run_split_core_cases.py")
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(value, iterations):
    p50, p99, count = value["p50_us"], value["p99_us"], value["samples"]
    if (
        type(count) is not int
        or isinstance(p50, bool)
        or isinstance(p99, bool)
        or count != iterations
        or not math.isfinite(p50)
        or not math.isfinite(p99)
        or p50 <= 0
        or p99 < p50
    ):
        raise ValueError("invalid/incorrect-sample-count native timing")
    return dict(p50_ms=p50 / 1000, p99_ms=p99 / 1000, samples=count)


def parse_urma(group, expected, heads, iterations):
    """Only fresh child reports, with fixture geometry and oracle metadata."""
    found = {}
    for path in group.glob("*/case*-b*-prototype-graph/report.json"):
        match = re.fullmatch(r"case\d+-b(\d+)-prototype-graph", path.parent.name)
        b = int(match[1])
        if b not in expected or b in found:
            raise ValueError("unexpected/duplicate URMA batch report")
        row = expected[b]
        report = json.loads(path.read_text())
        fixture = report["fixture"]
        if (
            report["status"] != "pass"
            or fixture["shape_history"] != row["history"]
            or fixture["valid_history"] != row["valid_history"]
            or fixture["index_heads"] != heads
            or fixture["requested_miss_percent"] != row["miss_percent"]
            or not fixture["exact_miss"]
            or fixture["final_misses_per_query"] != [row["miss_per_query"]] * 4
            or fixture["unique_misses_per_request"] != 2 * row["miss_per_query"]
            or report["native_misses_per_request"] != [2 * row["miss_per_query"]] * b
        ):
            raise ValueError("URMA report/fixture differs from requested case")
        library = path.parent.parent / "universal-build/libv9_bridge.so"
        inputs = [path.parent / "fixture" / f"arg{i}.bin" for i in range(14)]
        config = path.parent / "fixture/fixture.json"
        prepared = [path.parent / "input" / f"arg{i}.bin" for i in range(14)]
        prepared.append(path.parent / "input/native_weights_bf16.bin")
        if not all(p.is_file() for p in [library, config, *inputs, *prepared]):
            raise ValueError("missing actual URMA binary/fixture evidence")
        seed = json.loads(config.read_text())
        if (
            any(
                seed[k] != v
                for k, v in dict(history=row["history"], resident=row["resident"], queries=4, batch=1).items()
            )
            or seed["full_chain_fixture"] != fixture
        ):
            raise ValueError("URMA fixture file/report mismatch")
        inputs.append(config)
        found[b] = {
            **stats(report["timing"]["prototype_with_update"], iterations),
            "native_reference_p50_ms": stats(report["timing"]["native_chain"], iterations)["p50_ms"],
            "binary_sha256": sha(library),
            "report_path": str(path.relative_to(group)),
            "report_sha256": sha(path),
            "fixture_seed_sha256": {p.name: sha(p) for p in inputs},
            "prepared_input_sha256": {p.name: sha(p) for p in prepared},
            "actual_input_sha256": None,  # Device-side transforms are not dumped by this harness.
            "resources": report["resources"],
            "resource_geometry": report["resource_geometry"],
            "metric_scope": "current streaming graph event; restore/verification/SFA excluded",
        }
    if set(found) != set(expected):
        raise ValueError("missing URMA batch report")
    return found


def parse_ubmem(log, expected, heads, iterations, device):
    """Parse the original worker once, never its duplicated console summary."""
    found, timing, counts, binary = {}, None, None, None
    for line in log.read_text().splitlines():
        if line.startswith("UBMEM_RUNTIME_BINARY "):
            matches = re.findall(r"sha256=([0-9a-f]{64})(?:\s|$)", line)
            if binary is not None or len(matches) != 1:
                raise ValueError("missing/duplicate UBmem binary identity")
            binary = matches[0]
        elif line.startswith("UBMEM_FETCH_COUNTS "):
            if counts is not None:
                raise ValueError("duplicate UBmem miss evidence")
            match = re.search(r"actual_misses_per_query=(\[[^\]]+\]) Q4_union_per_request=(\d+) ", line)
            if not match:
                raise ValueError("missing actual UBmem miss counts")
            counts = (json.loads(match[1]), int(match[2]))
        elif line.startswith("UBMEM_GRAPH_EVENT_US "):
            if timing is not None:
                raise ValueError("duplicate UBmem timing")
            match = re.match(r"UBMEM_GRAPH_EVENT_US p50=(\S+) p99=(\S+) iterations=(\d+) ", line)
            if not match:
                raise ValueError("malformed UBmem timing")
            timing = stats(dict(p50_us=float(match[1]), p99_us=float(match[2]), samples=int(match[3])), iterations)
        elif line.startswith("UBMEM_LIFETCH_PASS "):
            fields = dict(re.findall(r"(\w+)=(\S+)", line))
            b = int(fields["batch"])
            if b not in expected or b in found or timing is None or counts is None or binary is None:
                raise ValueError("unexpected/incomplete UBmem checked result")
            row = expected[b]
            exact = dict(
                device=device,
                heads=heads,
                history=row["history"],
                resident=row["resident"],
                token_bytes=656,
                Q4_unique_misses=2 * row["miss_per_query"],
                prefetch_tokens=512,
            )
            if (
                any(int(fields[k]) != v for k, v in exact.items())
                or float(fields["requested_miss_pct"]) != row["miss_percent"]
                or counts != ([row["miss_per_query"]] * 4, 2 * row["miss_per_query"])
            ):
                raise ValueError("UBmem checked geometry/miss differs from requested case")
            found[b] = {
                **timing,
                "binary_sha256": binary,
                "report_path": str(log.name),
                "report_sha256": sha(log),
                "actual_input_sha256": None,  # Existing child discards temporary input files.
                "metric_scope": "current streaming graph event; restore/verification/SFA excluded",
            }
            timing, counts = None, None
    if set(found) != set(expected) or timing is not None or counts is not None:
        raise ValueError("missing/incomplete UBmem batch result")
    return found


@contextlib.contextmanager
def native_lease(path):
    identity = dict(task="split-core-native-grouped", pid=os.getpid(), lease=uuid.uuid4().hex)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(identity, f)
    state = {"release": True}
    try:
        yield state
    except BaseException:
        state["release"] = False
        raise
    finally:
        if state["release"] and path.exists() and json.loads(path.read_text()).get("lease") == identity["lease"]:
            path.unlink()
        elif not state["release"]:
            print(
                f"Native completion/cleanup uncertain; lease retained: {path}. Verify owned workers have stopped before removing it.",
                file=sys.stderr,
            )


def launch(command, log):
    """No competing outer deadline: existing native harness owns its timeout.

    On interruption ask our process group to stop; nested sessions may survive,
    so caller retains the shared lease and stops the matrix on any failed child.
    """
    with log.open("w") as output:
        child = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return child.wait()
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGINT)
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            except ProcessLookupError:
                child.wait()
            raise


def save(output, rows):
    (output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (output / "results.csv").open("w", newline="") as f:
        # Union schema retains fields from a selected case beyond the first row.
        fields = list(
            dict.fromkeys(
                k
                for row in rows
                for k in row
                if k not in ("actual_input_sha256", "fixture_seed_sha256", "prepared_input_sha256")
            )
        )
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: json.dumps(row[k]) if isinstance(row.get(k), (dict, list)) else row.get(k) for k in fields}
            )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("action", choices=("plan", "run"))
    p.add_argument("--transport", choices=("urma", "ubmem"), default="ubmem")
    p.add_argument("--budget", choices=("28/56", "32/64"), default="28/56")
    p.add_argument("--heads", type=int, choices=(32, 64), default=32)
    p.add_argument("--q-workers", type=int, choices=(1, 2, 4), default=2)
    p.add_argument("--case", action="append")
    p.add_argument("--soc")
    p.add_argument("--cann", default="/usr/local/Ascend/cann")
    p.add_argument("--kvaf-root", type=Path)
    p.add_argument("--server", default="127.0.0.1:28880")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--timeout", type=int, default=10800, help="Child harness timeout; no competing parent deadline")
    p.add_argument("--lease-file", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    all_cases = list(matrix.cases())
    if a.case and not set(a.case) <= {r["case"] for r in all_cases}:
        p.error("unknown standard case")
    if a.warmup < 0 or a.iterations < 1 or a.timeout < 20 or a.device < 0:
        p.error("invalid warmup/iterations/timeout/device")
    if a.action == "run" and not a.soc:
        p.error("run requires actual --soc")
    c, v = map(int, a.budget.split("/"))
    if a.case and any(r["batch"] > 16 and r["case"] in a.case for r in all_cases):
        p.error("explicit BS24/32 cannot execute native streaming")
    rows = []
    for row in all_cases:
        supported = row["batch"] <= 16 and 3 * row["batch"] + 4 * a.q_workers <= v
        rows.append(
            {
                **row,
                "policy": "current-streaming",
                "backend": "native",
                "transport": a.transport,
                "aic_budget": c,
                "aiv_budget": v,
                "heads": a.heads,
                "token_bytes": 656,
                "q_workers": a.q_workers,
                "warmup": a.warmup,
                "iterations": a.iterations,
                "status": "NOT_RUN" if supported else "UNSUPPORTED_NATIVE",
                "reason": "" if supported else "current B<=16 and 3B+4W<=V contract",
                "p50_ms": None,
                "p99_ms": None,
                "samples": None,
                "native_reference_p50_ms": None,
                "binary_sha256": None,
                "report_path": None,
                "report_sha256": None,
                "actual_input_sha256": None,
                "metric_scope": None,
                "error": None,
            }
        )
    selected = [r for r in rows if r["status"] == "NOT_RUN" and (not a.case or r["case"] in a.case)]
    if a.case and set(a.case) != {r["case"] for r in selected}:
        p.error("explicit selected case exceeds native role budget")
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    groups = []
    for miss in matrix.MISSES:
        batch_rows = [r for r in selected if r["miss_percent"] == miss]
        if not batch_rows:
            continue
        directory = a.output / f"m{miss}"
        command = [
            sys.executable,
            str(ROOT / "tests/operators/lifetch/test_lifetch.py"),
            "--transport",
            a.transport,
            "--batches",
            ",".join(str(r["batch"]) for r in batch_rows),
            "--history",
            "131072",
            "--valid-history",
            "131072",
            "--resident",
            "16384",
            "--exact-miss",
            "--miss-percent",
            str(miss),
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
            "--trace-rounds",
            "0",
            "--timeout-seconds",
            str(a.timeout),
            "--device",
            str(a.device),
            "--kvaf-server",
            a.server,
            "--cann",
            a.cann,
            "--output",
            str(directory),
        ]
        if a.soc:
            command += ["--soc", a.soc]
        if a.kvaf_root:
            command += ["--kvaf-root", str(a.kvaf_root.resolve())]
        groups.append(dict(miss_percent=miss, cases=[r["case"] for r in batch_rows], command=command))
    (a.output / "commands.json").write_text(json.dumps(groups, indent=2) + "\n")
    coverage = dict(
        standard_cases=42,
        native_selected=len(selected),
        child_groups=len(groups),
        policy_capabilities={
            "current-streaming": "native BS<=16; actual run still requires hardware",
            "serial6": "no compatible 42-case full-chain runner",
            "fusion3": "historical fixed-point/mock evidence; no compatible 42-case runner",
            "proposed-po-owner-pool": "not implemented",
        },
    )
    (a.output / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    save(a.output, rows)
    if a.action == "plan":
        print(json.dumps(coverage))
        return 0
    tracked = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "ls-files",
            "--",
            "tests/operators/lifetch",
            "csrc/operators/lightning_indexer_with_fetch",
            "csrc/operators/lightning_indexer_with_fetch_ubmem",
            "csrc/operators/common/lifetch",
            "python/kvip/operators/lifetch",
            "python/kvip/operators/lifetch_ubmem",
            "python/kvip/runtime",
            "scripts/build.py",
            "scripts/env.sh",
        ],
        text=True,
    ).splitlines()
    paths = sorted({Path(__file__), ROOT / "scripts/run_split_core_cases.py", *[ROOT / p for p in tracked]}, key=str)
    (a.output / "source.json").write_text(
        json.dumps(
            dict(
                head=subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                source_sha256={str(path.relative_to(ROOT)): sha(path) for path in paths},
                scope="Native graph event timing, no hardware cache flush, no full-model TPOT",
            ),
            indent=2,
        )
        + "\n"
    )
    failed = False
    with native_lease(a.lease_file or Path(f"/tmp/kvip-npu-{a.device}.lease")) as lease:
        for group in groups:
            directory = a.output / f"m{group['miss_percent']}"
            directory.mkdir()
            batch_rows = [r for r in selected if r["case"] in group["cases"]]
            for row in batch_rows:
                row["status"] = "RUNNING"
            save(a.output, rows)
            print(
                f"RUN miss={group['miss_percent']} cases={len(batch_rows)} log={directory / 'console.log'}", flush=True
            )
            try:
                code = launch(group["command"], directory / "console.log")
                if code:
                    lease["release"] = False
                    raise RuntimeError(f"native child exit {code}; stop matrix, cleanup not independently proven")
                expected = {r["batch"]: r for r in batch_rows}
                if a.transport == "urma":
                    results = parse_urma(directory, expected, a.heads, a.iterations)
                else:
                    logs = list(directory.glob("*/worker.log"))
                    if len(logs) != 1:
                        raise ValueError("expected one fresh UBmem worker log")
                    results = parse_ubmem(logs[0], expected, a.heads, a.iterations, a.device)
                    for value in results.values():
                        value["report_path"] = str(logs[0].relative_to(directory))
                for row in batch_rows:
                    row.update(results[row["batch"]], status="PASSED_NPU")
                    row["report_path"] = f"m{group['miss_percent']}/" + row["report_path"]
            except Exception as error:
                for row in batch_rows:
                    row.update(status="FAILED", error=str(error))
                failed = True
                save(a.output, rows)
                break
            save(a.output, rows)
    print(
        json.dumps(
            dict(
                passed=sum(r["status"] == "PASSED_NPU" for r in rows),
                failed=failed,
                unsupported=sum(r["status"] == "UNSUPPORTED_NATIVE" for r in rows),
            )
        )
    )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
