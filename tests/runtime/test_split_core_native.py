"""CPU-only checks for grouped native commands and trustworthy event metrics."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("native_benchmark", ROOT / "experiments/split_core/native_benchmark.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def expected(batch=1, miss=10):
    row = next(c for c in bench.matrix.cases() if (c["batch"], c["miss_percent"]) == (batch, miss))
    return {batch: row}


def ubmem_log(path, *, batch=1, m=205, p50="1500.0", samples=20):
    path.write_text(
        "UBMEM_RUNTIME_BINARY path=/not-used-in-cpu-test/lib.so sha256=" + "a" * 64 + "\n"
        f"UBMEM_FETCH_COUNTS actual_misses_per_query={[m] * 4} Q4_union_per_request={2 * m} fixture=seed\n"
        f"UBMEM_GRAPH_EVENT_US p50={p50} p99=2500.0 iterations={samples} includes=init,LI\n"
        f"UBMEM_LIFETCH_PASS device=0 batch={batch} heads=32 history=131072 resident=16384 "
        f"token_bytes=656 requested_miss_pct=10 Q4_unique_misses={2 * m} prefetch_tokens=512 put=checked\n"
    )
    return path


def test_full_plan_groups_thirty_supported_rows_in_six_children(tmp_path):
    output = tmp_path / "plan"
    subprocess.run(
        ["bash", str(ROOT / "scripts/run_split_core_native.sh"), "plan", "--output", str(output)],
        check=True,
        capture_output=True,
        env={"KVIP_PYTHON": sys.executable},
    )
    rows = json.loads((output / "results.json").read_text())
    assert len(rows) == 42
    assert sum(r["status"] == "NOT_RUN" for r in rows) == 30
    assert sum(r["status"] == "UNSUPPORTED_NATIVE" for r in rows) == 12
    assert all(r["p50_ms"] is None for r in rows)
    commands = json.loads((output / "commands.json").read_text())
    assert len(commands) == 6 and sum(len(g["cases"]) for g in commands) == 30
    for group in commands:
        command = group["command"]
        assert command[command.index("--batches") + 1] == "1,2,4,8,16"
        assert command[command.index("--trace-rounds") + 1] == "0"
        assert "--comparison-output" not in command
    assert json.loads((output / "coverage.json").read_text())["policy_capabilities"]["serial6"].startswith(
        "no compatible"
    )


def test_selected_case_retains_all_rows_and_rejects_fresh_output_reuse(tmp_path):
    output = tmp_path / "plan"
    args = ["plan", "--case", "b16-h100-m50", "--output", str(output)]
    assert bench.main(args) == 0
    groups = json.loads((output / "commands.json").read_text())
    assert len(groups) == 1 and groups[0]["cases"] == ["b16-h100-m50"]
    with pytest.raises(FileExistsError):
        bench.main(args)


@pytest.mark.parametrize(
    "args",
    [
        ["run"],
        ["plan", "--case", "b24-h100-m10"],
        ["plan", "--case", "b16-h100-m10", "--q-workers", "4", "--budget", "28/56"],
    ],
)
def test_invalid_execution_rejected_before_output(tmp_path, args):
    output = tmp_path / "absent"
    with pytest.raises(SystemExit):
        bench.main([*args, "--output", str(output)])
    assert not output.exists()


def test_ubmem_converts_microseconds_to_milliseconds(tmp_path):
    log = ubmem_log(tmp_path / "worker.log")
    row = bench.parse_ubmem(log, expected(), 32, 20, 0)[1]
    assert (row["p50_ms"], row["p99_ms"], row["samples"]) == (1.5, 2.5, 20)
    assert row["actual_input_sha256"] is None
    assert row["report_sha256"] == bench.sha(log)


@pytest.mark.parametrize("kwargs", [{"m": 204}, {"p50": "nan"}, {"samples": 19}, {"batch": 2}])
def test_ubmem_rejects_mismatched_or_nonfinite_results(tmp_path, kwargs):
    with pytest.raises(ValueError):
        bench.parse_ubmem(ubmem_log(tmp_path / "worker.log", **kwargs), expected(), 32, 20, 0)


def test_ubmem_rejects_duplicate_or_incomplete_timing(tmp_path):
    log = ubmem_log(tmp_path / "worker.log")
    data = log.read_text()
    timing = next(s for s in data.splitlines() if s.startswith("UBMEM_GRAPH"))
    log.write_text(data.replace(timing, timing + "\n" + timing))
    with pytest.raises(ValueError):
        bench.parse_ubmem(log, expected(), 32, 20, 0)
    log.write_text(data.replace(timing, ""))
    with pytest.raises(ValueError):
        bench.parse_ubmem(log, expected(), 32, 20, 0)


def test_urma_uses_actual_child_report_and_fixture(tmp_path):
    child = tmp_path / "fresh-run"
    folder = child / "case0-b1-prototype-graph"
    fixture = folder / "fixture"
    fixture.mkdir(parents=True)
    for i in range(14):
        (fixture / f"arg{i}.bin").write_bytes(b"child seed input")
    prepared = folder / "input"
    prepared.mkdir()
    for i in range(14):
        (prepared / f"arg{i}.bin").write_bytes(b"prepared CPU file")
    (prepared / "native_weights_bf16.bin").write_bytes(b"BF16 weights")
    library = child / "universal-build/libv9_bridge.so"
    library.parent.mkdir()
    library.write_bytes(b"actual binary")
    report = dict(
        status="pass",
        fixture=dict(
            shape_history=131072,
            valid_history=131072,
            index_heads=32,
            requested_miss_percent=10,
            exact_miss=True,
            final_misses_per_query=[205] * 4,
            unique_misses_per_request=410,
        ),
        timing=dict(
            prototype_with_update=dict(p50_us=1200, p99_us=1800, samples=20),
            native_chain=dict(p50_us=2400, p99_us=3000, samples=20),
        ),
        native_misses_per_request=[410],
        resources={"effective_aic": 28},
        resource_geometry={"intermediate_fetches": 2},
    )
    path = folder / "report.json"
    (fixture / "fixture.json").write_text(
        json.dumps(dict(history=131072, resident=16384, queries=4, batch=1, full_chain_fixture=report["fixture"]))
    )
    path.write_text(json.dumps(report))
    row = bench.parse_urma(tmp_path, expected(), 32, 20)[1]
    assert row["p50_ms"] == 1.2 and row["native_reference_p50_ms"] == 2.4
    assert row["actual_input_sha256"] is None
    assert row["fixture_seed_sha256"]["arg0.bin"] == bench.sha(fixture / "arg0.bin")
    assert "native_weights_bf16.bin" in row["prepared_input_sha256"]
    assert row["resource_geometry"]["intermediate_fetches"] == 2
    (prepared / "arg13.bin").unlink()
    with pytest.raises(ValueError, match="missing actual"):
        bench.parse_urma(tmp_path, expected(), 32, 20)
    (prepared / "arg13.bin").write_bytes(b"restored")
    report["native_misses_per_request"] = [409]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="fixture differs"):
        bench.parse_urma(tmp_path, expected(), 32, 20)
    report["native_misses_per_request"] = [410]
    report["fixture"]["final_misses_per_query"] = [204] * 4
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="fixture differs"):
        bench.parse_urma(tmp_path, expected(), 32, 20)


def test_csv_keeps_metrics_from_nonfirst_row(tmp_path):
    bench.save(tmp_path, [{"case": "first", "p50_ms": None}, {"case": "later", "p50_ms": 1.2, "report_sha256": "abc"}])
    assert "report_sha256" in (tmp_path / "results.csv").read_text()
    assert "abc" in (tmp_path / "results.csv").read_text()


@pytest.mark.parametrize(
    "value",
    [
        dict(p50_us=True, p99_us=2, samples=20),
        dict(p50_us=1, p99_us=2, samples=20.0),
        dict(p50_us=1, p99_us=float("inf"), samples=20),
    ],
)
def test_stats_rejects_invalid_types_and_nonfinite(value):
    with pytest.raises(ValueError):
        bench.stats(value, 20)


def test_launch_has_single_timeout_owner_and_forwards_interrupt(tmp_path, monkeypatch):
    calls, signals = [], []

    class Child:
        pid = 12345

        def wait(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise KeyboardInterrupt
            return 0

    monkeypatch.setattr(bench.subprocess, "Popen", lambda *args, **kwargs: Child())
    monkeypatch.setattr(bench.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(KeyboardInterrupt):
        bench.launch(["CPU_TEST_ONLY"], tmp_path / "child.log")
    assert calls == [{}, {"timeout": 10}]
    assert signals == [(12345, bench.signal.SIGINT)]


def test_lease_retained_on_unknown_completion_and_not_stolen(tmp_path):
    lock = tmp_path / "native.lease"
    with bench.native_lease(lock):
        with pytest.raises(FileExistsError):
            with bench.native_lease(lock):
                pass
    assert not lock.exists()
    with bench.native_lease(lock) as state:
        state["release"] = False
    assert lock.exists()
    lock.unlink()
    with pytest.raises(KeyboardInterrupt):
        with bench.native_lease(lock):
            raise KeyboardInterrupt
    assert lock.exists()


def test_group_failure_stops_matrix_and_retains_device_lease(tmp_path, monkeypatch):
    calls = []

    def failed(command, log):
        calls.append(command)
        log.touch()
        return 124

    monkeypatch.setattr(bench, "launch", failed)
    output, lock = tmp_path / "run", tmp_path / "device.lease"
    assert bench.main(["run", "--soc", "CPU_TEST_ONLY", "--lease-file", str(lock), "--output", str(output)]) == 1
    rows = json.loads((output / "results.json").read_text())
    assert len(calls) == 1 and lock.exists()
    assert sum(r["status"] == "FAILED" for r in rows) == 5
    assert sum(r["status"] == "NOT_RUN" for r in rows) == 25
    assert all(r["p50_ms"] is None for r in rows)
