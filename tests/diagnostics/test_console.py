# SPDX-License-Identifier: Apache-2.0
"""Subprocess I/O and saved JSON only; these fixtures are not timing evidence."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def execute(source, tmp_path):
    return subprocess.run(
        [sys.executable, "-c", source],
        env={**os.environ, "PYTHONPATH": str(ROOT / "python"), "TMPDIR": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=30,
    )


def saved_case():
    return dict(
        status="pass",
        transport="ubmem",
        mode="fused",
        geometry={"batch": 16},
        timing=dict(mean_us=10.0, p50_us=9.0),
        tuning={"cold_only_prefetch": True},
        stage_diagnostics={"stages": {"init": {"p50_us": 1.25}}},
        reads_samples=[
            dict(
                prefetch=4,
                final=2,
                unused_prefetch=1,
                bytes=6 * 2624,
                prefetch_cold=3,
                prefetch_candidate_hot=1,
                unused_prefetch_cold=0,
                unused_prefetch_candidate_hot=1,
                prefetch_unclassified=0,
            )
        ],
    )


def test_quiet_captures_native_and_buffered_c_stdio_but_keeps_progress(tmp_path):
    result = execute(
        r"""
import ctypes, os
from kvip.diagnostics.console import NativeLog, emit
with NativeLog(True):
    emit("CASE_BEGIN real-case")
    os.write(1,b"native stdout\n")
    os.write(2,b"native stderr\n")
    ctypes.CDLL(None).printf(b"native buffered printf\n")
    print("ordinary python chatter")
    emit("CASE_PASS real-case")
print("restored stdout")
""",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0].startswith("RUN_LOG ") and lines[0].endswith("(temporary; removed on success)")
    assert lines[1:] == ["CASE_BEGIN real-case", "CASE_PASS real-case", "restored stdout"]
    assert not result.stderr
    assert not list(tmp_path.glob("*.log"))


def test_quiet_announces_retained_log_before_native_hard_exit(tmp_path):
    result = execute(
        """
import os
from kvip.diagnostics.console import NativeLog
with NativeLog(True):
    os.write(2, b"native fatal evidence" + bytes([10]))
    os._exit(23)
""",
        tmp_path,
    )
    assert result.returncode == 23
    logs = list(tmp_path.glob("kvip-replay-*.log"))
    assert len(logs) == 1 and str(logs[0].resolve()) in result.stdout
    assert "native fatal evidence" in logs[0].read_text()
    assert not result.stderr  # No Python cleanup can run after a native hard exit.


def test_replay_failure_restores_stderr_saves_full_log_before_retained_owner_exit(tmp_path):
    output, cleanup = tmp_path / "failed.json", tmp_path / "unsafe-cleanup"
    result = execute(
        f"""
import atexit, os, sys
from pathlib import Path
from kvip.diagnostics import replay
atexit.register(lambda: Path({str(cleanup)!r}).write_text("must not run"))
replay.load_snapshot=lambda path: object()
replay.inspect_snapshot=lambda archive: {{"status":"verified_snapshot"}}
def fail(args, archive):
    replay._RETAINED_RUNS.append({{"unsafe_native_owner": object()}})
    for n in range(100): os.write(2, ("native-line-%03d\\n" % n).encode())
    raise RuntimeError("actual CQ timeout")
replay.run=fail
sys.argv=["replay", "--snapshot", "/snapshot", "--build-dir", "/build", "--soc", "soc", "--quiet", "--output", {str(output)!r}]
replay.main()
""",
        tmp_path,
    )
    assert result.returncode == 70
    assert "CASE_BEGIN ubmem-fused" in result.stdout
    assert "CASE_PASS" not in result.stdout
    assert "actual CQ timeout" in result.stderr and "Failure log:" in result.stderr
    assert "native-line-099" in result.stderr and "native-line-000" not in result.stderr
    saved = json.loads(output.read_text())
    log = Path(saved["failure_log"])
    assert log.exists() and "native-line-000" in log.read_text()
    assert not cleanup.exists()  # Unknown completion still uses os._exit, not Python cleanup.


@pytest.mark.parametrize("module", ("replay", "compare"))
@pytest.mark.parametrize("kind", ("single", "paired", "comparison"))
def test_report_only_reads_saved_results_without_runtime_or_new_files(tmp_path, module, kind):
    case = saved_case()
    if kind == "single":
        report = case
    elif kind == "paired":
        report = dict(
            status="pass", geometry=case["geometry"], cases=[dict(name="before", **case), dict(name="after", **case)]
        )
    else:
        report = dict(
            status="pass",
            geometry=case["geometry"],
            table={
                "ubmem-fused": dict(
                    **case["timing"],
                    tuning=case["tuning"],
                    reads_per_round=case["reads_samples"][0],
                )
            },
            stage_diagnostics={"ubmem-fused": case["stage_diagnostics"]},
            fusion_vs_serial={},
        )
    path = tmp_path / "result.json"
    path.write_text(json.dumps(report))
    result = execute(
        f"""
import sys
from kvip.diagnostics import {module} as cli
def forbidden(*args, **kwargs): raise AssertionError("report-only entered device work")
cli.run=forbidden
if hasattr(cli, "subprocess"): cli.subprocess.run=forbidden
sys.argv=["cli", "--report-only", {str(path)!r}, "--quiet"]
cli.main()
assert "torch_npu" not in sys.modules and "kvaf" not in sys.modules
""",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "CORRECTNESS_PASS" in result.stdout
    assert "stage p50 us (instrumented): init=1.250" in result.stdout
    assert "cold=3/0" in result.stdout and "candidate_hot=1/1" in result.stdout
    assert sorted(p.name for p in tmp_path.iterdir()) == ["result.json"]


@pytest.mark.parametrize("module", ("replay", "compare"))
def test_report_only_rejects_execution_options(module, tmp_path):
    if module == "replay":
        from kvip.diagnostics.replay import arguments
    else:
        from kvip.diagnostics.compare import arguments
    assert arguments(["--report-only", str(tmp_path / "result.json")]).report_only
    with pytest.raises(SystemExit):
        arguments(["--report-only", "/saved.json", "--snapshot", "/snapshot"])


def test_compare_keeps_failed_child_native_log_and_reports_exit(tmp_path):
    output = tmp_path / "result.json"
    result = execute(
        f"""
import sys
from pathlib import Path
from types import SimpleNamespace
from kvip.diagnostics import compare
args=SimpleNamespace(report_only=None,result_dir=Path({str(tmp_path / "results")!r}),quiet=True,device=4,warmup=2,iterations=10,stage_iterations=3)
compare.arguments=lambda:args
compare.case_plan=lambda args:[("urma","fused",None)]
compare.command=lambda *args:([sys.executable,"-c","import os;os.write(2,b'native fatal evidence'+bytes([10]));os._exit(70)"],Path({str(output)!r}))
compare.main()
""",
        tmp_path,
    )
    assert result.returncode == 1
    assert "CASE_BEGIN result" in result.stdout and "CASE_PASS" not in result.stdout
    assert "exited 70" in result.stderr and "native fatal evidence" in result.stderr
    logs = list(tmp_path.glob("kvip-compare-*.log"))
    assert len(logs) == 1 and "native fatal evidence" in logs[0].read_text()


def test_compare_success_removes_spools_and_keeps_case_jsons(tmp_path, monkeypatch, capsys):
    from kvip.diagnostics import compare
    from types import SimpleNamespace
    import tempfile

    spool = tmp_path / "spools"
    spool.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(spool))
    destination = tmp_path / "results"
    args = compare.arguments(
        [
            "--snapshot",
            "/snapshot",
            "--ubmem-build-dir",
            "/ub",
            "--urma-build-dir",
            "/ur",
            "--soc",
            "soc",
            "--kvaf-server",
            "127.0.0.1:28880",
            "--result-dir",
            str(destination),
            "--quiet",
        ]
    )
    monkeypatch.setattr(compare, "arguments", lambda: args)

    def child(argv, *, stdout, stderr, check):
        transport, mode = (argv[argv.index("--" + name) + 1] for name in ("transport", "mode"))
        output = Path(argv[argv.index("--output") + 1])
        case = dict(
            saved_case(),
            transport=transport,
            mode=mode,
            snapshot_sha256="one",
            scope="one-group",
            soc="soc",
            device=0,
            kvip_commit="one",
            runtime_workers=2,
        )
        output.write_text(json.dumps(case))
        stdout.write("native setup chatter\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(compare.subprocess, "run", child)
    compare.main()
    text = capsys.readouterr().out
    assert text.count("CASE_BEGIN") == text.count("CASE_PASS") == 4
    assert "native setup chatter" not in text
    assert "cold=3/0" in text and "stage p50 us" in text
    assert not list(spool.iterdir())
    assert len(list(destination.glob("*.json"))) == 5


def test_compact_report_keeps_host_boundary_separate_and_skips_different_work_ratios(capsys):
    from kvip.diagnostics.console import compact_report

    before = dict(saved_case(), name="before", replay_selection={"active": 1})
    after = dict(saved_case(), name="after", replay_selection={"active": 16})
    after["candidate_call"] = dict(
        wall={"mean_us": 30.0, "p50_us": 29.0},
        cpu_prepare={"p50_us": 2.0},
        h2d_diagnostic={"p50_us": 3.0},
    )
    compact_report(dict(status="pass", cases=[before, after]))
    text = capsys.readouterr().out
    assert "device graph mean=10.000" in text
    assert "separate candidate-to-output wall: mean=30.000" in text
    assert "H2D diagnostic p50=3.000" in text
    assert "after vs before" not in text


@pytest.mark.parametrize(
    "report", ({"status": "pass", "cases": []}, {"status": "pass", "timing": {"mean_us": 0, "p50_us": 0}})
)
def test_report_only_rejects_invalid_saved_results(tmp_path, report, capsys):
    from kvip.diagnostics.console import report_only

    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(report))
    with pytest.raises(SystemExit, match="2"):
        report_only(path)
    assert "FAILED reading report" in capsys.readouterr().err


@pytest.mark.parametrize("module", ("replay", "compare"))
def test_report_only_preserves_failed_status_in_exit_code(tmp_path, module):
    path = tmp_path / "failed.json"
    path.write_text(json.dumps(dict(status="failed", error="CQ timeout", failure_log="/tmp/retained.log")))
    result = execute(
        f"""
import sys
from kvip.diagnostics import {module}
sys.argv=["cli", "--report-only", {str(path)!r}]
{module}.main()
""",
        tmp_path,
    )
    assert result.returncode == 1
    assert "FAILED: CQ timeout" in result.stderr
    assert "Failure log: /tmp/retained.log" in result.stderr
    assert "PASS" not in result.stdout


def payload_case():
    mib = 1 << 20
    return dict(
        saved_case(),
        mode="serial",
        geometry={"token_bytes": 1024},
        timing={"mean_us": 200.0, "p50_us": 100.0},
        reads_samples=[
            dict(bytes=3 * mib, useful_transfer_bytes=2 * mib, cold_demand=512),
            dict(bytes=5 * mib + 1, useful_transfer_bytes=4 * mib, cold_demand=1536),
        ],
        stage_diagnostics={
            "stages": {"batch_get": {"mean_us": 50.0, "p50_us": 25.0}},
            "reads_samples": [{"bytes": mib}, {"bytes": 3 * mib}],
        },
    )


@pytest.mark.parametrize("variants", (False, True))
def test_payload_rates_use_matching_mean_bytes_and_mean_latency(capsys, variants):
    from kvip.diagnostics.console import compact_report

    case = payload_case()
    report = dict(status="pass", cases=[dict(name="before", **case)]) if variants else case
    compact_report(report)
    text = capsys.readouterr().out
    # Actual includes a wasted MiB; useful includes two MiB beyond cold demand.
    assert "bytes=4,194,304.5" in text
    assert "actual=4.000 MiB @ 20.972 GB/s; useful=3.000 MiB @ 15.729 GB/s; cold=1.000 MiB @ 5.243 GB/s" in text
    assert "payload / graph mean (decimal GB/s, not link peak)" in text
    # Stage bytes differ from full-graph bytes. Neither full bytes nor p50 is its denominator/numerator.
    assert (
        "payload / Get stage mean (instrumented) (decimal GB/s, not link peak): actual=2.000 MiB @ 41.943 GB/s" in text
    )


def test_compare_payload_uses_saved_normalized_metrics_and_available_bytes(capsys):
    from kvip.diagnostics.console import compact_report

    mib = 1 << 20
    compact_report(
        dict(
            status="pass",
            table={
                "ubmem-fused": dict(
                    mean_us=200.0,
                    p50_us=100.0,
                    mean_read_bytes=4 * mib,
                    mean_useful_transfer_bytes=3 * mib,
                    demand_per_round={"cold_payload_bytes": mib},
                    graph_normalized_payload_GBps={
                        "actual_transfers": 4 * mib / 200000,
                        "useful_transfers": 3 * mib / 200000,
                        "cold_demand": mib / 200000,
                    },
                )
            },
        )
    )
    text = capsys.readouterr().out
    assert "bytes=4,194,304" in text
    assert "actual=4.000 MiB @ 20.972 GB/s; useful=3.000 MiB @ 15.729 GB/s; cold=1.000 MiB @ 5.243 GB/s" in text
    assert "Get stage mean" not in text


def test_get_payload_is_not_inferred_from_uninstrumented_samples(capsys):
    from kvip.diagnostics.console import compact_report

    case = payload_case()
    del case["stage_diagnostics"]["reads_samples"]
    compact_report(case)
    text = capsys.readouterr().out
    assert "payload / graph mean" in text and "stage p50 us" in text
    assert "Get stage mean" not in text
