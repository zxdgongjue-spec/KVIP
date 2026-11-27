# SPDX-License-Identifier: Apache-2.0
import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
import fixed_q4_a5_12 as runner
from fixed_q4_a5_12 import plan, summarize_cell, write_summary


@pytest.mark.parametrize("options, kind, group", [((), "real", 2), (("--synthetic",), "synthetic", 0)])
def test_dry_run_needs_no_a5_dependencies(options, kind, group):
    script = Path(__file__).resolve().parent / "support/fixed_q4_a5_12.py"
    result = subprocess.check_output([sys.executable, "-B", script, "--dry-run", *options], text=True)
    data = json.loads(result)
    assert data["batch_sizes"] == [1, 2, 4, 8, 16, 20, 24, 28, 32]
    assert data["group"] == group
    assert data["input_kind"] == kind
    assert data["cohort"] == 0
    assert data["blocks"] == [["serial", "fused"], ["fused", "serial"]]
    assert data["oracles"] == 9
    assert data["measured_graphs"] == 36
    assert data["calls_per_graph"] == 10
    assert data["graph_warmups"] == 4
    assert data["prefetch"] == {"tokens": 1024, "windows_requested": 4, "rank_start": 1024}
    if kind == "synthetic":
        assert data["synthetic_fixture_config"] == {
            "history": 131072, "resident": 16384, "miss_percent": 10, "index_heads": 32, "seed": 43,
        }


def test_synthetic_and_trace_are_exclusive():
    script = Path(__file__).resolve().parent / "support/fixed_q4_a5_12.py"
    result = subprocess.run(
        [sys.executable, "-B", script, "--synthetic", "--trace", "/unused", "--dry-run"],
        capture_output=True, text=True,
    )
    assert result.returncode == 2 and "not allowed with argument" in result.stderr


def test_synthetic_prepares_once_and_uses_fresh_oracles(tmp_path, monkeypatch):
    kvaf = tmp_path / "kvaf"
    cpu = kvaf / "build-cpu/runtime/kvaf_cpu_service"
    cpu.parent.mkdir(parents=True)
    cpu.touch()
    cpu.chmod(0o700)
    out = tmp_path / "out"
    args = SimpleNamespace(kvaf_root=kvaf, cpu_numa=None, cpu_ub_device=None, device=2, physical_npu=2,
                           synthetic=True, trace=None, output=out)

    def git_output(command, **_kwargs):
        if command[3] == "status":
            return ""
        return runner.KVAF_REVISION if command[2] == kvaf else runner.NATIVE

    monkeypatch.setattr(runner.subprocess, "check_output", git_output)
    monkeypatch.setattr(runner, "same_production", lambda *_args: None)
    monkeypatch.setattr(runner.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(runner, "cpu_affinity", lambda _node: ("0", "0"))
    prepared, calls, cells, summaries = [], [], [], []
    fixture_info = {"actual_miss_percent_per_query": [9.96, 10.01, 9.99, 10.04]}

    def prepare(fixture, **config):
        prepared.append((fixture, config))
        return fixture_info

    def arm(_args, directory, batch, route, source_options, group, node, cpus, *, oracle=None, adapter=None):
        assert source_options == ["--synthetic-fixture", str(out / "fixture")]
        assert group == 0 and (node, cpus) == ("0", "0")
        assert directory.is_relative_to(out / f"synthetic/bs{batch}/cohort0")
        assert adapter == ("adapter" if calls else None)
        calls.append((batch, route, directory, oracle))
        return "adapter"

    def cell(_out, directory, batch, group):
        assert group == 0 and directory == out / f"synthetic/bs{batch}/cohort0"
        result = {"batch": batch}
        cells.append(result)
        return result

    monkeypatch.setattr(runner, "prepare_compact_fixture", prepare)
    monkeypatch.setattr(runner, "run_arm", arm)
    monkeypatch.setattr(runner, "summarize_cell", cell)
    monkeypatch.setattr(runner, "report_cell", lambda _row: None)
    monkeypatch.setattr(runner, "write_summary", lambda _out, rows, metadata: summaries.append((len(rows), metadata)))
    runner.run(args)
    assert prepared == [(out / "fixture", runner.SYNTHETIC_FIXTURE)]
    assert calls[:9] == [
        (batch, "serial", out / f"synthetic/bs{batch}/cohort0/oracle/serial", None)
        for batch in runner.FIXED_BATCHES
    ]
    expected = []
    for batch in runner.FIXED_BATCHES:
        case = out / f"synthetic/bs{batch}/cohort0"
        for block, order in enumerate(runner.BLOCKS, 1):
            expected.extend((batch, route, case / f"block{block}" / route, case / "oracle/serial") for route in order)
    assert calls[9:] == expected
    assert len(cells) == 9
    count, metadata = summaries[-1]
    assert count == 9 and metadata["input_kind"] == "synthetic" and metadata["group"] == 0
    assert metadata["synthetic_fixture"] == fixture_info and "trace" not in metadata


def test_one_cohort_summary_reuses_pair_evidence(tmp_path):
    for key, value in [("native", "test-sha"), ("host", "a5-12"), ("physical_npu", "2")]:
        (tmp_path / f"{key}.txt").write_text(value)
    cell = tmp_path / "g2/bs8/cohort0"

    def write_arm(block, arm, graph):
        out = cell / f"block{block}" / arm
        out.mkdir(parents=True)
        (out / "placement_verified").touch()
        (out / "thp.json").write_text(json.dumps(dict(
            status="measured", page_mode="thp", coverage_min_pct=100, ub_device="adapter",
            numa_pages_overlapping_vmas={"0": 100},
        )))
        result = dict(
            identity=dict(group=2, batch=8, host="a5-12", native="test-sha", physical_npu=2,
                          donor_rows=list(range(0, 16, 2)), sequence_sha256="same"),
            arm=arm, calls=10, checked_queries=320, checked_rows=80, correctness="PASS",
            get_workers=2 if arm == "serial" else 40, graph_event_us=graph * 10,
            graph_mean_us=graph, mean_us=dict(LI=0.6 * graph, tail=0.3 * graph),
        )
        if block == 2 and arm == "fused":
            result["identity"]["sequence_sha256"] = "different"
        (out / "packed_result.json").write_text(json.dumps(result))
        return out / "packed_result.json"

    first_serial = write_arm(1, "serial", 100)
    write_arm(1, "fused", 80)
    write_arm(2, "serial", 110)
    last_fused = write_arm(2, "fused", 90)
    with pytest.raises(ValueError, match="inputs, runtime or memory placement"):
        summarize_cell(tmp_path, cell, 8)
    data = json.loads(last_fused.read_text())
    data["identity"] = json.loads(first_serial.read_text())["identity"]
    last_fused.write_text(json.dumps(data))

    row = summarize_cell(tmp_path, cell, 8)
    assert row["input_kind"] == "real"
    assert row["cohorts"] == 1 and row["cohort"] == 0
    assert row["mean_us"]["serial"]["graph"] == 105
    assert row["mean_us"]["fused"]["graph"] == 85
    assert row["graph_gain_pct"] == pytest.approx(100 * 20 / 105)
    write_summary(tmp_path, [row], plan())
    report = json.loads((tmp_path / "packed_summary.json").read_text())
    assert report["cells"][0]["observations"][1]["order"] == ["fused", "serial"]
    with (tmp_path / "summary.csv").open(newline="") as handle:
        data = list(csv.DictReader(handle))[0]
    assert data["fused_graph_us"] == "85" and data["input_kind"] == "real"
