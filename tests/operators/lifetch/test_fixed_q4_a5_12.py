# SPDX-License-Identifier: Apache-2.0
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
from fixed_q4_a5_12 import plan, summarize_cell, write_summary


def test_dry_run_needs_no_a5_dependencies():
    script = Path(__file__).resolve().parent / "support/fixed_q4_a5_12.py"
    result = subprocess.check_output([sys.executable, "-B", script, "--dry-run"], text=True)
    data = json.loads(result)
    assert data["batch_sizes"] == [1, 2, 4, 8, 16, 20, 24, 28, 32]
    assert data["group"] == 2
    assert data["cohort"] == 0
    assert data["blocks"] == [["serial", "fused"], ["fused", "serial"]]
    assert data["oracles"] == 9
    assert data["measured_graphs"] == 36
    assert data["calls_per_graph"] == 10
    assert data["graph_warmups"] == 4
    assert data["prefetch"] == {"tokens": 1024, "windows_requested": 4, "rank_start": 1024}


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
    assert row["cohorts"] == 1 and row["cohort"] == 0
    assert row["mean_us"]["serial"]["graph"] == 105
    assert row["mean_us"]["fused"]["graph"] == 85
    assert row["graph_gain_pct"] == pytest.approx(100 * 20 / 105)
    write_summary(tmp_path, [row], plan())
    report = json.loads((tmp_path / "packed_summary.json").read_text())
    assert report["cells"][0]["observations"][1]["order"] == ["fused", "serial"]
    with (tmp_path / "summary.csv").open(newline="") as handle:
        assert list(csv.DictReader(handle))[0]["fused_graph_us"] == "85"
