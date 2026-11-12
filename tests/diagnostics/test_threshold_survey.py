# SPDX-License-Identifier: Apache-2.0
"""Run the CPU CLI over real schema-4 files, not a patched reader."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from kvip.diagnostics import threshold_survey as survey
from score_archive_fixture import score_archive, write_archive


@pytest.fixture
def recorded_series(tmp_path):
    root = tmp_path / "trace"
    for rank in range(2):
        rounds = []
        for number in range(1, 4):
            trace = score_archive(rank=rank, scheduler_round=100 + number)
            path = root / f"dp{rank}" / f"sample-{number}"
            write_archive(trace, path)
            rounds.append(
                dict(
                    round_index=number,
                    scheduler_round=100 + number,
                    groups=[dict(group=2, manifest=f"sample-{number}/manifest.json")],
                )
            )
        layers = trace.manifest["metadata"]["layer_names"]
        record = dict(
            schema_version=1,
            kind="kvaf_decode_snapshot_series",
            scope="target",
            content="scores",
            status="complete",
            dp_rank=rank,
            dp_ranks=[0, 1],
            dp_size=2,
            expected_rounds=3,
            completed_rounds=3,
            batch=2,
            queries=4,
            expected_target_layer_count=4,
            target_layer_names=layers,
            group_selection="all",
            groups=[dict(group=2, layer_names=layers)],
            seed_rounds=1,
            consecutive_required=True,
            real_input_rounds=[],
            real_input_groups=[],
            request_owner_signature=[
                dict(request_id=p["request_id"], owner=p["owner"]) for p in trace.manifest["metadata"]["requests"]
            ],
            rounds=rounds,
        )
        (root / f"dp{rank}" / "series.json").write_text(json.dumps(record))
    return root


def test_complete_cli_survey_recent_counts_and_csvs(recorded_series, tmp_path):
    result = tmp_path / "result"
    command = [
        sys.executable,
        "-m",
        "kvip.diagnostics.threshold_survey",
        "--root",
        str(recorded_series),
        "--result-dir",
        str(result),
        "--workers",
        "2",
        "--expect-dp",
        "0",
        "1",
        "--expect-rounds",
        "3",
        "--expect-layers",
        "4",
    ]
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "python"))
    proc = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    report = json.loads((result / "summary.json").read_text())
    assert report["request_rows"] == 8 and report["query_rows"] == 32
    assert report["schema_version"] == 2 and report["status"] == "complete"
    assert report["overall"]["previous_topk_recent_coverage_percent"] < 100
    assert report["overall"]["recent_extra_candidates"] <= report["overall"]["hot_extra_candidates"]
    assert "no window" in report["scope"]["bounded_vs_full"]
    assert not (result / "INCOMPLETE").exists()
    for name, count in [("requests", 8), ("queries", 32), ("layers", 4), ("round_layers", 8), ("miss_bins", 7)]:
        with (result / f"{name}.csv").open() as f:
            assert len(list(csv.DictReader(f))) == count
    assert "THRESHOLD_SURVEY_PASS" in proc.stdout and "recent_reduce%" in proc.stdout


def test_cli_import_never_loads_torch():
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "python"))
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import kvip.diagnostics.threshold_survey; assert 'torch' not in sys.modules",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("cold,selected,expected", [(0, 100, "[0,1)%"), (5, 100, "[5,10)%"), (100, 100, "[20,100]%")])
def test_exact_miss_bin_boundary(cold, selected, expected):
    assert survey.miss_bin(cold, selected) == expected
