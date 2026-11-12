# SPDX-License-Identifier: Apache-2.0
"""Small CPU archive surveys preserve sample, byte and denominator accounting."""

import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from kvip.diagnostics import threshold_survey as survey
from kvip.diagnostics.token_trace import TokenIdTraceRecorder
from kvip.diagnostics.trajectory import inspect_series
from test_score_trace import score_fixture


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def recorded_series(tmp_path_factory):
    root = (tmp_path_factory.mktemp("threshold-series") / "trace").resolve()
    layers = [f"model.layers.{index}" for index in range(2, 6)]
    for rank in range(2):
        rounds = []
        for number in range(1, 4):
            selector, metadata, publish = score_fixture(real=False)
            metadata.update(dp_rank=rank, dp_size=2, group=2, layer_names=layers)
            metadata["context"].update(round=100 + number, global_decode_confirmed=True)
            # Nonstandard names ensure readers resolve the series entries.
            directory = root / f"dp{rank}" / f"sample-{number}" / "selected-group"
            recorder = TokenIdTraceRecorder(torch, directory, metadata)
            recorder.before(selector)
            scores = publish(recorder)
            scores[:, :, 5001] = 0xBF00  # One actually unused prefetch below final +1.
            selector.rotating.arguments[12].copy_(torch.from_numpy(scores.view("uint8").reshape(-1)))
            recorder.after(selector)
            rounds.append(
                dict(
                    round_index=number,
                    scheduler_round=100 + number,
                    groups=[dict(group=2, manifest=str(Path(recorder.result_path).relative_to(root / f"dp{rank}")))],
                )
            )
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
            request_owner_signature=[dict(request_id=f"request{i}", owner=owner) for i, owner in enumerate((3, 1))],
            rounds=rounds,
        )
        (root / f"dp{rank}" / "series.json").write_text(json.dumps(record))
    return root


def _tasks(root):
    reports = inspect_series(root, expected_dp=[0, 1], expected_rounds=3, expected_layers=4, expected_content="scores")
    return survey.survey_tasks(root, reports)


@pytest.mark.parametrize(
    "cold,selected,expected",
    [
        (0, 100, "[0,1)%"),
        (99, 10000, "[0,1)%"),
        (1, 100, "[1,2)%"),
        (2, 100, "[2,5)%"),
        (5, 100, "[5,10)%"),
        (10, 100, "[10,15)%"),
        (15, 100, "[15,20)%"),
        (20, 100, "[20,100]%"),
        (100, 100, "[20,100]%"),
    ],
)
def test_miss_bins_use_left_closed_exact_count_boundaries(cold, selected, expected):
    assert survey.miss_bin(cold, selected) == expected


@pytest.mark.parametrize("cold,selected", [(-1, 100), (101, 100), (0, 0)])
def test_miss_bins_reject_invalid_denominators(cold, selected):
    with pytest.raises(ValueError):
        survey.miss_bin(cold, selected)


def test_each_archive_loads_once_seed_is_excluded_and_arrays_are_released(recorded_series, monkeypatch):
    task = _tasks(recorded_series)[0]
    original = survey.load_token_trace
    paths, archives = [], []

    def tracked_load(path):
        paths.append(path)
        archive = original(path)
        archives.append(archive)
        return archive

    monkeypatch.setattr(survey, "load_token_trace", tracked_load)
    requests, queries = survey._run_task(task)

    assert len(paths) == len(set(paths)) == 3
    assert all(not archive.arrays for archive in archives)
    assert len(requests) == 4 and len(queries) == 16
    assert {row["capture_round"] for row in requests + queries} == {2, 3}
    assert {row["scheduler_round"] for row in requests + queries} == {102, 103}
    assert all(row["hot_extra_candidates"] == 1 and row["seed_hot_extra_candidates"] == 0 for row in requests)


def test_cross_dp_layer_bytes_conserve_group_total_and_precision_has_denominators(recorded_series):
    totals = survey.SurveyTotals()
    for task in _tasks(recorded_series):
        requests, _ = survey._run_task(task)
        for row in requests:
            totals.add(row)

    overall = totals.overall.record()
    layers, rounds = totals.layer_rows(), totals.round_layer_rows()
    assert overall["requests"] == 8 and overall["samples"] == 4
    assert overall["actual_read_bytes"] == 8 * 3 * 2624
    assert overall["hot_extra_bytes"] == 8 * 2624
    assert len(layers) == 4 and len(rounds) == 8
    assert sum(row["actual_read_bytes"] for row in layers) == overall["actual_read_bytes"]
    assert sum(row["hot_extra_bytes"] for row in layers) == overall["hot_extra_bytes"]
    assert all(row["token_bytes"] == "656" for row in layers)
    assert all(row["requests"] == 4 for row in rounds)  # Two requests on each DP.
    assert overall["hot_precision_percent"] == pytest.approx(200 / 3)
    assert overall["actual_prefetch_precision_percent"] == overall["actual_prefetch_waste_percent"] == 50
    assert overall["actual_unused_vs_cold_percent"] == 100
    assert overall["actual_unused_below_final"] == 8 and overall["actual_unused_final_ties"] == 0
    assert overall["seed_hot_extra_reduction_percent"] == 100
    assert overall["gap_mean"] == overall["gap_max"] == 1
    assert overall["gap_equal_percent"] == 0
    assert totals.bin_rows()[-1]["actual_prefetch_precision_percent"] is None

    # A weighted aggregate must not average per-request miss percentages.
    row = copy.deepcopy(requests[0])
    weighted = survey.Totals()
    row.update(selected_unique=100, cold_demand=1)
    weighted.add(row)
    row.update(selected_unique=1000, cold_demand=100)
    weighted.add(row)
    assert weighted.record()["cold_miss_percent"] == pytest.approx(101 / 1100 * 100)


def test_small_multiprocess_cli_writes_complete_outputs_without_overwriting(recorded_series, tmp_path):
    output = tmp_path / "survey"
    command = [
        sys.executable,
        "-m",
        "kvip.diagnostics.threshold_survey",
        "--root",
        str(recorded_series),
        "--result-dir",
        str(output),
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
    env = dict(os.environ, PYTHONPATH=str(ROOT / "python"))
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "complete" and summary["seed_rounds"] == 1
    assert summary["request_rows"] == 8 and summary["query_rows"] == 32
    for name, count in (("requests", 8), ("queries", 32), ("round_layers", 8), ("layers", 4), ("miss_bins", 7)):
        with (output / f"{name}.csv").open() as source:
            assert len(list(csv.DictReader(source))) == count
    assert not (output / "INCOMPLETE").exists()
    assert "OVERALL" in result.stdout and "actual_waste%" in result.stdout
    assert "THRESHOLD_SURVEY_PASS" in result.stdout and "GAP_NOTE" in result.stdout
    assert "uncapped" in summary["scope"]["predicted"].lower()
    repeated = subprocess.run(command, env=env, text=True, capture_output=True, timeout=30)
    assert repeated.returncode == 1 and "THRESHOLD_SURVEY_FAILED" in repeated.stderr
    assert json.loads((output / "summary.json").read_text()) == summary


def test_survey_import_does_not_load_torch_or_npu():
    code = (
        "import sys; import kvip.diagnostics.threshold_survey; "
        "assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ, PYTHONPATH=str(ROOT / "python")),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
