"""Offline consistency checks for exported logs/tables; does not replace raw tensor replay."""

import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/split_core"


def main():
    package = json.loads((DOC / "results/model-records.json").read_text())
    records = package["records"]
    assert len(records) == len({r["id"] for r in records}) == 35
    by_id = {r["id"]: r for r in records}
    for record in records:
        assert record["tested_base"] == "81b1472809b7a5f3469e26467feee0ca1c61d71a"
        data = (DOC / "results" / record["exported_log"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == record["exported_log_sha256"]
        ticks = re.findall(r"Total tick:\s*(\d+)", data.decode())
        assert len(ticks) == 1 and int(ticks[0]) == record["total_model_ticks"]
        assert b"Model stopped successfully." in data
    with (DOC / "split_core_v2_history_results.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 11
    for row in rows:
        record = by_id[row["evidence"].split("#", 1)[1]]
        assert int(row["whole_model_ticks"]) == record["total_model_ticks"]
        assert row["per_token_score_parity"] == "True" and record["per_token_parity_with_original"]
    standard = {r["name"]: int(r["whole_model_ticks"]) for r in rows if r["profile"] == "standard"}
    assert standard["original"] == 1340108 and standard["history-p32-t8"] == 693586
    assert standard["history-p28-t8"] == 698318
    with (DOC / "split_core_v2_component_results.csv").open() as f:
        component = list(csv.DictReader(f))
    assert len(component) == 21
    for row in component:
        record = by_id[row["evidence"].split("#", 1)[1]]
        assert int(row["whole_model_ticks"]) == record["total_model_ticks"]
        if row["completed_worker_timing_valid"] == "False":
            assert row["local_worker_ticks"] == row["merge_worker_ticks"] == row["sum_phase_worker_ticks"] == ""
            assert record["completed_worker_timing_valid"] is False
    with (DOC / "split_core_v2_full_results.csv").open() as f:
        pairs = list(csv.DictReader(f))
    assert len(pairs) == 2
    for row in pairs:
        base = by_id[row["baseline_evidence"].split("#", 1)[1]]
        candidate = by_id[row["candidate_evidence"].split("#", 1)[1]]
        assert int(row["baseline_ticks"]) == base["total_model_ticks"]
        assert int(row["candidate_ticks"]) == candidate["total_model_ticks"]
        assert base["driver_sha256"] == candidate["driver_sha256"]
    current = json.loads((DOC / "results/current-core-smoke.json").read_text())
    data = (DOC / "results/current-core-smoke.log").read_bytes()
    assert hashlib.sha256(data).hexdigest() == current["log_sha256"]
    assert re.findall(r"Total tick:\s*(\d+)", data.decode()) == [str(current["ticks"])]
    assert (
        hashlib.sha256((ROOT / "csrc/operators/common/lifetch/split_topk.cpp").read_bytes()).hexdigest()
        == current["source_sha256"]
    )
    matrix = json.loads((DOC / "results/current-model-matrix.json").read_text())
    assert len(matrix) == 60
    completed = [r for r in matrix if r["status"] == "PASSED_MODEL"]
    assert len(completed) == 2 and sum(r["status"] == "NOT_RUN" for r in matrix) == 58
    for row in completed:
        data = (DOC / "results" / (row["case"] + ".log")).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["log_sha256"]
        assert re.findall(r"Total tick:\s*(\d+)", data.decode()) == [str(row["metric"])]
        assert b"Model stopped successfully." in data
    cpu = json.loads((DOC / "results/cpu-matrix.json").read_text())
    assert len(cpu) == 60 and all(r["status"] == "PASSED_CPU" and r["metric"] is None for r in cpu)
    for row in cpu:
        fixture = row["fixture_proof"]["fixture"]
        assert fixture["exact_miss"] and fixture["final_misses_per_query"] == [row["miss_per_query"]] * 4
        assert fixture["unique_misses_per_request"] == 2 * row["miss_per_query"]
    from compare_pipelines import csv_text, validate

    pipeline_rows = validate()
    assert (DOC / "pipeline-comparison.csv").read_text() == csv_text(pipeline_rows)
    print(
        json.dumps(
            dict(
                status="passed",
                historical_Model_logs=35,
                imported_Fusion3_logs=2,
                unified_comparison_measured_rows=9,
                current_CPU_cases=60,
                current_Model_component_logs=3,
                scope="exported-log/table consistency only; raw large tensors are not in this package",
            )
        )
    )


if __name__ == "__main__":
    main()
