"""Verify complete worker coverage and vendor call_count accounting."""

import importlib.util
import copy
import csv
import json
from pathlib import Path
import re
import struct

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("bqt", ROOT / "experiments/split_core/topk_bqt_results.py")
bqt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bqt)


def worker(case, block, workers=2, calls=11, phase="local"):
    core, vector = divmod(block, 2)
    csv = case / "exports" / phase / f"core{core}" / "OPPROF" / "simulator" / f"core{core}.veccore{vector}"
    csv.mkdir(parents=True, exist_ok=True)
    path = csv / f"core{core}.veccore{vector}_instr_exe_20261113.csv"
    path.write_text(
        "instr,addr,pipe,call_count,cycles\nADD,100,SCALAR," + str(calls) + ",30\nSIMT_LDG64,104,SIMT,3,10\n"
    )
    raw = case / "filtered" / phase / f"core{core}" / f"core{core}.veccore{vector}.instr_log.dump"
    raw.parent.mkdir(parents=True, exist_ok=True)
    (raw.parent / "object_dump.txt").write_text(f"/test/raw\nsplit_topk_{phase}_1")
    raw.write_text(
        f"[info] core: {core}, subcore: {vector + 1}, start: 1, tick: 100, kernelId: 0, "
        f"blkID: {block}, blkDim: {workers}, rtsqId: 2, streamId: 0, done\n"
    )
    return path, raw


def test_call_count_not_csv_rows_and_no_hash_dedup(tmp_path):
    first, _ = worker(tmp_path, 0)
    second, _ = worker(tmp_path, 1)
    assert first.read_bytes() == second.read_bytes()
    assert bqt.instruction_counts(first)["calls"] == 14
    result = bqt.phase_summary(tmp_path, "local", 2)
    assert result["exported_calls"] == 28
    assert len(result["cores"]) == 2
    assert result["pipes"] == {"SCALAR": 22, "SIMT": 6}


def test_partial_coverage_is_not_extrapolated(tmp_path):
    worker(tmp_path, 0)
    with pytest.raises(ValueError, match="incomplete"):
        bqt.phase_summary(tmp_path, "local", 2)


def test_duplicate_block_or_csv_is_rejected(tmp_path):
    worker(tmp_path, 0)
    _, raw = worker(tmp_path, 1)
    raw.write_text(raw.read_text().replace("blkID: 1", "blkID: 0"))
    with pytest.raises(ValueError, match="duplicate"):
        bqt.phase_summary(tmp_path, "local", 2)


def test_incomplete_footer_is_rejected(tmp_path):
    _, raw = worker(tmp_path, 0)
    raw.write_text(raw.read_text().replace("done", "interrupted"))
    with pytest.raises(ValueError, match="done footer"):
        bqt.footer(raw)


def test_duplicate_footer_anywhere_or_reverse_clock_is_rejected(tmp_path):
    _, raw = worker(tmp_path, 0)
    done = raw.read_text()
    raw.write_text(done + "unrelated padding\n" * 1000 + done)
    with pytest.raises(ValueError, match="exactly one"):
        bqt.footer(raw)
    raw.write_text(done.replace("start: 1, tick: 100", "start: 100, tick: 1"))
    with pytest.raises(ValueError, match="exceed"):
        bqt.footer(raw)


@pytest.mark.parametrize("value", ["-1", "2.5", "NaN", ""])
def test_malformed_call_counts_are_rejected(tmp_path, value):
    csv, _ = worker(tmp_path, 0, calls=value)
    with pytest.raises(ValueError, match="nonnegative integer"):
        bqt.instruction_counts(csv)


def test_full_timer_geometry_and_inactive_rows(tmp_path):
    path = tmp_path / "timers.bin"
    values = [0] * 512
    values[:8] = [100, 200, 0, 0, 120, 230, 1, 0]
    values[256:260] = [300, 350, 0, 0]
    path.write_bytes(struct.pack("<512Q", *values))
    result = bqt.timer_summary(path, 2, 1)
    assert result["sum_phase_max_worker_delta"] == 160
    values[8] = 1
    path.write_bytes(struct.pack("<512Q", *values))
    with pytest.raises(ValueError, match="inactive"):
        bqt.timer_summary(path, 2, 1)


def test_existing_completed_b1_results_are_full_coverage():
    work = ROOT / ".aiinfra/local/SPLIT-CORE-011"
    if not (work / "b1-t4/exports").exists():
        pytest.skip("raw captures kept local; synthetic conservation checks above remain portable")
    for t, expected in ((1, 4), (4, 20)):
        result = bqt.case_summary(work / f"b1-t{t}", 1, t)
        assert sum(p["workers"] for p in result["phases"]) == expected
        assert result["exported_instruction_calls"] > 0


def test_capture_receipt_preserves_failure_and_rejects_missing_export(tmp_path):
    log = tmp_path / "b1-t1/exports/local/core0/export.log"
    log.parent.mkdir(parents=True)
    log.write_text("Profiling data parse finished.\n")
    row = dict(
        case="b1-t1",
        batch=1,
        shards=1,
        total_model_ticks=100,
        log_sha256="capture",
        phases=[dict(cores=[dict(csv_path="exports/local/core0/core0.csv")])],
    )
    receipt = dict(
        status="PASSED_MODEL_AND_OFFLINE_EXPORT",
        batch=1,
        shards=1,
        total_model_ticks=100,
        capture_log_sha256="capture",
        collector_exit=143,
        collector_postprocess_recovered=True,
        exports=[
            dict(
                phase="local",
                core=0,
                exit_code=0,
                log_sha256=bqt.sha(log),
                csv_paths=["b1-t1/exports/local/core0/core0.csv"],
            )
        ],
    )
    bqt.bind_capture_receipt(tmp_path, row, receipt)
    assert row["collector_exit"] == 143 and row["collector_postprocess_recovery_requested"]
    changed = copy.deepcopy(receipt)
    changed["exports"][0]["csv_paths"] = []
    with pytest.raises(ValueError, match="coverage mismatch"):
        bqt.bind_capture_receipt(tmp_path, row, changed)
    changed = copy.deepcopy(receipt)
    changed["exports"][0]["exit_code"] = 1
    with pytest.raises(ValueError, match="receipt/log mismatch"):
        bqt.bind_capture_receipt(tmp_path, row, changed)
    log.write_text("interrupted before parse completed\n")
    changed = copy.deepcopy(receipt)
    changed["exports"][0]["log_sha256"] = bqt.sha(log)
    with pytest.raises(ValueError, match="receipt/log mismatch"):
        bqt.bind_capture_receipt(tmp_path, row, changed)


def test_published_nine_point_bundle_conserves_work_and_coverage():
    path = ROOT / "docs/research/split_core/results/topk-bqt-results.json"
    if not path.exists():
        pytest.skip("nine-point capture still running")
    result = json.loads(path.read_text())
    with path.with_suffix(".csv").open() as f:
        csv_rows = {r["case"]: r for r in csv.DictReader(f)}
    assert {(r["batch"], r["shards"]) for r in result["rows"]} == {(b, t) for b in (1, 8, 16) for t in (1, 4, 8)}
    assert len(result["rows"]) == 9 and not result["physical_npu"]
    count = 0
    for row in result["rows"]:
        assert row["history"] == 131072 and row["miss_per_query"] == 205
        b, t = row["batch"], row["shards"]
        assert row["local_workers"] == min(4 * b * t, 56)
        assert row["merge_workers"] == (min(4 * b, 56) if t > 1 else 0)
        assert [p["phase"] for p in row["phases"]] == (["local", "merge"] if t > 1 else ["local"])
        assert row["exported_instruction_calls"] == sum(p["exported_calls"] for p in row["phases"])
        assert row["raw_kernel_envelope_sum_ticks"] <= row["total_model_ticks"]
        envelopes = [(min(c["start"] for c in p["cores"]), max(c["tick"] for c in p["cores"])) for p in row["phases"]]
        assert row["raw_kernel_envelope_sum_ticks"] == sum(end - start for start, end in envelopes)
        assert row["raw_phase_gap"] == (envelopes[1][0] - envelopes[0][1] if t > 1 else None)
        assert row["non_kernel_envelope_model_ticks"] == row["total_model_ticks"] - sum(
            end - start for start, end in envelopes
        )
        for key in (
            "total_model_ticks",
            "raw_kernel_envelope_sum_ticks",
            "exported_instruction_calls",
            "kernel_envelope_reduction_vs_t1_pct",
            "exported_calls_change_vs_t1_pct",
        ):
            assert float(csv_rows[row["case"]][key]) == row[key]
        logs = path.parent / "bqt-logs"
        case = row["case"]
        log, oracle, timer = (logs / f"{case}{suffix}" for suffix in (".capture.log", ".oracle.json", ".timers.bin"))
        assert (bqt.sha(log), bqt.sha(oracle), bqt.sha(timer)) == (
            row["log_sha256"],
            row["oracle_sha256"],
            row["timers_sha256"],
        )
        assert "Model stopped successfully." in log.read_text()
        assert re.findall(r"Total tick:\s*(\d+)", log.read_text()) == [str(row["total_model_ticks"])]
        check = bqt.timer_summary(timer, row["local_workers"], row["merge_workers"])
        assert all(row[k] == v for k, v in check.items())
        assert json.loads(oracle.read_text())["status"] == "passed"
        base = next(r for r in result["rows"] if r["batch"] == b and r["shards"] == 1)
        assert row["kernel_envelope_reduction_vs_t1_pct"] == 100 * (
            1 - row["raw_kernel_envelope_sum_ticks"] / base["raw_kernel_envelope_sum_ticks"]
        )
        for phase in row["phases"]:
            assert phase["workers"] == row[f"{phase['phase']}_workers"]
            assert {c["block"] for c in phase["cores"]} == set(range(phase["workers"]))
            assert phase["exported_calls"] == sum(c["exported_calls"] for c in phase["cores"])
            count += len(phase["cores"])
    assert count == 548
    model = result["local_instruction_count_model"]
    assert model["common_fixed_term_supported"] and model["fixed_worker_calls"] == 612
    assert all(h["residual"] == 0 for h in model["holdout_b16"])
