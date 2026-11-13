"""Verify B/Q/T Model captures and count canonical per-worker exported calls.

PipeUtilization exports omit synchronization details; totals are exported
instruction calls, not all hardware micro-operations or native latency.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import struct


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instruction_counts(path):
    pipes, instructions, rows = Counter(), Counter(), 0
    with path.open() as f:
        reader = csv.DictReader(f)
        if not {"instr", "pipe", "call_count"} <= set(reader.fieldnames or []):
            raise ValueError("missing official instruction CSV columns")
        for row in reader:
            if not re.fullmatch(r"\d+", row["call_count"]):
                raise ValueError("instruction call_count must be a nonnegative integer")
            count = int(row["call_count"])
            pipes[row["pipe"]] += count
            instructions[row["instr"]] += count
            rows += 1
    if not rows or sum(pipes.values()) <= 0:
        raise ValueError("empty instruction evidence")
    return dict(calls=sum(pipes.values()), rows=rows, pipes=dict(pipes), instructions=dict(instructions))


def instruction_pc_counts(path):
    result = Counter()
    with path.open() as f:
        for row in csv.DictReader(f):
            result[(row["instr"], row["addr"], row["pipe"])] += int(row["call_count"])
    return result


def fixed_pc_evidence(work, rows):
    def phase_counters(b, t):
        row = next(r for r in rows if r["batch"] == b and r["shards"] == t)
        return [instruction_pc_counts(work / row["case"] / c["csv_path"]) for c in row["phases"][0]["cores"]]

    fits = {}
    for t in (4, 8):
        b1, b8 = phase_counters(1, t), phase_counters(8, t)
        c1, c8 = sum(b1, Counter()), sum(b8, Counter())
        denominator = 8 * len(b1) - len(b8)
        fits[t] = {pc: (8 * c1[pc] - c8[pc]) / denominator for pc in c1.keys() | c8.keys() if 8 * c1[pc] != c8[pc]}
    same = fits[4] == fits[8]
    fixed = fits[4]
    once = same and all(v == 1 for v in fixed.values())
    checks = {}
    for row in rows:
        counters = phase_counters(row["batch"], row["shards"])
        checks[row["case"]] = all(c[pc] == 1 for c in counters for pc in fixed)
    by_pipe = Counter()
    for pc, coefficient in fixed.items():
        by_pipe[pc[2]] += coefficient
    return dict(
        scope="Empirical PC-level local-kernel fixed term; B1/8 calibration, B16 held out",
        same_pc_coefficients_from_t4_t8=same,
        all_nonzero_coefficients_one=once,
        fixed_pc_groups=len(fixed),
        fixed_calls_by_pipe=dict(by_pipe),
        each_fixed_pc_once_per_worker=checks,
        fixed_pcs=[
            dict(instruction=pc[0], address=pc[1], pipe=pc[2], coefficient=v) for pc, v in sorted(fixed.items())
        ],
    )


def footer(path):
    pattern = re.compile(
        r"core: (\d+), subcore: (\d+), start: (\d+), tick: (\d+), "
        r"kernelId: (\d+), blkID: (\d+), blkDim: (\d+),[^\n]*done"
    )
    with path.open() as f:
        matches = [match for line in f if "done" in line for match in pattern.finditer(line)]
    if len(matches) != 1:
        raise ValueError("raw worker must have exactly one complete done footer")
    match = matches[0]
    names = ("core", "subcore", "start", "tick", "kernel_id", "block", "block_dim")
    result = dict(zip(names, map(int, match.groups())))
    if result["tick"] <= result["start"]:
        raise ValueError("worker footer end must exceed start")
    return result


def phase_summary(case, phase, workers):
    """Each physical family is exported separately; never add auto-export copies."""
    result, pipes, seen, launches = [], Counter(), set(), set()
    for path in sorted((case / "exports" / phase).rglob("*_instr_exe*.csv")):
        identity = re.fullmatch(r"core(\d+)\.veccore([01])_instr_exe(?:_\d+)?\.csv", path.name)
        if not identity:
            raise ValueError("unexpected instruction identity")
        core, vector = map(int, identity.groups())
        raw = case / "filtered" / phase / f"core{core}" / f"core{core}.veccore{vector}.instr_log.dump"
        metadata = (raw.parent / "object_dump.txt").read_text().splitlines()
        if len(metadata) != 2 or not metadata[1].startswith("split_topk_" + phase + "_"):
            raise ValueError("phase/raw kernel locator mismatch")
        done = footer(raw)
        if (done["core"], done["subcore"]) != (core, vector + 1):
            raise ValueError("CSV/raw core identity mismatch")
        if done["block_dim"] != workers or done["block"] in seen or done["block"] >= workers:
            raise ValueError("duplicate or incorrect worker coverage")
        seen.add(done["block"])
        launches.add(done["kernel_id"])
        counts = instruction_counts(path)
        pipes.update(counts["pipes"])
        result.append(
            dict(
                **done,
                exported_calls=counts["calls"],
                csv_rows=counts["rows"],
                simt_exported_calls=sum(n for k, n in counts["instructions"].items() if k.startswith("SIMT_")),
                csv_path=str(path.relative_to(case)),
                csv_sha256=sha(path),
                raw_path=str(raw.relative_to(case)),
                raw_sha256=sha(raw),
            )
        )
    if seen != set(range(workers)) or len(launches) != 1:
        raise ValueError(f"incomplete {phase} coverage: {len(seen)}/{workers}")
    return dict(
        phase=phase,
        workers=workers,
        observed_workers=len(result),
        coverage="complete",
        exported_calls=sum(r["exported_calls"] for r in result),
        pipes=dict(pipes),
        cores=result,
    )


def timer_summary(path, local, merge):
    data = path.read_bytes()
    if len(data) != 4096:
        raise ValueError("expected complete two-phase 64x4 uint64 timers")
    values = struct.unpack("<512Q", data)
    result = {}
    for phase, count, offset in (("local", local, 0), ("merge", merge, 256)):
        durations = []
        for i in range(64):
            start, end, block, unused = values[offset + 4 * i : offset + 4 * i + 4]
            if i < count:
                if block != i or end <= start or unused != 0:
                    raise ValueError("invalid active worker timer")
                durations.append(end - start)
            elif (start, end, block, unused) != (0, 0, 0, 0):
                raise ValueError("inactive worker timer is not zero")
        result[f"{phase}_worker_deltas"] = durations
        result[f"{phase}_max_worker_delta"] = max(durations, default=0)
        result[f"{phase}_worker_time_balance"] = (
            sum(durations) / (len(durations) * max(durations)) if durations else None
        )
    result["sum_phase_max_worker_delta"] = result["local_max_worker_delta"] + result["merge_max_worker_delta"]
    return result


def case_summary(directory, b, t):
    oracle = json.loads((directory / "oracle.json").read_text())
    log = (directory / "capture.log").read_text(errors="replace")
    ticks = re.findall(r"Total tick:\s*(\d+)", log)
    if len(ticks) != 1 or "Model stopped successfully." not in log:
        raise ValueError("no complete Model run")
    if (
        oracle["status"] != "passed"
        or oracle["batch"] != b
        or oracle["shards"] != t
        or oracle["history"] != 131072
        or oracle["valid"] != 131072
        or oracle["miss_per_query"] != 205
        or oracle["aiv_budget"] != 56
        or oracle["aic_workers"] != 0
        or not all(oracle[k] for k in ("guards", "selected_set_and_scores", "all_local_candidates", "input_unchanged"))
    ):
        raise ValueError("oracle contract mismatch")
    local, merge = min(4 * b * t, 56), min(4 * b, 56) if t > 1 else 0
    if (oracle["local_workers"], oracle["merge_workers"]) != (local, merge):
        raise ValueError("worker geometry mismatch")
    timers = timer_summary(directory / "oracle.json.timers.bin", local, merge)
    if (
        timers["sum_phase_max_worker_delta"] != oracle["sum_phase_max_worker_cycle_delta"]
        or timers["local_max_worker_delta"] != oracle["local_max_worker_cycle_delta"]
        or timers["merge_max_worker_delta"] != oracle["merge_max_worker_cycle_delta"]
    ):
        raise ValueError("oracle/timer disagreement")
    phases = [phase_summary(directory, "local", local)]
    if merge:
        phases.append(phase_summary(directory, "merge", merge))
    first_start = min(c["start"] for c in phases[0]["cores"])
    local_done = max(c["tick"] for c in phases[0]["cores"])
    last_done = max(c["tick"] for p in phases for c in p["cores"])
    gap = min(c["start"] for c in phases[1]["cores"]) - local_done if merge else None
    kernel_sum = sum(max(c["tick"] for c in p["cores"]) - min(c["start"] for c in p["cores"]) for p in phases)
    if last_done > int(ticks[0]) or (gap is not None and gap < 0):
        raise ValueError("raw phase clocks do not establish a sequential Model envelope")
    calls = sum(p["exported_calls"] for p in phases)
    return dict(
        case=directory.name,
        batch=b,
        queries_per_request=4,
        history=131072,
        resident=16384,
        miss_per_query=205,
        nominal_miss_percent=10,
        actual_miss_percent=100 * 205 / 2048,
        fixture_index_heads=32,
        aic_workers=0,
        physical_npu=False,
        measurement_scope="AIV-only Final TopK component; no QK or transport",
        instruction_metric="PipeUtilization exported call_count; sync detail omitted",
        shards=t,
        aiv_budget=56,
        local_tasks=4 * b * t,
        local_workers=local,
        merge_tasks=4 * b if t > 1 else 0,
        merge_workers=merge,
        total_model_ticks=int(ticks[0]),
        exported_instruction_calls=calls,
        exported_calls_per_query=calls / (4 * b),
        local_exported_calls=phases[0]["exported_calls"],
        merge_exported_calls=phases[1]["exported_calls"] if merge else 0,
        coverage="complete active AIV worker coverage",
        raw_model_first_start=first_start,
        raw_model_last_done=last_done,
        raw_phase_gap=gap,
        raw_kernel_envelope_sum_ticks=kernel_sum,
        non_kernel_envelope_model_ticks=int(ticks[0]) - kernel_sum,
        phases=phases,
        **timers,
        oracle_sha256=sha(directory / "oracle.json"),
        log_sha256=sha(directory / "capture.log"),
        timers_sha256=sha(directory / "oracle.json.timers.bin"),
    )


def bind_capture_receipt(work, row, receipt):
    if receipt["status"] != "PASSED_MODEL_AND_OFFLINE_EXPORT":
        raise ValueError("capture/export journal not complete")
    if (receipt["batch"], receipt["shards"], receipt["total_model_ticks"], receipt["capture_log_sha256"]) != (
        row["batch"],
        row["shards"],
        row["total_model_ticks"],
        row["log_sha256"],
    ):
        raise ValueError("capture journal differs from raw result")
    exported = set()
    for entry in receipt["exports"]:
        log = work / row["case"] / "exports" / entry["phase"] / f"core{entry['core']}" / "export.log"
        if (
            entry["exit_code"] != 0
            or sha(log) != entry["log_sha256"]
            or "Profiling data parse failed" in log.read_text()
            or "Profiling data parse finished." not in log.read_text()
        ):
            raise ValueError("offline export receipt/log mismatch")
        exported.update(entry["csv_paths"])
    actual = {str(Path(row["case"]) / core["csv_path"]) for phase in row["phases"] for core in phase["cores"]}
    if exported != actual:
        raise ValueError("offline export receipt coverage mismatch")
    row.update(
        collector_exit=receipt["collector_exit"],
        collector_postprocess_recovery_requested=receipt["collector_postprocess_recovered"],
        instruction_export_origin="separate filtered per-core SDK offline exports",
        offline_exports=len(receipt["exports"]),
    )


def build(work, source, captures):
    rows = [case_summary(work / f"b{b}-t{t}", b, t) for b in (1, 8, 16) for t in (1, 4, 8)]
    for row in rows:
        matching = [c for c in captures if c["case"] == row["case"]]
        if len(matching) != 1:
            raise ValueError("missing/duplicate capture journal row")
        bind_capture_receipt(work, row, matching[0])
        base = next(r for r in rows if r["batch"] == row["batch"] and r["shards"] == 1)
        row["tick_reduction_vs_t1_pct"] = 100 * (1 - row["total_model_ticks"] / base["total_model_ticks"])
        row["model_tick_speedup_vs_t1"] = base["total_model_ticks"] / row["total_model_ticks"]
        row["kernel_envelope_reduction_vs_t1_pct"] = 100 * (
            1 - row["raw_kernel_envelope_sum_ticks"] / base["raw_kernel_envelope_sum_ticks"]
        )
        row["exported_calls_change_vs_t1_pct"] = 100 * (
            row["exported_instruction_calls"] / base["exported_instruction_calls"] - 1
        )
    if sum(sum(p["workers"] for p in r["phases"]) for r in rows) != 548:
        raise ValueError("nine-point coverage total differs from independently expected 548")
    fixed = {}
    for t in (4, 8):
        b1 = next(r for r in rows if r["batch"] == 1 and r["shards"] == t)
        b8 = next(r for r in rows if r["batch"] == 8 and r["shards"] == t)
        fixed[t] = (8 * b1["local_exported_calls"] - b8["local_exported_calls"]) / (
            8 * b1["local_workers"] - b8["local_workers"]
        )
    count_model = dict(
        scope="Local exported calls only; calibration on the repeated Q4 seed at B1/8, not latency",
        fixed_worker_estimates=fixed,
        common_fixed_term_supported=fixed[4] == fixed[8],
    )
    if fixed[4] == fixed[8]:
        intercept = fixed[4]
        units = {
            r["shards"]: r["local_exported_calls"] - intercept * r["local_workers"] for r in rows if r["batch"] == 1
        }
        count_model.update(
            fixed_worker_calls=intercept,
            per_request_work_calls=units,
            formula="C_local(B,T)=B*U[T]+F*min(4*B*T,56)",
            holdout_b16=[
                dict(
                    shards=r["shards"],
                    predicted_local_calls=16 * units[r["shards"]] + intercept * r["local_workers"],
                    measured_local_calls=r["local_exported_calls"],
                    residual=r["local_exported_calls"] - (16 * units[r["shards"]] + intercept * r["local_workers"]),
                )
                for r in rows
                if r["batch"] == 16
            ],
        )
    return dict(
        scope="Model Final TopK, not QK/score production or full streaming; PipeUtilization exported calls omit sync detail",
        physical_npu=False,
        source=source,
        rows=rows,
        expected_runs=9,
        expected_kernel_launches=15,
        complete_worker_csvs=548,
        samples_per_condition=1,
        warmup=0,
        local_instruction_count_model=count_model,
    )


def write(result, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "topk-bqt-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    rows = [
        {k: v for k, v in r.items() if k not in ("phases", "local_worker_deltas", "merge_worker_deltas")}
        for r in result["rows"]
    ]
    with (output / "topk-bqt-results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def historical_li_summary(work, archive_sha256):
    """Original mixed LI archive: do not relabel its total as pure QK/score."""
    cores = []
    expected = {"core0.cubecore0", "core0.veccore0", "core0.veccore1"}
    for path in sorted((work / "full-export").rglob("*_instr_exe*.csv")):
        identity = re.fullmatch(r"(core0\.(?:cubecore0|veccore[01]))_instr_exe(?:_\d+)?\.csv", path.name)
        if not identity or identity[1] not in expected:
            raise ValueError("unexpected/duplicate historical LI core export")
        core = identity[1]
        expected.remove(core)
        raw = work / "full-input" / (core + ".instr_log.dump")
        done = footer(raw)
        if done["block"] != 0 or done["block_dim"] != 1:
            raise ValueError("historical LI launch differs from original single mixed block")
        counts = instruction_counts(path)
        with raw.open() as f:
            raw_pc_records = sum(bool(re.match(r"\[info\] \[\d+\] \(PC: 0x[0-9a-fA-F]+\)", line)) for line in f)
        cores.append(
            dict(
                core=core,
                exported_calls=counts["calls"],
                raw_main_pc_records=raw_pc_records,
                pipes=counts["pipes"],
                footer=done,
                raw_sha256=sha(raw),
                csv_sha256=sha(path),
                csv_path=str(path.relative_to(work)),
            )
        )
    if expected:
        raise ValueError("incomplete historical LI mixed-core coverage")
    log = work / "full-export.log"
    if "Profiling data parse finished." not in log.read_text() or "parse failed" in log.read_text():
        raise ValueError("historical SDK export did not complete")
    return dict(
        scope="Historical mixed LI kernel: QK/score and TopK, not pure score production",
        batch=1,
        queries=4,
        history=8192,
        resident=4096,
        heads=32,
        miss_percent=0,
        application_rerun=False,
        export_origin="SDK offline parse of original device0 raw companions",
        original_archive="mainv4-baseline-zero-b1-h32-r1.tar.gz",
        archive_sha256=archive_sha256,
        binary_sha256=sha(work / "full-input/aicore_binary.o"),
        export_log_sha256=sha(log),
        pure_qk_score_instruction_count=None,
        comparable_to_current_h128k=False,
        exported_instruction_calls=sum(c["exported_calls"] for c in cores),
        raw_main_pc_records=sum(c["raw_main_pc_records"] for c in cores),
        cores=cores,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write(build(args.work, json.loads(args.source.read_text()), json.loads(args.captures.read_text())), args.output)
