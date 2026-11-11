# SPDX-License-Identifier: Apache-2.0
"""CLI-only native output capture and saved-result summaries; never timed."""

from collections import deque
import ctypes
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import traceback

from kvip.diagnostics.prefetch_admission import mean_admission

_ACTIVE = None


def emit(message, *, file=None, end="\n", flush=True):
    """Keep explicit progress visible while native fd 1/2 chatter is spooled."""
    target = file or sys.stdout
    if _ACTIVE is not None:
        target = _ACTIVE.stderr if target is sys.stderr else _ACTIVE.stdout
    print(message, file=target, end=end, flush=flush)


def flush_output():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            pass
    # Native printf may be buffered independently of Python's text wrappers.
    try:
        ctypes.CDLL(None).fflush(None)
    except (OSError, AttributeError):
        pass


def failure_tail(path, *, message, lines=80):
    """Read bounded memory, retain the complete spool, and expose its location."""
    try:
        with Path(path).open(errors="replace") as source:
            tail = "".join(deque(source, maxlen=lines))
    except OSError as error:
        tail = f"Could not read retained log: {error}\n"
    emit(f"FAILED: {message}\nFailure log: {Path(path).resolve()}\n--- last {lines} log lines ---", file=sys.stderr)
    if tail:
        emit(tail, file=sys.stderr, end="" if tail.endswith("\n") else "\n")


class NativeLog:
    """Restore fds before an exception reaches the CLI's retained-owner _exit."""

    def __init__(self, enabled):
        self.enabled = enabled
        self.failure_path = None
        self.path = None

    def __enter__(self):
        global _ACTIVE
        if not self.enabled:
            return self
        if _ACTIVE is not None:
            raise RuntimeError("Native CLI logging cannot be nested")
        flush_output()
        fd, path = tempfile.mkstemp(prefix="kvip-replay-", suffix=".log")
        self.path = Path(path)
        self.stdout = self.stderr = None
        try:
            self.stdout = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
            self.stderr = os.fdopen(os.dup(2), "w", encoding="utf-8", buffering=1)
            os.dup2(fd, 1)
            os.dup2(fd, 2)
        except BaseException:
            for target, original in ((1, self.stdout), (2, self.stderr)):
                if original is not None:
                    os.dup2(original.fileno(), target)
                    original.close()
            self.path.unlink(missing_ok=True)
            raise
        finally:
            os.close(fd)
        _ACTIVE = self
        emit(f"RUN_LOG {self.path.resolve()} (temporary; removed on success)")
        return self

    def __exit__(self, kind, error, tb):
        global _ACTIVE
        if not self.enabled:
            return False
        try:
            if error is not None:
                traceback.print_exception(kind, error, tb)
        finally:
            flush_output()
            os.dup2(self.stdout.fileno(), 1)
            os.dup2(self.stderr.fileno(), 2)
            _ACTIVE = None
        try:
            if error is None:
                self.path.unlink(missing_ok=True)
            else:
                self.failure_path = self.path
                failure_tail(self.path, message=f"{kind.__name__}: {error}")
        finally:
            self.stdout.close()
            self.stderr.close()
        return False


def _read_means(case):
    samples = case.get("reads_samples", [])
    if not samples:
        return {}
    return {
        name: sum(row[name] for row in samples) / len(samples)
        for name in samples[0]
        if all(isinstance(row.get(name), (int, float)) for row in samples)
    }


def _payload_bytes(reads, token_bytes=None):
    result = {
        name: reads[key]
        for name, key in (("actual_transfers", "bytes"), ("useful_transfers", "useful_transfer_bytes"))
        if key in reads
    }
    if token_bytes is not None and "cold_demand" in reads:
        result["cold_demand"] = reads["cold_demand"] * token_bytes
    return result


def _payload_bandwidth(label, duration_us, payload, rates=None):
    if not payload:
        return
    if not math.isfinite(duration_us) or duration_us <= 0:
        raise ValueError("Invalid saved payload timing")
    values = []
    for key, name in (("actual_transfers", "actual"), ("useful_transfers", "useful"), ("cold_demand", "cold")):
        if key not in payload:
            continue
        size = payload[key]
        rate = (rates or {}).get(key, size / (duration_us * 1000))
        if not math.isfinite(size) or size < 0 or not math.isfinite(rate) or rate < 0:
            raise ValueError("Invalid saved payload bytes or rate")
        values.append(f"{name}={size / (1 << 20):.3f} MiB @ {rate:.3f} GB/s")
    emit(f"  {label} (decimal GB/s, not link peak): " + "; ".join(values))


def print_admission(admission):
    if admission is None:
        return

    def vector(key):
        return "[" + ",".join(f"{value:g}" for value in admission[key]) + "]"

    emit(
        "  admission/round mean: "
        f"packed/active_rows={admission['packed_rows']}/{admission['active_rows']}, "
        f"window_limited={admission['window_limited_epochs']:g}, "
        f"other_or_tied={admission['other_or_tied_limited_epochs']:g}, "
        f"budget_exhausted={admission['budget_exhausted_epochs']:g}, "
        f"active_unused_budget={admission['remaining_admissible_budget']:g}, "
        f"suppressed_with_budget_left_rows={admission['rows_suppressed_with_budget_left']:g}; "
        f"sent={vector('posted_by_epoch')} suppressed={vector('suppressed_by_epoch')} "
        "(epochs include padding; suppressed candidates can repeat, not final misses)"
    )


def _case(
    name, timing, stages, reads, tuning=None, *, token_bytes=None, payload=None, payload_rates=None, admission=None
):
    if not all(math.isfinite(timing[key]) and timing[key] > 0 for key in ("mean_us", "p50_us")):
        raise ValueError(f"Invalid saved timing: {name}")
    if tuning is not None:
        emit(f"CONFIG {name}: {json.dumps(tuning, sort_keys=True)}")
    emit(f"{name}: device graph mean={timing['mean_us']:.3f} us p50={timing['p50_us']:.3f} us")
    keys = ("prefetch", "final", "unused_prefetch", "bytes", "cold_demand", "candidate_hot_refetch")
    values = [
        f"{key}=" + (f"{reads[key]:,.3f}".rstrip("0").rstrip(".") if key == "bytes" else f"{reads[key]:g}")
        for key in keys
        if key in reads
    ]
    if values:
        emit("  reads/round mean: " + ", ".join(values))
    _payload_bandwidth(
        "payload / graph mean",
        timing["mean_us"],
        {**_payload_bytes(reads, token_bytes), **(payload or {})},
        payload_rates,
    )
    categories = ("cold", "candidate_hot", "other_hot", "abc")
    if "prefetch_cold" in reads:
        emit(
            "  prefetch/unused mean: "
            + ", ".join(
                f"{name}={reads.get('prefetch_' + name, 0):g}/{reads.get('unused_prefetch_' + name, 0):g}"
                for name in categories
            )
            + f"; unclassified={reads.get('prefetch_unclassified', 0):g}"
        )
    print_admission(admission)
    stage_details = stages or {}
    stage_times = stage_details.get("stages", {})
    if stage_times:
        emit(
            "  stage p50 us (instrumented): "
            + ", ".join(f"{name}={value['p50_us']:.3f}" for name, value in stage_times.items())
        )
    get_mean = stage_times.get("batch_get", {}).get("mean_us")
    stage_reads = _read_means(stage_details)
    if get_mean is not None and "bytes" in stage_reads:
        _payload_bandwidth(
            "payload / Get stage mean (instrumented)", get_mean, {"actual_transfers": stage_reads["bytes"]}
        )


def _candidate_call(case):
    call = case.get("candidate_call")
    if call is not None:
        emit(
            f"  separate candidate-to-output wall: mean={call['wall']['mean_us']:.3f} us "
            f"p50={call['wall']['p50_us']:.3f} us; CPU prepare p50={call['cpu_prepare']['p50_us']:.3f} us"
        )
        if call.get("h2d_diagnostic"):
            emit(f"  H2D diagnostic p50={call['h2d_diagnostic']['p50_us']:.3f} us")
        emit("  Starts with this group's captured candidates ready; excludes full-step/shared preparation.")


def compact_report(report, *, path=None):
    """Render existing JSON evidence; no runtime imports, inference or writes."""
    if not isinstance(report, dict) or "status" not in report:
        raise ValueError("Not a saved diagnostic report")
    if report.get("content") == "real_li_synthetic_kv":
        emit("CONTENT real_li_synthetic_kv: real Q/Index/scores; generated KV bytes; single-device operator timing")
    elif report.get("content") == "synthetic_token_replay":
        emit("CONTENT synthetic_token_replay: generated scores/KV bytes; not real-model timing")
    if report.get("score_comparison"):
        emit("LI_SCORE_CHECK " + json.dumps(report["score_comparison"], sort_keys=True))
    status = report["status"]
    if status == "failed":
        emit(f"FAILED: {report.get('error', 'saved run failed')}", file=sys.stderr)
        if report.get("failure_log"):
            emit(f"Failure log: {report['failure_log']}", file=sys.stderr)
    elif status == "verified_snapshot":
        emit("SNAPSHOT_VERIFIED " + json.dumps(report.get("geometry", {}), sort_keys=True))
    elif status != "pass":
        raise ValueError(f"Unsupported diagnostic status: {status}")
    else:
        emit("CORRECTNESS_PASS " + json.dumps(report.get("geometry", {}), sort_keys=True))
        if "cases" in report:
            if not report["cases"]:
                raise ValueError("Saved report has no measured cases")
            for case in report["cases"]:
                _case(
                    case["name"],
                    case["timing"],
                    case.get("stage_diagnostics"),
                    _read_means(case),
                    case.get("tuning"),
                    token_bytes=case.get("geometry", report.get("geometry", {})).get("token_bytes"),
                    admission=mean_admission(case.get("reads_samples", [])),
                )
                _candidate_call(case)
            baseline = report["cases"][0]
            for case in report["cases"][1:]:
                if any(case.get(key) != baseline.get(key) for key in ("geometry", "replay_selection", "scope")):
                    continue
                old, new = baseline["timing"]["mean_us"], case["timing"]["mean_us"]
                emit(f"{case['name']} vs {baseline['name']}: {old / new:.3f}x, reduction={(1 - new / old) * 100:.2f}%")
        elif "table" in report:
            for name, row in report["table"].items():
                if row.get("serial_get_config"):
                    emit(
                        f"CONFIG {name}: workers={row.get('runtime_workers')}; "
                        + json.dumps(row["serial_get_config"], sort_keys=True)
                    )
                _case(
                    name,
                    row,
                    report.get("stage_diagnostics", {}).get(name),
                    dict(
                        row.get("reads_per_round", {}),
                        **({"bytes": row["mean_read_bytes"]} if "mean_read_bytes" in row else {}),
                    ),
                    row.get("tuning"),
                    token_bytes=report.get("geometry", {}).get("token_bytes"),
                    payload={
                        **(
                            {"useful_transfers": row["mean_useful_transfer_bytes"]}
                            if "mean_useful_transfer_bytes" in row
                            else {}
                        ),
                        **(
                            {"cold_demand": row["demand_per_round"]["cold_payload_bytes"]}
                            if "cold_payload_bytes" in row.get("demand_per_round", {})
                            else {}
                        ),
                    },
                    payload_rates=row.get("graph_normalized_payload_GBps"),
                    admission=row.get("prefetch_admission"),
                )
            for name, value in report.get("fusion_vs_serial", {}).items():
                median = value.get("p50_latency_reduction_percent")
                median_text = "" if median is None else f", p50 reduction={median:.2f}%"
                emit(
                    f"{name}: {value['serial_over_fused']:.3f}x, reduction={value['latency_reduction_percent']:.2f}%{median_text}, "
                    f"target_met={value['target_met']} (device graph mean)"
                )
        elif "timing" in report:
            _case(
                f"{report.get('transport', 'unknown')}-{report.get('mode', 'unknown')}",
                report["timing"],
                report.get("stage_diagnostics"),
                _read_means(report),
                report.get("tuning"),
                token_bytes=report.get("geometry", {}).get("token_bytes"),
                admission=mean_admission(report.get("reads_samples", [])),
            )
            _candidate_call(report)
        else:
            raise ValueError("Saved report has no measured cases")
        emit(
            "Stage p50 values are separate diagnostics, not additive full-graph components; correctness is not a performance target."
        )
    if path is not None:
        emit(f"Results: {Path(path).resolve()}")


def report_only(path):
    try:
        report = json.loads(Path(path).read_text())
        compact_report(report, path=path)
    except (OSError, ValueError, KeyError, TypeError) as error:
        emit(f"FAILED reading report {path}: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    if report["status"] == "failed":
        raise SystemExit(1)
