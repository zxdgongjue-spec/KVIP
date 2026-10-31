"""Gate real NPU/KVAF TPOT and selection latency, including p95/p99 tails.

Requires at least 100 chronological samples per metric, matching warmup/run
configuration, and hashed raw logs plus passing functional reports. File hashes
bind submitted evidence; this tool cannot prove log authenticity, representative
sampling, independence, or statistical significance and never measures hardware.
"""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

MIN_SAMPLES = 100
METRICS = ("tpot_ms", "selection_latency_us")
FUNCTIONAL_CHECKS = frozenset(
    {"accuracy", "selection", "feedback", "lifecycle", "eager", "graph"}
)


def evidence_artifacts(record, directory):
    references = record.get("evidence_source", {})
    if not isinstance(references, dict) or references.get("kind") != "real-npu-kvaf":
        raise ValueError("evidence_source must identify real-npu-kvaf evidence")
    artifacts = {}
    for name in ("raw_log", "functional_report"):
        reference = references.get(name, {})
        if (
            not isinstance(reference, dict)
            or not isinstance(reference.get("path"), str)
            or not reference["path"]
        ):
            raise ValueError(f"missing evidence artifact {name}")
        path = Path(reference["path"])
        path = path if path.is_absolute() else Path(directory) / path
        digest = reference.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError(f"invalid evidence SHA256 for {name}")
        try:
            content = path.read_bytes()
        except OSError as error:
            raise ValueError(f"unreadable evidence artifact {name}: {path}") from error
        if not content or hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"evidence artifact hash mismatch: {name}")
        artifacts[name] = content
    try:
        report = json.loads(artifacts["functional_report"])
    except (UnicodeError, ValueError) as error:
        raise ValueError("functional report must be JSON") from error
    if not isinstance(report, dict) or report.get("functional_pass") is not True:
        raise ValueError("functional report did not pass")
    checks = report.get("checks", {})
    if not isinstance(checks, dict) or any(
        checks.get(name) is not True for name in FUNCTIONAL_CHECKS
    ):
        raise ValueError(
            "functional report must pass accuracy, selection, feedback, lifecycle, eager and graph"
        )
    for field in (
        "transport",
        "execution_device",
        "soc",
        "cann",
        "kvaf",
        "model",
        "workload",
        "source_revision",
        "library_sha256",
    ):
        if report.get(field) != record.get(field):
            raise ValueError(f"functional report provenance mismatch: {field}")


def validate_measurements(record):
    config = record.get("measurement_config")
    if not isinstance(config, dict):
        raise ValueError(
            "measurement_config with explicit warmup/run configuration is required"
        )
    for name, minimum in (
        ("warmup_iterations", 1),
        ("measurement_iterations", MIN_SAMPLES),
        ("repetitions", 3),
    ):
        if type(config.get(name)) is not int or config[name] < minimum:
            raise ValueError(f"measurement_config {name} must be >= {minimum}")
    if (
        config.get("sampling_policy") != "all-measured-samples"
        or not isinstance(config.get("timing_method"), str)
        or not config["timing_method"]
    ):
        raise ValueError(
            "explicit timing_method and all-measured-samples policy are required"
        )
    for metric in METRICS:
        samples = record.get(metric)
        if (
            not isinstance(samples, list)
            or len(samples) != config["measurement_iterations"]
            or any(
                type(x) not in (int, float) or not math.isfinite(x) or x <= 0
                for x in samples
            )
        ):
            raise ValueError(
                f"{metric} requires >= {MIN_SAMPLES} positive finite samples matching measurement_iterations"
            )


def percentiles(samples):
    ordered = sorted(samples)
    return {
        "median": statistics.median(ordered),
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "p99": ordered[math.ceil(0.99 * len(ordered)) - 1],
    }


def compare(
    baseline,
    candidate,
    *,
    max_regression_percent=0.0,
    baseline_dir=Path("."),
    candidate_dir=Path("."),
):
    if (
        type(max_regression_percent) not in (int, float)
        or not math.isfinite(max_regression_percent)
        or max_regression_percent < 0
    ):
        raise ValueError(
            "regression tolerance must be explicit, finite and nonnegative"
        )
    for record, directory in ((baseline, baseline_dir), (candidate, candidate_dir)):
        if not isinstance(record, dict):
            raise ValueError("performance evidence must be a JSON object")
        if record.get("transport") != "kvaf" or record.get("execution_device") != "npu":
            raise ValueError(
                "qualification requires real NPU/KVAF evidence, not CPU or memory stubs"
            )
        if record.get("functional_pass") is not True:
            raise ValueError("functional accuracy/lifecycle must pass first")
        for field in (
            "soc",
            "cann",
            "kvaf",
            "model",
            "workload",
            "source_revision",
            "library_sha256",
        ):
            if not record.get(field):
                raise ValueError(f"missing evidence field {field}")
        library_hash = record["library_sha256"]
        if (
            not isinstance(library_hash, str)
            or len(library_hash) != 64
            or any(digit not in "0123456789abcdef" for digit in library_hash)
        ):
            raise ValueError("library_sha256 must identify the measured binary")
        if not isinstance(record.get("command"), str) or not record["command"]:
            raise ValueError("the full collection command is required")
        validate_measurements(record)
        evidence_artifacts(record, directory)
    for field in ("soc", "cann", "kvaf", "model", "workload", "measurement_config"):
        if baseline[field] != candidate[field]:
            raise ValueError(f"incomparable hardware/workload: {field}")
    metrics = {}
    for metric in METRICS:
        previous, current = (
            percentiles(record[metric]) for record in (baseline, candidate)
        )
        deltas = {name: (current[name] / previous[name] - 1) * 100 for name in previous}
        metrics[metric] = {
            "baseline": previous,
            "candidate": current,
            "regression_percent": deltas,
            "passed": all(delta <= max_regression_percent for delta in deltas.values()),
            "sample_count": len(baseline[metric]),
        }
    return {
        "passed": all(metric["passed"] for metric in metrics.values()),
        "metrics": metrics,
        "allowed_regression_percent": max_regression_percent,
        "percentile_method": "nearest-rank; median uses midpoint for even counts",
        "evidence_authenticity_and_statistical_significance": "not_proven",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--max-regression-percent", default=0.0, type=float)
    args = parser.parse_args()
    try:
        result = compare(
            json.loads(args.baseline.read_text()),
            json.loads(args.candidate.read_text()),
            max_regression_percent=args.max_regression_percent,
            baseline_dir=args.baseline.resolve().parent,
            candidate_dir=args.candidate.resolve().parent,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
