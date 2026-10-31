"""Fail-closed comparison of externally collected real-NPU performance evidence."""

import argparse
import json
import math
import statistics
from pathlib import Path


def compare(baseline, candidate, *, max_regression_percent=0.0):
    for record in (baseline, candidate):
        if record.get("transport") != "kvaf" or record.get("execution_device") != "npu":
            raise ValueError(
                "qualification requires real NPU/KVAF evidence, not CPU or memory stubs"
            )
        if record.get("functional_pass") is not True:
            raise ValueError("functional accuracy/lifecycle must pass first")
        samples = record.get("tpot_ms", [])
        if len(samples) < 5 or any(
            type(x) not in (int, float) or not math.isfinite(x) or x <= 0
            for x in samples
        ):
            raise ValueError("at least five positive finite TPOT samples are required")
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
    for field in ("soc", "cann", "kvaf", "model", "workload"):
        if baseline[field] != candidate[field]:
            raise ValueError(f"incomparable hardware/workload: {field}")
    if not math.isfinite(max_regression_percent) or max_regression_percent < 0:
        raise ValueError(
            "regression tolerance must be explicit, finite and nonnegative"
        )
    previous, current = (statistics.median(r["tpot_ms"]) for r in (baseline, candidate))
    delta = (current / previous - 1) * 100
    return {
        "passed": delta <= max_regression_percent,
        "baseline_median_tpot_ms": previous,
        "candidate_median_tpot_ms": current,
        "regression_percent": delta,
        "allowed_regression_percent": max_regression_percent,
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
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
