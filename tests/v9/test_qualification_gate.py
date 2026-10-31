import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def gate():
    spec = importlib.util.spec_from_file_location(
        "kvop_qualification", ROOT / "tools/qualify_performance.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compare


def evidence(tmp_path, name="baseline", **changes):
    record = dict(
        transport="kvaf",
        execution_device="npu",
        functional_pass=True,
        soc="test-soc",
        cann="test-cann",
        kvaf="test-sdk",
        model="test-model",
        workload={"batch": 1},
        source_revision="test-revision",
        library_sha256="a" * 64,
        command="test-only collection command; no hardware execution claimed",
        measurement_config=dict(
            warmup_iterations=20,
            measurement_iterations=100,
            repetitions=3,
            sampling_policy="all-measured-samples",
            timing_method="test-only",
        ),
        tpot_ms=[1.0] * 100,
        selection_latency_us=[10.0] * 100,
    )
    record.update(changes)
    raw = tmp_path / f"{name}.log"
    raw.write_text("synthetic test fixture; never hardware evidence\n")
    report = tmp_path / f"{name}-functional.json"
    fields = (
        "transport",
        "execution_device",
        "soc",
        "cann",
        "kvaf",
        "model",
        "workload",
        "source_revision",
        "library_sha256",
    )
    report.write_text(
        json.dumps(
            {
                **{field: record[field] for field in fields},
                "functional_pass": True,
                "checks": dict.fromkeys(
                    (
                        "accuracy",
                        "selection",
                        "feedback",
                        "lifecycle",
                        "eager",
                        "graph",
                    ),
                    True,
                ),
            }
        )
    )
    record["evidence_source"] = {
        "kind": "real-npu-kvaf",
        **{
            key: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for key, path in (("raw_log", raw), ("functional_report", report))
        },
    }
    return record


def test_gate_rejects_stubs_incomparable_workloads_and_regression(tmp_path):
    compare = gate()
    baseline = evidence(tmp_path)
    with pytest.raises(ValueError, match="real NPU"):
        compare(baseline, baseline | {"transport": "npu-memory"})
    with pytest.raises(ValueError, match="incomparable"):
        compare(baseline, evidence(tmp_path, "other", workload={"batch": 2}))
    with pytest.raises(ValueError, match="functional"):
        compare(baseline, baseline | {"functional_pass": False})
    assert not compare(baseline, baseline | {"tpot_ms": [1.01] * 100})["passed"]
    assert compare(baseline, baseline)["passed"]


@pytest.mark.parametrize("metric", ["tpot_ms", "selection_latency_us"])
@pytest.mark.parametrize("tail", [2, 6])
def test_median_improvement_cannot_hide_p95_or_p99_regression(tmp_path, metric, tail):
    baseline = evidence(tmp_path)
    value = baseline[metric][0]
    candidate = baseline | {metric: [value * 0.9] * (100 - tail) + [value * 2] * tail}
    result = gate()(baseline, candidate)
    assert not result["passed"]
    assert result["metrics"][metric]["regression_percent"]["median"] < 0
    assert result["metrics"][metric]["regression_percent"]["p99"] > 0


@pytest.mark.parametrize(
    "change",
    [
        {"selection_latency_us": [1.0] * 5},
        {"tpot_ms": [float("nan")] * 100},
        {"tpot_ms": [True] * 100},
        {"measurement_config": {"warmup_iterations": 0}},
        {
            "measurement_config": {
                "warmup_iterations": 1,
                "measurement_iterations": 100,
                "repetitions": 1,
            }
        },
    ],
)
def test_insufficient_or_invalid_measurements_are_rejected(tmp_path, change):
    baseline = evidence(tmp_path)
    with pytest.raises(ValueError):
        gate()(baseline, baseline | change)


def test_same_samples_with_different_warmup_are_incomparable(tmp_path):
    baseline = evidence(tmp_path)
    candidate = baseline | {
        "measurement_config": baseline["measurement_config"] | {"warmup_iterations": 30}
    }
    with pytest.raises(ValueError, match="measurement_config"):
        gate()(baseline, candidate)


@pytest.mark.parametrize("artifact", ["raw_log", "functional_report"])
def test_missing_or_modified_evidence_artifact_is_rejected(tmp_path, artifact):
    baseline = evidence(tmp_path)
    path = Path(baseline["evidence_source"][artifact]["path"])
    path.write_text("modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        gate()(baseline, baseline)
    path.unlink()
    with pytest.raises(ValueError, match="unreadable"):
        gate()(baseline, baseline)


@pytest.mark.parametrize(
    "change",
    [
        {"functional_pass": False},
        {"checks": {"accuracy": True}},
        {"source_revision": "wrong"},
    ],
)
def test_hashed_report_must_pass_full_functional_checks_and_match_revision(
    tmp_path, change
):
    baseline = evidence(tmp_path)
    reference = baseline["evidence_source"]["functional_report"]
    path = Path(reference["path"])
    path.write_text(json.dumps(json.loads(path.read_text()) | change))
    reference["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="functional report"):
        gate()(baseline, baseline)


def test_result_states_limits_of_the_evidence_gate(tmp_path):
    baseline = evidence(tmp_path)
    result = gate()(baseline, baseline)
    assert result["evidence_authenticity_and_statistical_significance"] == "not_proven"
    assert set(result["metrics"]) == {"tpot_ms", "selection_latency_us"}


def test_artifact_paths_are_resolved_relative_to_evidence_json(tmp_path):
    baseline = evidence(tmp_path)
    for reference in baseline["evidence_source"].values():
        if isinstance(reference, dict):
            reference["path"] = Path(reference["path"]).name
    assert gate()(baseline, baseline, baseline_dir=tmp_path, candidate_dir=tmp_path)[
        "passed"
    ]
