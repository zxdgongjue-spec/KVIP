import importlib.util
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


def evidence():
    return dict(
        transport="kvaf",
        execution_device="npu",
        functional_pass=True,
        soc="test-soc",
        cann="test-cann",
        kvaf="test-sdk",
        model="test-model",
        workload={"batch": 1},
        source_revision="test-revision",
        library_sha256="test-only",
        tpot_ms=[1.0] * 5,
    )


def test_gate_rejects_stubs_incomparable_workloads_and_regression():
    compare = gate()
    baseline = evidence()
    with pytest.raises(ValueError, match="real NPU"):
        compare(baseline, baseline | {"transport": "npu-memory"})
    with pytest.raises(ValueError, match="incomparable"):
        compare(baseline, baseline | {"workload": {"batch": 2}})
    with pytest.raises(ValueError, match="functional"):
        compare(baseline, baseline | {"functional_pass": False})
    assert not compare(baseline, baseline | {"tpot_ms": [1.01] * 5})["passed"]
    assert compare(baseline, baseline)["passed"]
