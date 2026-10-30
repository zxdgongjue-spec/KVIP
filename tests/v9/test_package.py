"""Installed package tests: no AST extraction, framework, CANN or KVAF."""

import json
import subprocess
import sys

import pytest
from benchmark_contract import parse_args, run_cpu
from contract_fixture import CASES, make_fixture
from kvop.v9 import ABI_VERSION, native_source, validate_request


def test_import_is_independent():
    code = """
import sys
from kvop.v9 import ABI_VERSION, native_source, ContractLaunch
assert ABI_VERSION == 13
assert (native_source() / 'CMakeLists.txt').is_file()
assert not any(n == 'torch' or n.startswith(('vllm', 'torch_npu', 'kvaf')) for n in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_sources_complete():
    source = native_source()
    manifest = json.loads((source / "SOURCE.json").read_text())
    assert ABI_VERSION == 13
    assert all((source / name).is_file() for name in manifest["sha256"])


@pytest.mark.parametrize("query", [1, 2, 3, 5])
def test_future_query_modes_rejected(query):
    with pytest.raises(ValueError, match="MTP3/Q4"):
        validate_request(queries_per_request=query, candidate_lengths=[16384], resident_budgets=[12288])


def test_future_lengths_rejected():
    with pytest.raises(ValueError, match="equal"):
        validate_request(queries_per_request=4, candidate_lengths=[16384, 16385], resident_budgets=[12288, 12288])
    validate_request(queries_per_request=4, candidate_lengths=[16384, 16384], resident_budgets=[12288, 12288])


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
@pytest.mark.parametrize("windows", [0, 4])
def test_host_contract(case, windows, tmp_path):
    args = parse_args(["--backend", "cpu", "--layers", "1", "--steps", "1", "--intermediate-fetches", str(windows)])
    geometry, arrays, scores = make_fixture(case=case)
    result = run_cpu(args, geometry, arrays, scores, case, tmp_path)
    assert result["status"] == "pass"
    assert result["performance_status"] == "not_measured"


def test_lifecycle_reorder_and_rewrite(tmp_path):
    args = parse_args(
        [
            "--backend",
            "cpu",
            "--batches",
            "2",
            "--layers",
            "2",
            "--state-mode",
            "stateful",
            "--same-length-rewrite",
            "--steps",
            "6",
        ]
    )
    for layer in range(2):
        geometry, arrays, scores = make_fixture(batch=2, layers=2, layer=layer, candidate=args.history + args.steps * 4)
        arrays["candidate_lens"].fill(args.history)
        arrays["logical_lens"].fill(args.history + args.tail)
        result = run_cpu(args, geometry, arrays, scores, CASES[0], tmp_path)
        assert result["status"] == "pass"


def test_real_public_buffer_validation():
    torch = pytest.importorskip("torch")
    from contract_fixture import to_device
    from kvop.v9.contract import allocate_workspace, validate_buffers

    g, arrays, _ = make_fixture()
    tensors = to_device(torch, arrays, g, torch.device("cpu"))
    workspace = allocate_workspace(torch, g, torch.device("cpu"))
    validate_buffers(g, tensors, workspace, device_type="cpu")
    tensors["resident_table"] = tensors["resident_table"][:, :-1].contiguous()
    with pytest.raises(ValueError, match="resident_table"):
        validate_buffers(g, tensors, workspace, device_type="cpu")
