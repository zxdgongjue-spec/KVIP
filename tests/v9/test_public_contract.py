"""Consumer-visible contract and regression tests, including the audited gaps."""

import ctypes
import ast
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from contract_fixture import make_fixture, to_device
from kvop.v9 import (
    ABI_VERSION,
    CONTRACT_VERSION,
    ContractMismatchError,
    ContractRequirements,
    LayoutSpec,
    admit_request,
    negotiate_contract,
    query_capabilities,
    validate_request,
)
from kvop.v9.contract import DeviceContract, allocate_workspace, validate_buffers
from kvop.v9.index_cache import allocate_cache, bind_caches


def test_machine_manifest_and_public_modules_have_no_reverse_framework_dependency():
    import kvop
    from kvop.v9.contract_info import describe_contract

    manifest = json.loads(
        (
            Path(__file__).resolve().parents[2] / "manifests/compatibility.json"
        ).read_text()
    )
    info = describe_contract()
    assert manifest["public_contract_version"] == info["contract_version"]
    assert manifest["package_version"] == info["package_version"]
    assert manifest["device_abi"] == info["device_abi"]
    assert manifest["hardware_certified"] is False
    for path in Path(kvop.__file__).parent.rglob("*.py"):
        if "_native" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            imports = (
                [x.name for x in node.names] if isinstance(node, ast.Import) else []
            )
            if isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
            assert not any(
                name == "vllm" or name.startswith(("vllm.", "vllm_ascend"))
                for name in imports
            )


def test_discovery_is_immutable_independent_and_has_no_implicit_certification():
    info = query_capabilities()
    assert (CONTRACT_VERSION, ABI_VERSION, ctypes.sizeof(DeviceContract)) == (
        "2.0.0",
        13,
        528,
    )
    assert info.layout == LayoutSpec()
    assert not info.hardware_certified
    with pytest.raises(FrozenInstanceError):
        info.device_abi = 14
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from kvop.v9 import query_capabilities; import sys; "
            "query_capabilities(); assert not any(n == 'torch' or n.startswith(('torch_npu','vllm','kvaf')) "
            "for n in sys.modules)",
        ],
        check=True,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("contract_major", 1),
        ("minimum_minor", 1),
        ("device_abi", 12),
        ("profile", "legacy-selection-transfer"),
        ("layout", LayoutSpec(token_bytes=657)),
        ("features", frozenset({"q2"})),
        ("require_hardware_certified", True),
        ("graph_transport_abi", 2),
        ("index_writer_abi", 3),
    ],
)
def test_incompatible_public_contract_fails_closed(field, value):
    with pytest.raises(ContractMismatchError):
        negotiate_contract(replace(ContractRequirements(), **{field: value}))


def test_required_capability_and_minor_compatibility():
    required = ContractRequirements(
        features=frozenset({"fixed-q4-graph", "owned-index"})
    )
    info = query_capabilities()
    assert (
        negotiate_contract(
            required, provider=replace(info, contract_minor=1)
        ).device_abi
        == 13
    )
    with pytest.raises(ContractMismatchError, match="production transport"):
        negotiate_contract(provider=replace(info, transport="host-contract-test"))


@pytest.mark.parametrize(
    "changes",
    [
        {"candidate_lengths": [0]},
        {"candidate_lengths": [262145]},
        {"candidate_lengths": [16384] * 33, "resident_budgets": [12288] * 33},
        {"candidate_lengths": [True]},
        {"candidate_lengths": [1.0]},
        {"queries_per_request": 4.0},
        {"resident_budgets": [True]},
        {"logical_lengths": [1]},
        {"pool_rows": [-1]},
        {"aic_count": 0},
        {
            "candidate_lengths": [16384, 16384],
            "resident_budgets": [12288, 12288],
            "pool_rows": [1, 1],
        },
    ],
)
def test_host_admission_rejects_before_any_ownership_side_effect(changes):
    request = dict(
        queries_per_request=4, candidate_lengths=[16384], resident_budgets=[12288]
    )
    request.update(changes)
    assert not admit_request(**request).supported
    with pytest.raises(ValueError):
        validate_request(**request)


def test_request_geometry_tail_pool_and_resources():
    g, arrays, _ = make_fixture(batch=2)
    request = dict(
        queries_per_request=4,
        candidate_lengths=arrays["candidate_lens"],
        resident_budgets=arrays["budgets"],
        logical_lengths=arrays["logical_lens"],
        pool_rows=arrays["pool_rows"],
        geometry=g,
        aic_count=28,
    )
    assert admit_request(**request).supported
    assert not admit_request(**(request | {"logical_lengths": [999999] * 2})).supported
    assert not admit_request(**(request | {"pool_rows": [g.max_requests, 0]})).supported
    assert not admit_request(**(request | {"geometry": replace(g, batch=1)})).supported


@pytest.mark.parametrize("field", ["extra_prefetch_slots", "extra_prefetch_counts"])
def test_extra_output_cannot_alias_read_only_input(field):
    torch = pytest.importorskip("torch")
    g, arrays, _ = make_fixture()
    t = to_device(torch, arrays, g, torch.device("cpu"))
    w = allocate_workspace(torch, g, "cpu")
    backing = torch.zeros(2048, dtype=torch.int32)
    t["budgets"] = backing[:1]
    t[field] = backing[: t[field].numel()].reshape(t[field].shape)
    with pytest.raises(ValueError, match="overlapping"):
        validate_buffers(g, t, w, device_type="cpu")


def test_index_caches_use_full_ranges_and_reject_workspace_aliases():
    torch = pytest.importorskip("torch")
    g, arrays, _ = make_fixture(batch=2)
    t = to_device(torch, arrays, g, torch.device("cpu"))
    w = allocate_workspace(torch, g, "cpu")
    first = allocate_cache(torch, g, "cpu", owned=True)
    size = first.numel()
    backing = torch.zeros(size * 2, dtype=torch.uint8)
    overlap = [backing[:size], backing[128 : 128 + size]]
    with pytest.raises(ValueError, match="overlapping"):
        bind_caches(torch, t, g, overlap, workspace=w)
    # The byte count is 128-aligned for supported geometry; adjacent is legal.
    adjacent = [backing[:size], backing[size:]]
    bind_caches(torch, t, g, adjacent, workspace=w)
    w["hot_counts"] = backing[: g.batch * 4].view(torch.int32)
    with pytest.raises(ValueError, match="overlapping"):
        bind_caches(torch, t, g, adjacent, workspace=w)
