"""Consumer-visible contract and regression tests, including the audited gaps."""

import ast
import ctypes
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest
from contract_fixture import make_fixture, to_device

from kvip.v9 import (
    ABI_VERSION,
    CONTRACT_VERSION,
    ContractMismatchError,
    ContractRequirements,
    Geometry,
    LayoutSpec,
    admit_request,
    negotiate_contract,
    query_capabilities,
    validate_request,
)
from kvip.v9.contract import DeviceContract, allocate_workspace, validate_buffers
from kvip.v9.index_cache import allocate_cache, bind_caches


def test_machine_manifest_and_public_modules_have_no_reverse_framework_dependency():
    import kvip
    from kvip.v9.contract_info import describe_contract

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
    for path in Path(kvip.__file__).parent.rglob("*.py"):
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
        "3.0.0",
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
            "from kvip.v9 import query_capabilities; import sys; "
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
        negotiate_contract(
            replace(
                ContractRequirements(features=query_capabilities().features),
                **{field: value},
            )
        )


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


@pytest.mark.parametrize("features", [frozenset(), frozenset({"eager"})])
def test_unused_graph_and_index_writer_abis_do_not_restrict_eager_consumers(features):
    info = query_capabilities()
    assert (
        negotiate_contract(
            ContractRequirements(
                features=features, graph_transport_abi=99, index_writer_abi=99
            ),
            provider=info,
        )
        is info
    )


@pytest.mark.parametrize("feature", ["fixed-q4-graph", "owned-index"])
def test_each_optional_feature_negotiates_only_its_own_abi(feature):
    info = query_capabilities()
    required = ContractRequirements(
        features=frozenset({feature}),
        graph_transport_abi=info.graph_transport_abi
        if feature == "fixed-q4-graph"
        else 99,
        index_writer_abi=info.index_writer_abi if feature == "owned-index" else 99,
    )
    assert negotiate_contract(required, provider=info) is info
    field = "graph_transport_abi" if feature == "fixed-q4-graph" else "index_writer_abi"
    with pytest.raises(ContractMismatchError):
        negotiate_contract(replace(required, **{field: 99}), provider=info)


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


def admission_geometry(batch, candidate, budget, tail):
    """Construct metadata at large limits without allocating device-sized pools."""
    candidate_capacity = (candidate + 255) // 256 * 256
    resident_capacity = (budget + 255) // 256 * 256
    index_columns = candidate_capacity // 128
    resident_columns = (resident_capacity + tail + 127) // 128
    resident_blocks = batch * resident_columns
    return Geometry(
        batch=batch,
        queries=4,
        candidate_capacity=candidate_capacity,
        resident_capacity=resident_capacity,
        pool_capacity=batch + 1,
        index_columns=index_columns,
        resident_columns=resident_columns,
        index_blocks=batch * index_columns,
        resident_blocks=resident_blocks,
        layer_id=0,
        num_layers=1,
        max_requests=batch,
        qcap=index_columns,
        table_capacity=batch * index_columns,
        output_tokens=resident_blocks * 128,
        score_stride=candidate_capacity + 128,
    ).validate()


@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("candidate", [1, 262144])
@pytest.mark.parametrize("budget", [2048, 16384])
@pytest.mark.parametrize("tail", [0, 128, 129])
def test_host_admission_accepts_real_boundary_combinations(
    batch, candidate, budget, tail
):
    g = admission_geometry(batch, candidate, budget, tail)
    request = dict(
        queries_per_request=4,
        candidate_lengths=[candidate] * batch,
        resident_budgets=[budget] * batch,
        logical_lengths=[candidate + tail] * batch,
        pool_rows=list(range(batch)),
        geometry=g,
        aic_count=max(1, batch // 2),
    )
    result = admit_request(**request)
    assert result.supported, (result.reason, result.message)
    validate_request(**request)


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"candidate_lengths": [], "resident_budgets": []}, "batch_outside_v9"),
        ({"resident_budgets": [2047]}, "invalid_metadata"),
        ({"resident_budgets": [16385]}, "invalid_metadata"),
        ({"logical_lengths": [-1]}, "invalid_metadata"),
        ({"logical_lengths": [2**31]}, "invalid_metadata"),
        ({"pool_rows": [2**31]}, "invalid_metadata"),
        ({"pool_rows": [False]}, "invalid_metadata"),
        ({"aic_count": 33}, "insufficient_aiv"),
        ({"aic_count": True}, "insufficient_aiv"),
        ({"candidate_lengths": 16384}, "invalid_metadata"),
        ({"resident_budgets": []}, "invalid_metadata"),
        ({"logical_lengths": []}, "invalid_metadata"),
        ({"pool_rows": []}, "invalid_metadata"),
    ],
)
def test_host_admission_rejects_missing_metadata_and_values_just_outside_limits(
    changes, reason
):
    request = dict(
        queries_per_request=4, candidate_lengths=[16384], resident_budgets=[12288]
    )
    result = admit_request(**(request | changes))
    assert not result.supported and result.reason == reason


@pytest.mark.parametrize(
    "field", ["candidate_lengths", "resident_budgets", "logical_lengths"]
)
def test_host_admission_requires_each_length_and_budget_to_be_uniform(field):
    request = dict(
        queries_per_request=4,
        candidate_lengths=[16384, 16384],
        resident_budgets=[12288, 12288],
        logical_lengths=[16513, 16513],
    )
    request[field][1] += 1
    result = admit_request(**request)
    assert not result.supported and result.reason == "nonuniform"


def test_host_admission_resource_threshold_and_prepared_capacity():
    request = dict(
        queries_per_request=4,
        candidate_lengths=[16384] * 3,
        resident_budgets=[12288] * 3,
        logical_lengths=[16384] * 3,
        pool_rows=[0, 1, 2],
        geometry=admission_geometry(3, 16384, 12288, 0),
    )
    assert not admit_request(**request, aic_count=1).supported
    assert admit_request(**request, aic_count=2).supported
    assert not admit_request(
        **(request | {"candidate_lengths": [16385] * 3, "logical_lengths": [16385] * 3})
    ).supported
    assert not admit_request(**(request | {"resident_budgets": [12289] * 3})).supported
    assert not admit_request(**(request | {"logical_lengths": [16385] * 3})).supported


def test_numpy_host_scalars_do_not_wrap_tail_capacity_arithmetic():
    g = admission_geometry(1, 16384, 12288, 0)
    result = admit_request(
        queries_per_request=np.int32(4),
        candidate_lengths=np.array([16384], np.int32),
        resident_budgets=np.array([12288], np.int32),
        logical_lengths=np.array([np.iinfo(np.int32).max], np.int32),
        pool_rows=np.array([0], np.int32),
        geometry=g,
    )
    assert not result.supported and result.reason == "capacity"
