"""Package boundaries, shared workspace ABI and source-relocation contracts."""

import hashlib
import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_public_and_legacy_imports_share_implementation():
    import kvip
    from kvip.operators.lifetch.forward import LIFetchForward
    from kvip.operators.lifetch.workspace import LIFetchWorkspace
    from kvip.adapters.vllm.inputs import QueryInputs

    assert kvip.LIFetch is LIFetchForward
    assert kvip.FusedWorkspace is kvip.LIFetchWorkspace is LIFetchWorkspace
    assert kvip.FusedQueryInputs is kvip.QueryInputs is QueryInputs
    for old, new, symbol in (
        ("fused_inputs", "adapters.vllm.inputs", "FusedQueryInputs"),
        ("fused_workspace", "operators.lifetch.workspace", "FusedWorkspace"),
        ("pipeline", "operators.lifetch.forward", "FusedForward"),
        ("_lifetch.launch", "operators.lifetch.binding", "RotatingLaunch"),
    ):
        legacy = importlib.import_module("kvip." + old)
        current = importlib.import_module("kvip." + new)
        assert legacy is current
        assert getattr(legacy, symbol) is not None
    assert not (ROOT / "python/kvop").exists()


def test_workspace_spec_matches_all_reserved_buffers():
    import torch
    from kvip.operators.lifetch.workspace import LIFetchWorkspace, rotation_spec
    from kvip.operators.lifetch.geometry import li_geometry, final_aivs

    for batch in (1, 8, 16):
        workspace = LIFetchWorkspace(
            torch,
            batch=batch,
            history=4096,
            aics=32,
            windows=8,
            device=torch.device("cpu"),
            pin_memory=False,
            index_heads=32,
        )
        workspace.allocate_rotation(torch, batch=batch, history=4096, resident=2048, aics=32, windows=8)
        geo = li_geometry(batch, intermediate_fetches=8)
        spec = rotation_spec(
            batch, 4096, 2048, final_aivs(batch), geo["intermediate_fetches"], geo["service_aivs_per_window"], 1216
        )
        assert workspace.scratch.keys() == spec.keys()
        for name, (shape, dtype) in spec.items():
            assert tuple(workspace.scratch[name].shape) == shape
            assert workspace.scratch[name].dtype == getattr(torch, dtype)


def test_relocated_cpp_bodies_match_pre_refactor_sources():
    manifest = json.loads((ROOT / "scripts/relocation.json").read_text())
    for name, expected in manifest["body_sha256"].items():
        text = (ROOT / name).read_text()
        body = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#include"))
        assert hashlib.sha256(body.encode()).hexdigest() == expected, name


def test_actual_operator_cmake_resolves_relocated_sources(tmp_path):
    import subprocess

    cann = tmp_path / "cann"
    module = cann / "compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake"
    module.parent.mkdir(parents=True)
    module.write_text(
        "function(ascendc_library)\n add_library(${ARGV})\nendfunction()\n"
        "function(ascendc_include_directories)\n target_include_directories(${ARGV})\nendfunction()\n"
        "function(ascendc_compile_options)\nendfunction()\n"
        "add_library(host_intf_pub INTERFACE)\nadd_library(ascendcl INTERFACE)\n"
    )
    headers = tmp_path / "headers"
    headers.mkdir()
    (headers / "kvaf_batch_get_device.h").touch()
    source = ROOT / "csrc/operators/lightning_indexer_with_fetch"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(tmp_path / "build"),
            f"-DASCEND_CANN_PACKAGE_PATH={cann}",
            f"-DKVAF_INCLUDE={headers}",
            "-DRESOLVER_COMPARE=1",
        ],
        check=True,
        capture_output=True,
    )
    includes = [
        source / "include",
        source / "op_kernel",
        source / "op_kernel/indexer",
        ROOT / "tests/operators/lifetch",
    ]
    import re

    for cpp in source.rglob("*"):
        if cpp.suffix not in (".cpp", ".h"):
            continue
        for include in re.findall(r'^#include "([^"]+)"', cpp.read_text(), re.M):
            if include in (
                "api.h",
                "context.h",
                "runtime_shape.h",
                "prefetch.h",
                "resolver.h",
                "index_cache_update.h",
            ) or include.startswith("resolver_reference/"):
                assert any((folder / include).is_file() for folder in [cpp.parent, *includes]), (cpp, include)
