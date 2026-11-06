"""Package boundaries, shared workspace ABI and current-source build contracts."""

import importlib
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


def test_build_cache_reuses_and_rebuilds_current_sources(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from kvip.runtime import build

    source = tmp_path / "csrc/operators/lightning_indexer_with_fetch"
    source.mkdir(parents=True)
    kernel = source / "prefetch.h"
    kernel.write_text("// baseline kernel\n", encoding="utf-8")
    headers = tmp_path / "kvaf-headers"
    headers.mkdir()
    header = headers / "kvaf_batch_get_device.h"
    header.write_text("// baseline interface\n", encoding="utf-8")
    monkeypatch.setattr(build, "__file__", str(tmp_path / "python/kvip/runtime/build.py"))
    commands = []

    def compile_libraries(command, **kwargs):
        commands.append(command)
        if command[1] == "--build":
            directory = Path(command[2])
            for name in ("libv9_bridge.so", "libkvop_cache_update.so"):
                (directory / name).write_bytes(b"test library")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(build.subprocess, "run", compile_libraries)
    options = dict(cache_dir=tmp_path / "cache", cann=tmp_path / "cann", headers=headers, soc="Ascend950PR_9599")
    original = build.build_libraries(**options)
    assert build.build_libraries(**options) == original
    assert len(commands) == 2

    kernel.write_text("// optimized kernel\n", encoding="utf-8")
    optimized = build.build_libraries(**options)
    assert optimized != original
    assert all(path.is_file() for path in optimized)

    header.write_text("// updated interface\n", encoding="utf-8")
    updated = build.build_libraries(**options)
    assert updated != optimized
    assert all(path.is_file() for path in updated)
    assert len(commands) == 6


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
