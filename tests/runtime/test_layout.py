"""Package boundaries, shared workspace ABI and current-source build contracts."""

import importlib
from pathlib import Path

import pytest

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
    selector = source.parent / "lidu/op_kernel/hot_slot_selection.h"
    selector.parent.mkdir(parents=True)
    selector.write_text("// shared LIDU selector\n")
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
            for name in ("libv9_bridge.so",):
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

    selector.write_text("// updated shared LIDU selector\n")
    current = build.build_libraries(**options)
    assert current != updated
    assert len(commands) == 8
    before = build.build_libraries(**options, recent_pipeline=False, recent_unified_topk=False)
    pipeline = build.build_libraries(**options, recent_pipeline=True, recent_unified_topk=False)
    assert len({current, before, pipeline}) == 3
    assert build.build_libraries(**options) == current
    assert len(commands) == 12
    assert "-DKVIP_RECENT_PIPELINE=0" in commands[-4]
    assert "-DKVIP_RECENT_UNIFIED_TOPK=0" in commands[-4]
    assert "-DKVIP_RECENT_PIPELINE=1" in commands[-2]
    assert "-DKVIP_RECENT_UNIFIED_TOPK=0" in commands[-2]
    with_skip = build.build_libraries(**options, recent_skip=True)
    assert with_skip not in {current, before, pipeline}
    assert "-DKVIP_RECENT_SKIP=1" in commands[-2]
    assert "-DKVIP_RECENT_SKIP=0" in commands[-4]
    assert build.build_libraries(**options, recent_skip=True) == with_skip
    assert len(commands) == 14
    threshold_only = build.build_libraries(**options, recent_threshold_only=True)
    assert threshold_only not in {current, before, pipeline, with_skip}
    assert "-DKVIP_RECENT_THRESHOLD_ONLY=1" in commands[-2]
    assert "-DKVIP_RECENT_THRESHOLD_ONLY=0" in commands[-4]
    assert build.build_libraries(**options, recent_threshold_only=True) == threshold_only
    assert len(commands) == 16
    with pytest.raises(ValueError, match="threshold-only"):
        build.build_libraries(**options, recent_threshold_only=1)
    release = build.build_libraries(**options, recent_threshold_only=True, recent_profile=False)
    seed = build.build_libraries(**options, recent_threshold_only=True, recent_profile=False, recent_seed_ub=True)
    assert len({threshold_only, release, seed}) == 3
    assert "-DKVIP_RECENT_PROFILE=0" in commands[-4]
    assert "-DKVIP_RECENT_SEED_UB=0" in commands[-4]
    assert "-DKVIP_RECENT_PROFILE=0" in commands[-2]
    assert "-DKVIP_RECENT_SEED_UB=1" in commands[-2]
    assert (
        build.build_libraries(**options, recent_threshold_only=True, recent_profile=False, recent_seed_ub=True) == seed
    )
    assert len(commands) == 20
    with pytest.raises(ValueError, match="requires threshold-only"):
        build.build_libraries(**options, recent_seed_ub=True)
    for invalid in ({"recent_profile": 0}, {"recent_seed_ub": 1}):
        with pytest.raises(ValueError, match="must be boolean"):
            build.build_libraries(**options, **invalid)
    local_victims = build.build_libraries(
        **options, recent_threshold_only=True, recent_profile=False, recent_local_victims=True
    )
    assert local_victims not in {threshold_only, release, seed}
    assert "-DKVIP_RECENT_LOCAL_VICTIMS=1" in commands[-2]
    assert "-DKVIP_RECENT_LOCAL_VICTIMS=0" in commands[-4]
    assert (
        build.build_libraries(**options, recent_threshold_only=True, recent_profile=False, recent_local_victims=True)
        == local_victims
    )
    assert len(commands) == 22
    with pytest.raises(ValueError, match="requires threshold-only"):
        build.build_libraries(**options, recent_local_victims=True)
    with pytest.raises(ValueError, match="must be boolean"):
        build.build_libraries(**options, recent_local_victims=1)
    register_kth = build.build_libraries(
        **options, recent_threshold_only=True, recent_profile=False, recent_local_victims=True, recent_register_kth=True
    )
    assert register_kth not in {threshold_only, release, seed, local_victims}
    assert "-DKVIP_RECENT_REGISTER_KTH=1" in commands[-2]
    assert "-DKVIP_RECENT_REGISTER_KTH=0" in commands[-4]
    assert (
        build.build_libraries(
            **options,
            recent_threshold_only=True,
            recent_profile=False,
            recent_local_victims=True,
            recent_register_kth=True,
        )
        == register_kth
    )
    assert len(commands) == 24
    with pytest.raises(ValueError, match="requires threshold-only"):
        build.build_libraries(**options, recent_register_kth=True)
    with pytest.raises(ValueError, match="must be boolean"):
        build.build_libraries(**options, recent_register_kth=1)
    overlap_flags = dict(
        recent_threshold_only=True,
        recent_profile=False,
        recent_local_victims=True,
        recent_register_kth=True,
        recent_overlap_victims=True,
    )
    overlap = build.build_libraries(**options, **overlap_flags)
    assert overlap not in {threshold_only, release, seed, local_victims, register_kth}
    assert "-DKVIP_RECENT_OVERLAP_VICTIMS=1" in commands[-2]
    assert "-DKVIP_RECENT_OVERLAP_VICTIMS=0" in commands[-4]
    assert build.build_libraries(**options, **overlap_flags) == overlap
    assert len(commands) == 26
    for required in ("recent_register_kth", "recent_local_victims"):
        with pytest.raises(ValueError, match="overlap-victims requires"):
            build.build_libraries(**options, **(overlap_flags | {required: False}))
    with pytest.raises(ValueError, match="must be boolean"):
        build.build_libraries(**options, recent_overlap_victims=1)


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
            ):
                assert any((folder / include).is_file() for folder in [cpp.parent, *includes]), (cpp, include)


def test_ubmem_previous_import_paths_resolve_to_merged_modules():
    from kvip.adapters.ubmem import UbmemLease as old_lease
    from kvip.adapters.kvaf import UbmemLease
    from kvip.adapters.vllm.ubmem_binding import bind_selectors as old_bind, prepare_workspace as old_prepare
    from kvip.adapters.vllm.binding import bind_selectors, prepare_workspace

    assert old_lease is UbmemLease
    assert old_bind is bind_selectors
    assert old_prepare is prepare_workspace
