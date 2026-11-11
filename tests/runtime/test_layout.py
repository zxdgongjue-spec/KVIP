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

    selector.write_text("// updated shared LIDU selector\n")
    current = build.build_libraries(**options)
    assert current != updated
    assert len(commands) == 8
    compact = build.build_libraries(**options, compact_reserved=True)
    assert compact != current
    assert "-DCOMPACT_RESERVED=1" in commands[-2]
    assert build.build_libraries(**options, compact_reserved=True) == compact
    assert build.build_libraries(**options) == current
    assert len(commands) == 10
    parallel = build.build_libraries(**options, parallel_final=True)
    assert parallel != current and parallel != compact
    assert "-DURMA_PARALLEL_FINAL=1" in commands[-2]
    assert build.build_libraries(**options, parallel_final=True) == parallel
    assert build.build_libraries(**options) == current
    assert len(commands) == 12
    asynchronous = build.build_libraries(**options, parallel_final=True, async_final=True)
    assert asynchronous != parallel
    assert "-DURMA_ASYNC_FINAL=1" in commands[-2]
    assert build.build_libraries(**options, parallel_final=True, async_final=True) == asynchronous
    with pytest.raises(ValueError, match="requires parallel_final"):
        build.build_libraries(**options, async_final=True)
    assert len(commands) == 14
    for index, flag in enumerate(
        ("cold_only_prefetch", "threshold_only", "simt_init", "staged_outputs", "rolling_threshold"), 1
    ):
        specialized = build.build_libraries(**options, **{flag: True})
        assert specialized != current
        assert f"-D{flag.upper()}=1" in commands[-2]
        assert build.build_libraries(**options, **{flag: True}) == specialized
        with pytest.raises(ValueError, match="must be booleans"):
            build.build_libraries(**options, **{flag: 1})
        assert len(commands) == 14 + index * 2

    for flag in ("single_planner", "skip_final_partition"):
        specialized = build.build_libraries(**options, parallel_final=True, **{flag: True})
        assert specialized != parallel
        assert f"-DURMA_{flag.upper()}=1" in commands[-2]
        with pytest.raises(ValueError, match="require parallel_final"):
            build.build_libraries(**options, **{flag: True})
        with pytest.raises(ValueError, match="without async_final"):
            build.build_libraries(**options, parallel_final=True, async_final=True, **{flag: True})
    with pytest.raises(ValueError, match="resident TopK seed"):
        build.build_libraries(**options, rolling_threshold=True, threshold_only=True)
    posted = build.build_libraries(**options, parallel_final=True, post_final_before_index=True)
    assert posted != parallel
    assert "-DPOST_FINAL_BEFORE_INDEX=1" in commands[-2]
    assert build.build_libraries(**options, parallel_final=True, post_final_before_index=True) == posted
    for invalid in (
        dict(post_final_before_index=True),
        dict(parallel_final=True, async_final=True, post_final_before_index=True),
        dict(parallel_final=True, post_final_before_index=1),
    ):
        with pytest.raises(ValueError):
            build.build_libraries(**options, **invalid)
    merged = build.build_libraries(
        **options, parallel_final=True, post_final_before_index=True, merge_final_completion=True
    )
    assert merged != posted
    assert "-DMERGE_FINAL_COMPLETION=1" in commands[-2]
    for invalid in (
        dict(merge_final_completion=True),
        dict(parallel_final=True, merge_final_completion=True),
        dict(parallel_final=True, post_final_before_index=True, merge_final_completion=1),
    ):
        with pytest.raises(ValueError):
            build.build_libraries(**options, **invalid)
    bitmap = build.build_libraries(**options, bitmap_index=True)
    assert bitmap != compact
    assert "-DBITMAP_INDEX=1" in commands[-2]
    assert build.build_libraries(**options, bitmap_index=True) == bitmap
    with pytest.raises(ValueError, match="booleans"):
        build.build_libraries(**options, bitmap_index=1)
    staged_reserved = build.build_libraries(**options, compact_reserved=True, reserved_ub=True)
    assert staged_reserved != compact
    assert "-DRESERVED_UB=1" in commands[-2]
    for invalid in (
        dict(reserved_ub=1),
        dict(reserved_ub=True),
        dict(reserved_ub=True, compact_reserved=True, simt_init=True),
    ):
        with pytest.raises(ValueError):
            build.build_libraries(**options, **invalid)


@pytest.mark.parametrize("bitmap_index", (0, 1))
def test_actual_operator_cmake_resolves_relocated_sources(tmp_path, bitmap_index):
    import subprocess

    cann = tmp_path / "cann"
    module = cann / "compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake"
    module.parent.mkdir(parents=True)
    module.write_text(
        "function(ascendc_library)\n add_library(${ARGV})\nendfunction()\n"
        "function(ascendc_include_directories)\n target_include_directories(${ARGV})\nendfunction()\n"
        "function(ascendc_compile_options)\n target_compile_options(${ARGV})\n"
        ' file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/${ARGV0}-options.txt" '
        'CONTENT "$<TARGET_PROPERTY:${ARGV0},COMPILE_OPTIONS>")\nendfunction()\n'
        "add_library(host_intf_pub INTERFACE)\nadd_library(ascendcl INTERFACE)\n"
        'file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/bridge-definitions.txt" '
        'CONTENT "$<TARGET_PROPERTY:v9_bridge,COMPILE_DEFINITIONS>")\n'
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
            f"-DBITMAP_INDEX={bitmap_index}",
        ],
        check=True,
        capture_output=True,
    )
    # The Index planner is a separate native target. A bridge getter or a
    # top-level configure option alone must not masquerade as kernel support.
    update_options = (tmp_path / "build/kvop_cache_update_kernel-options.txt").read_text().split(";")
    bridge_definitions = (tmp_path / "build/bridge-definitions.txt").read_text().split(";")
    assert f"-DV9_BITMAP_INDEX={bitmap_index}" in update_options
    assert f"V9_BITMAP_INDEX={bitmap_index}" in bridge_definitions
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
