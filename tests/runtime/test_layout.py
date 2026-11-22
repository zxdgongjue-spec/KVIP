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
    assert build.build_libraries(**options) != updated
    assert len(commands) == 8


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


def test_terminal_queues_are_disjoint_and_rebind_output():
    import struct
    import pytest
    from kvip.operators.lifetch.terminal_queues import bind_queues, wqe_template

    def view(i):
        words = [0] * 48
        words[6:8] = [0xABCDEF, 0x98765432]
        words[12] = 0xDEADBEEF
        words[14:18] = [0x100000 + i * 4096, 0x12, 0x200000 + i * 128, 0x3F]
        words[19:22] = [8192, 0xF1234567, 64]
        words[22:26] = [1, 2, 3, 4]
        words[26:30] = [0x300000 + i * 4096, 0x3F, 0x400000 + i * 128, 0x3F]
        return dict(
            entity_address=0x120000100000 + i * 192,
            queue_address=0x120000200000 + i * 64,
            output_address=0x123456789ABC,
            entity_bytes=struct.pack("<48I", *words),
        )

    for batch in (1, 2, 4, 8, 16):
        bindings = bind_queues(batch, [view(i) for i in range(batch * 5)], 656)
        used = {b.view for b in bindings}
        prefetch = set(range(0, 2 * batch, 2))
        assert len(used) == batch * 4 and not used & prefetch
        assert used | prefetch == set(range(batch * 5))
    v = view(0)
    a = wqe_template(v["entity_bytes"], v["output_address"], 656)
    assert a[12:18] == (656, 0xDEADBEEF, 0x56789ABC, 0x1234, 0, 8192)
    assert a[18:20] == (0, 0)  # Live queue heads must not be captured.
    b = wqe_template(v["entity_bytes"], 0x130012345678, 656)
    assert a[14:16] != b[14:16] and a[:14] + a[16:] == b[:14] + b[16:]
    for offset in (7, 8, 13, 14):
        views = [view(i) for i in range(5)]
        words = list(struct.unpack("<24Q", views[1]["entity_bytes"]))
        words[offset] = struct.unpack("<24Q", views[0]["entity_bytes"])[offset]
        views[1]["entity_bytes"] = struct.pack("<24Q", *words)
        with pytest.raises(ValueError, match="exclusive"):
            bind_queues(1, views, 656)


def test_parallel_terminal_same_graph_accounting():
    from diagnostics import account, decode_terminal_timeline
    from performance import split_graph_invocations

    events = [dict(name="replay", ph="X", ts=0, dur=100)]
    for origin in (1, 41):
        for index, (name, duration) in enumerate(
            (("v9_li_kernel_0", 5), ("v9_li_kernel_1_1", 5), ("v9_li_kernel_2_2", 5), ("v9_parallel_terminal_0", 15))
        ):
            events.append(dict(name=name, ph="X", cat="kernel", ts=origin + index * 5, dur=duration))
    accounting = account(events, "replay", 100, full=True, fused_final=True)
    assert accounting["stages_us"] == {"li": 30, "topk_resolve_update": 30}
    result = split_graph_invocations(accounting, 2)
    assert len(result["samples"]) == 2
    t = [0] * 32
    for index, slot in enumerate((0, 1, 2, 3, 11, 7, 4, 5, 8, 16, 17, 10), 1):
        t[slot] = index * 1000
    timeline = decode_terminal_timeline([[t, t, t, t]], 1000)
    assert len(timeline["cores"]) == 4
    assert timeline["observed_span_us"] == 11
    for core in timeline["cores"]:
        assert sum(s["duration_us"] for s in core["stages"]) == core["duration_us"]


def test_parallel_forward_uses_live_stream_without_rebinding(monkeypatch):
    from types import SimpleNamespace
    from kvip.operators.lifetch.forward import LIFetchForward
    from kvip.operators.lifetch.streaming import StreamingLIFetchForward

    calls = []
    forward = LIFetchForward.__new__(LIFetchForward)
    forward.bindings = SimpleNamespace(data_ptr=lambda: 101)
    forward.terminal = SimpleNamespace(workspace=SimpleNamespace(data_ptr=lambda: 202))
    forward.shape_pointer = 303
    stream = SimpleNamespace(npu_stream=404)
    forward.rotating = SimpleNamespace(owner=SimpleNamespace(lease=SimpleNamespace(check_stream=lambda: stream)))
    forward.terminal_function = lambda *args: calls.append(args) or 0
    forward()
    stream.npu_stream = 405
    forward()
    assert calls == [(101, 202, 404, 303), (101, 202, 405, 303)]
    forward.terminal = None
    monkeypatch.setattr(StreamingLIFetchForward, "__call__", lambda self: calls.append("legacy"))
    forward()
    assert calls[-1] == "legacy"


def test_terminal_counts_do_not_invent_retired_admission_identities():
    import numpy as np
    from diagnostics import algorithm_counts
    from correctness import transfer_coverage

    mapping = np.full((1, 4096), -1, dtype=np.int32)
    selected = np.tile(np.arange(2048), (1, 4, 1))
    rows = np.zeros((1, 256), dtype=np.uint32)
    rows[0, 9] = rows[0, 16] = 2
    rows[0, 247] = 3
    posts = np.zeros((1, 4, 32), dtype=np.int32)
    posts[0, 0, 0] = 3
    result = algorithm_counts(mapping, selected, rows, posts, 2048, 4, admission_records=False)
    assert result["totals"]["prefetch"] == 2 and result["totals"]["final"] == 3
    assert result["totals"]["unused_prefetch"] is None
    assert result["requests"][0]["windows"][0]["unused"] is None
    slots = np.array([[1, 2, 2048, 2049, 2050]])
    coverage = transfer_coverage(posts, [], np.arange(2048)[None], 8192, 4096, allocated_slots=slots, prior_counts=[2])
    assert coverage[0]["final"] == 3 and coverage[0]["fallback_slots"] == 3
