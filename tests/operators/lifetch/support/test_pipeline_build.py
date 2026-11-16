# SPDX-License-Identifier: Apache-2.0
"""Different native experiments must never reuse one another's cached binary."""

import fcntl
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from kvip.runtime import build


def test_pipeline_options_isolate_and_reuse_the_correct_binary(monkeypatch, tmp_path):
    calls = []

    def compile_command(command, **kwargs):
        calls.append(command)
        if command[1] == "--build":
            (Path(command[2]) / "libv9_bridge.so").write_bytes(b"compiled fixture")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(build.subprocess, "run", compile_command)
    arguments = dict(cache_dir=tmp_path / "cache", cann=tmp_path, headers=tmp_path / "headers", soc="A5")
    baseline = build.build_libraries(**arguments)
    assert build.build_libraries(**arguments) == baseline
    assert len(calls) == 2
    paths = {baseline}
    options = (
        "final_topk_prune", "selected_metadata", "early_fetch_ready", "compact_init",
        "ready_batch_fetch", "ready_direct_ranges", "adaptive_admission", "topk_prune_diagnostics",
        "skip_exhausted_prefetch", "early_prefetch_drain", "inverse_mapping",
    )
    for option in options:
        result = build.build_libraries(**arguments, **{option: True})
        assert result not in paths
        paths.add(result)
        manifest = json.loads((result[0].parent / "complete.json").read_text())
        assert manifest["KVIP_" + option.upper()] == "ON"
        enabled = [key for key, value in manifest.items() if key.startswith("KVIP_") and value == "ON"]
        assert enabled == ["KVIP_" + option.upper()]
        previous = len(calls)
        assert build.build_libraries(**arguments, **{option: True}) == result
        assert len(calls) == previous
    gated = build.build_libraries(**arguments, adaptive_admission=True, prefetch_gate=True)
    assert gated not in paths
    manifest = json.loads((gated[0].parent / "complete.json").read_text())
    assert manifest["KVIP_ADAPTIVE_ADMISSION"] == manifest["KVIP_PREFETCH_GATE"] == "ON"
    previous = len(calls)
    assert build.build_libraries(**arguments, adaptive_admission=True, prefetch_gate=True) == gated
    assert len(calls) == previous


@pytest.mark.parametrize("value", (1, 0, "OFF", None))
@pytest.mark.parametrize("option", ("compact_init", "adaptive_admission", "prefetch_gate",
                                    "skip_exhausted_prefetch", "early_prefetch_drain", "inverse_mapping",
                                    "low_miss_skip", "low_miss_tail", "auto_route", "low_miss_sparse_init",
                                    "auto_route_fast_probe", "auto_route_row_publish",
                                    "auto_route_simt", "low_miss_direct_index",
                                    "low_miss_compact_plan"))
def test_pipeline_options_reject_ambiguous_values_before_build(tmp_path, value, option):
    with pytest.raises(ValueError, match="must be bool"):
        build.build_libraries(cache_dir=tmp_path, cann=tmp_path, headers=tmp_path, soc="A5", **{option: value})


def test_prefetch_gate_requires_admission_before_build(monkeypatch, tmp_path):
    compile_ = Mock()
    monkeypatch.setattr(build.subprocess, "run", compile_)
    with pytest.raises(ValueError, match="prefetch_gate requires adaptive_admission"):
        build.build_libraries(cache_dir=tmp_path, cann=tmp_path, headers=tmp_path, soc="A5", prefetch_gate=True)
    compile_.assert_not_called()


def test_low_miss_skip_requires_admission_and_isolates_binary(monkeypatch, tmp_path):
    calls = []

    def compile_command(command, **kwargs):
        calls.append(command)
        if command[1] == "--build":
            (Path(command[2]) / "libv9_bridge.so").write_bytes(b"fixture")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(build.subprocess, "run", compile_command)
    args = dict(cache_dir=tmp_path, cann=tmp_path, headers=tmp_path, soc="A5")
    with pytest.raises(ValueError, match="low_miss_skip requires adaptive_admission"):
        build.build_libraries(**args, low_miss_skip=True)
    assert not calls
    ordinary = build.build_libraries(**args, adaptive_admission=True)
    light = build.build_libraries(**args, adaptive_admission=True, low_miss_skip=True)
    assert ordinary != light
    manifest = json.loads((light[0].parent / "complete.json").read_text())
    assert manifest["KVIP_LOW_MISS_SKIP"] == "ON"
    previous = len(calls)
    assert build.build_libraries(**args, adaptive_admission=True, low_miss_skip=True) == light
    assert len(calls) == previous


def test_low_miss_tail_requires_light_path_and_isolates_binary(monkeypatch, tmp_path):
    calls = []

    def compile_command(command, **kwargs):
        calls.append(command)
        if command[1] == "--build":
            (Path(command[2]) / "libv9_bridge.so").write_bytes(b"fixture")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(build.subprocess, "run", compile_command)
    args = dict(cache_dir=tmp_path, cann=tmp_path, headers=tmp_path, soc="A5")
    with pytest.raises(ValueError, match="low_miss_tail requires low_miss_skip"):
        build.build_libraries(**args, low_miss_tail=True)
    assert not calls
    light = dict(adaptive_admission=True, low_miss_skip=True)
    before = build.build_libraries(**args, **light)
    after = build.build_libraries(**args, **light, low_miss_tail=True)
    assert before != after
    manifest = json.loads((after[0].parent / "complete.json").read_text())
    assert manifest["KVIP_LOW_MISS_TAIL"] == "ON"
    count = len(calls)
    assert build.build_libraries(**args, **light, low_miss_tail=True) == after
    assert len(calls) == count


def test_auto_route_requires_complete_tail_and_isolates_binary(monkeypatch, tmp_path):
    calls = []

    def compile_command(command, **kwargs):
        calls.append(command)
        if command[1] == "--build":
            (Path(command[2]) / "libv9_bridge.so").write_bytes(b"fixture")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(build.subprocess, "run", compile_command)
    args = dict(cache_dir=tmp_path, cann=tmp_path, headers=tmp_path, soc="A5")
    with pytest.raises(ValueError, match="auto_route requires"):
        build.build_libraries(**args, auto_route=True)
    assert not calls
    light = dict(adaptive_admission=True, low_miss_skip=True,
                 ready_batch_fetch=True, early_fetch_ready=True)
    manual = build.build_libraries(**args, **light)
    automatic = build.build_libraries(**args, **light, auto_route=True)
    assert manual != automatic
    manifest = json.loads((automatic[0].parent / "complete.json").read_text())
    assert manifest["KVIP_AUTO_ROUTE"] == "ON"
    previous = len(calls)
    assert build.build_libraries(**args, **light, auto_route=True) == automatic
    assert len(calls) == previous
    sparse_flags = dict(auto_route=True, low_miss_tail=True, selected_metadata=True, low_miss_sparse_init=True)
    for missing in ("auto_route", "low_miss_tail", "selected_metadata"):
        with pytest.raises(ValueError, match="low_miss_sparse_init requires"):
            build.build_libraries(**args, **light, **(sparse_flags | {missing: False}))
    assert len(calls) == previous
    sparse = build.build_libraries(**args, **light, **sparse_flags)
    assert sparse != automatic
    manifest = json.loads((sparse[0].parent / "complete.json").read_text())
    assert manifest["KVIP_LOW_MISS_SPARSE_INIT"] == "ON"
    assert build.build_libraries(**args, **light, **sparse_flags) == sparse
    assert len(calls) == previous + 2
    with pytest.raises(ValueError, match="separate initialization policies"):
        build.build_libraries(**args, **light, **sparse_flags, inverse_mapping=True)
    assert len(calls) == previous + 2
    with pytest.raises(ValueError, match="auto_route_fast_probe requires"):
        build.build_libraries(**args, **light, auto_route_fast_probe=True)
    assert len(calls) == previous + 2
    fast = build.build_libraries(**args, **light, **sparse_flags, auto_route_fast_probe=True)
    assert fast != sparse
    manifest = json.loads((fast[0].parent / "complete.json").read_text())
    assert manifest["KVIP_AUTO_ROUTE_FAST_PROBE"] == "ON"
    assert build.build_libraries(**args, **light, **sparse_flags, auto_route_fast_probe=True) == fast
    assert len(calls) == previous + 4

    with pytest.raises(ValueError, match="auto_route_row_publish requires"):
        build.build_libraries(**args, **light, auto_route_row_publish=True)
    with pytest.raises(ValueError, match="row publication requires complete evaluation"):
        build.build_libraries(**args, **light, **sparse_flags,
                              auto_route_row_publish=True, auto_route_fast_probe=True)
    row_flags = light | sparse_flags | dict(auto_route_row_publish=True)
    row = build.build_libraries(**args, **row_flags)
    assert row not in (automatic, sparse, fast)
    direct = build.build_libraries(**args, **row_flags, low_miss_direct_index=True)
    assert direct not in (automatic, sparse, fast, row)
    manifest = json.loads((direct[0].parent / "complete.json").read_text())
    assert manifest["KVIP_AUTO_ROUTE_ROW_PUBLISH"] == "ON"
    assert manifest["KVIP_LOW_MISS_DIRECT_INDEX"] == "ON"
    count = len(calls)
    assert build.build_libraries(**args, **row_flags, low_miss_direct_index=True) == direct
    assert len(calls) == count
    with pytest.raises(ValueError, match="auto_route_simt requires"):
        build.build_libraries(**args, **light, **sparse_flags, auto_route_simt=True)
    assert len(calls) == count
    parallel_flags = row_flags | dict(low_miss_direct_index=True, auto_route_simt=True)
    parallel = build.build_libraries(**args, **parallel_flags)
    compact = build.build_libraries(**args, **parallel_flags, low_miss_compact_plan=True)
    read_through = build.build_libraries(**args, **row_flags, low_miss_read_through=True)
    assert read_through not in (automatic, sparse, fast, row, direct, parallel, compact)
    recorded = json.loads((read_through[0].parent / "complete.json").read_text())
    assert recorded["KVIP_LOW_MISS_READ_THROUGH"] == "ON"
    assert recorded["KVIP_AUTO_ROUTE_SIMT"] == "OFF"
    assert recorded["KVIP_LOW_MISS_COMPACT_PLAN"] == "OFF"
    assert len({automatic, sparse, fast, row, direct, parallel, compact}) == 7
    manifest = json.loads((compact[0].parent / "complete.json").read_text())
    assert manifest["KVIP_AUTO_ROUTE_SIMT"] == "ON"
    assert manifest["KVIP_LOW_MISS_COMPACT_PLAN"] == "ON"
    count = len(calls)
    assert build.build_libraries(**args, **parallel_flags, low_miss_compact_plan=True) == compact
    assert len(calls) == count


@pytest.mark.parametrize("feature", ("low_miss_compact_plan", "low_miss_read_through"))
@pytest.mark.parametrize("missing", ("auto_route", "low_miss_tail", "selected_metadata", "early_fetch_ready"))
def test_compact_plan_requires_complete_auto_path(monkeypatch, tmp_path, missing, feature):
    compile_ = Mock()
    monkeypatch.setattr(build.subprocess, "run", compile_)
    flags = dict(adaptive_admission=True, low_miss_skip=True, low_miss_tail=True,
                 selected_metadata=True, ready_batch_fetch=True, early_fetch_ready=True,
                 auto_route=True)
    flags[feature] = True
    flags[missing] = False
    with pytest.raises(ValueError, match="requires"):
        build.build_libraries(cache_dir=tmp_path, cann=tmp_path, headers=tmp_path,
                              soc="A5", **flags)
    compile_.assert_not_called()


@pytest.mark.parametrize("missing", ("low_miss_tail", "selected_metadata", "early_fetch_ready"))
def test_direct_index_requires_proved_streaming_path(monkeypatch, tmp_path, missing):
    compile_ = Mock()
    monkeypatch.setattr(build.subprocess, "run", compile_)
    flags = dict(adaptive_admission=True, low_miss_skip=True, low_miss_tail=True,
                 selected_metadata=True, early_fetch_ready=True, low_miss_direct_index=True)
    flags[missing] = False
    with pytest.raises(ValueError, match="low_miss_direct_index requires"):
        build.build_libraries(cache_dir=tmp_path, cann=tmp_path, headers=tmp_path,
                              soc="A5", **flags)
    compile_.assert_not_called()


@pytest.mark.parametrize("rank", (-1, 2049, True, False, 1536.0, "1536", None))
def test_prefetch_rank_rejects_invalid_values(tmp_path, rank):
    with pytest.raises(ValueError, match="prefetch_rank must be an integer"):
        build.build_libraries(cache_dir=tmp_path, cann=tmp_path, headers=tmp_path,
                              soc="A5", prefetch_rank=rank)


def test_prefetch_rank_requires_gate_and_isolates_binary(monkeypatch, tmp_path):
    calls = []

    def compile_command(command, **kwargs):
        calls.append(command)
        if command[1] == "--build":
            (Path(command[2]) / "libv9_bridge.so").write_bytes(b"fixture")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(build.subprocess, "run", compile_command)
    args = dict(cache_dir=tmp_path / "cache", cann=tmp_path,
                headers=tmp_path / "headers", soc="A5", adaptive_admission=True)
    with pytest.raises(ValueError, match="prefetch_rank requires prefetch_gate"):
        build.build_libraries(**args, prefetch_rank=1536)
    assert not calls
    paths = set()
    for rank in (0, 2048, 1792, 1536):
        result = build.build_libraries(**args, prefetch_gate=True, prefetch_rank=rank)
        assert result not in paths
        paths.add(result)
        manifest = json.loads((result[0].parent / "complete.json").read_text())
        assert manifest["KVIP_PREFETCH_RANK"] == str(rank)
        count = len(calls)
        assert build.build_libraries(**args, prefetch_gate=True, prefetch_rank=rank) == result
        assert len(calls) == count


def test_interrupted_build_cleans_only_failed_variant_without_replacing_lock(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("not a build artifact")
    directory = None
    attempt = 0
    lock_inode = None

    def compile_command(command, **kwargs):
        nonlocal directory, attempt, lock_inode
        build_dir = Path(command[2] if command[1] == "--build" else command[command.index("-B") + 1])
        assert len(kwargs["pass_fds"]) == 1
        assert os.fstat(kwargs["pass_fds"][0]).st_ino == (build_dir / "build.lock").stat().st_ino
        if command[1] != "--build":
            if attempt:
                assert not (directory / "preprocess").exists()
                assert not (directory / "outside-link").exists()
                assert not (directory / "libv9_bridge.so").exists()
                assert (directory / "build.lock").stat().st_ino == lock_inode
                with (directory / "build.lock").open() as probe:
                    with pytest.raises(BlockingIOError):
                        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return SimpleNamespace(returncode=0)
        directory = Path(command[2])
        attempt += 1
        if attempt == 1:
            lock_inode = (directory / "build.lock").stat().st_ino
            stale = directory / "preprocess"
            stale.mkdir()
            (stale / "indexer.cpp.o").touch()
            (stale / "build.stamp").touch()
            (directory / "libv9_bridge.so").touch()
            (directory / "outside-link").symlink_to(outside, target_is_directory=True)
            return SimpleNamespace(returncode=2)
        (directory / "libv9_bridge.so").write_bytes(b"rebuilt fixture")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(build.subprocess, "run", compile_command)
    args = dict(cache_dir=cache, cann=tmp_path, headers=tmp_path / "headers", soc="A5")
    with pytest.raises(RuntimeError, match="compile/link failed"):
        build.build_libraries(**args)
    assert not (directory / "complete.json").exists()
    neighbor = cache / "other-variant"
    neighbor.mkdir()
    (neighbor / "keep").write_text("valid cached binary")
    result = build.build_libraries(**args)
    assert result[0].read_bytes() == b"rebuilt fixture"
    assert (neighbor / "keep").read_text() == "valid cached binary"
    assert (outside / "keep").read_text() == "not a build artifact"
    assert (directory / "build.lock").stat().st_ino == lock_inode
    assert build.build_libraries(**args) == result
    assert attempt == 2


@pytest.mark.parametrize("invalid", ("missing", "truncated", "different_options", "empty_library"))
def test_invalid_complete_record_rebuilds_instead_of_reusing_binary(monkeypatch, tmp_path, invalid):
    calls = []

    def compile_command(command, **kwargs):
        calls.append(command)
        if command[1] == "--build":
            (Path(command[2]) / "libv9_bridge.so").write_bytes(b"fixture")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(build.subprocess, "run", compile_command)
    args = dict(cache_dir=tmp_path / "cache", cann=tmp_path, headers=tmp_path / "headers", soc="A5")
    (library,) = build.build_libraries(**args)
    marker = library.parent / "complete.json"
    expected = json.loads(marker.read_text())
    if invalid == "missing":
        marker.unlink()
    elif invalid == "truncated":
        marker.write_text('{"RUN_MODE":')
    elif invalid == "different_options":
        marker.write_text(json.dumps(dict(expected, SOC_VERSION="wrong")))
    else:
        library.write_bytes(b"")
    assert build.build_libraries(**args) == (library,)
    assert len(calls) == 4
    assert json.loads(marker.read_text()) == expected
    assert library.stat().st_size > 0
    assert not list(library.parent.glob(".complete-*"))


def test_complete_marker_is_published_atomically(monkeypatch, tmp_path):
    replace = Path.replace
    observed = []

    def publish(path, destination):
        assert path.name.startswith(".complete-")
        assert not destination.exists()
        assert json.loads(path.read_text()) == {"SOC_VERSION": "A5"}
        observed.append(path)
        return replace(path, destination)

    monkeypatch.setattr(Path, "replace", publish)
    build._publish_complete(tmp_path, {"SOC_VERSION": "A5"})
    assert len(observed) == 1
    assert json.loads((tmp_path / "complete.json").read_text()) == {"SOC_VERSION": "A5"}
    assert not list(tmp_path.glob(".complete-*"))
