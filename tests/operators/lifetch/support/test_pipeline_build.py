# SPDX-License-Identifier: Apache-2.0
"""Different native experiments must never reuse one another's cached binary."""

import json
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
                                    "low_miss_skip", "low_miss_tail", "auto_route"))
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
