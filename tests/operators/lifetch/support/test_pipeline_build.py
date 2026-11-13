# SPDX-License-Identifier: Apache-2.0
"""Different native experiments must never reuse one another's cached binary."""

import json
from pathlib import Path
from types import SimpleNamespace

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
    for option in ("final_topk_prune", "selected_metadata", "early_fetch_ready", "compact_init", "ready_batch_fetch", "ready_direct_ranges", "topk_prune_diagnostics"):
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


@pytest.mark.parametrize("value", (1, 0, "OFF", None))
def test_pipeline_options_reject_ambiguous_values_before_build(tmp_path, value):
    with pytest.raises(ValueError, match="must be bool"):
        build.build_libraries(cache_dir=tmp_path, cann=tmp_path, headers=tmp_path, soc="A5", compact_init=value)
