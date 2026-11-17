# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from kvip.diagnostics import replay, replay_isolated


def arguments(tmp_path):
    return replay.arguments([
        "--snapshot", str(tmp_path / "input"), "--build-dir", str(tmp_path / "build"),
        "--transport", "urma", "--mode", "fused", "--threshold-policy", "recent",
        "--study", "scan-order", "--scan-variants", "forward", "reverse", "forward-end",
        "--upstream-root", str(tmp_path / "source"), "--upstream-revision", "a" * 40,
        "--kvaf-server", "127.0.0.1:29093", "--device", "0", "--soc", "Ascend950DT_9582",
        "--cold-windows", "8", "--warmup", "5", "--iterations", "10",
        "--diagnostic-replays", "1", "--isolate-variants", "--quiet",
        "--output", str(tmp_path / "result.json"),
    ])


@pytest.mark.parametrize("failure", [None, "oom", "selection"])
def test_isolated_children_keep_controls_and_retain_failure_evidence(tmp_path, monkeypatch, failure):
    args = arguments(tmp_path)
    calls = []
    compiled = []

    def preflight(arguments, retained):
        compiled.extend(arguments.scan_variants)

    monkeypatch.setattr(replay_isolated, "preflight", preflight)

    def child(command, stdout, stderr):
        assert compiled == args.scan_variants
        parsed = replay.arguments(command[6:])
        assert not parsed.isolate_variants
        assert parsed.cold_windows == 8 and parsed.device == 0
        assert parsed.warmup == 5 and parsed.iterations == 10 and parsed.diagnostic_replays == 1
        assert parsed.upstream_revision == args.upstream_revision
        name, = parsed.scan_variants
        calls.append(name)
        if name == "reverse" and failure == "oom":
            parsed.output.write_text(json.dumps(dict(
                status="failed", progress=dict(variant=name, phase="bundle_create"),
                error=dict(type="OutOfMemoryError", message="memory exhausted"),
                memory_samples=[dict(checkpoint="failure", free_bytes=0, allocated_bytes=42, reserved_bytes=42)],
            )))
            return SimpleNamespace(returncode=70)
        case = dict(
            status="pass", timing=dict(mean_us=123, p50_us=120),
            selection_comparison=dict(ordered_output_sha256="changed" if failure == "selection" and name == "reverse" else "same"),
            threshold_comparison=dict(initial_keys=[1], final_keys=[2]),
        )
        parsed.output.write_text(json.dumps(dict(status="pass", cases={name: case}, execution_order=[name])))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(replay_isolated.subprocess, "run", child)
    if failure is None:
        replay_isolated.run(args)
        report = json.loads(args.output.read_text())
        assert report["status"] == "pass"
        assert report["execution_order"] == calls == args.scan_variants
        assert report["process_model"] == "one_fresh_process_per_variant_sequential"
        assert not list(tmp_path.glob(".result-variants.*"))
    else:
        with pytest.raises(SystemExit, match="70"):
            replay_isolated.run(args)
        report = json.loads(args.output.read_text())
        assert report["status"] == "failed"
        assert Path(report["isolated_directory"]).is_dir()
        if failure == "oom":
            assert calls == ["forward", "reverse"]
            assert list(report["completed_cases"]) == ["forward"]
            assert report["progress"]["phase"] == "bundle_create"
            assert report["memory_samples"][0]["free_bytes"] == 0
        else:
            assert "changed the final ordered TopK" in report["error"]["message"]


def test_child_command_does_not_forward_isolation(tmp_path):
    args = arguments(tmp_path)
    cmd = replay_isolated.command(args, "reverse", tmp_path / "child.json")
    # Interpreter + -B + -X + faulthandler + -m + module are six arguments.
    assert cmd[:6] == [replay_isolated.sys.executable, "-B", "-X", "faulthandler", "-m", "kvip.diagnostics.replay"]
    child = replay.arguments(cmd[6:])
    assert child.scan_variants == ["reverse"] and not child.isolate_variants


def test_compile_failure_starts_no_device_children(tmp_path, monkeypatch):
    args = arguments(tmp_path)

    def preflight(arguments, retained):
        retained["progress"].update(variant="reverse-incremental")
        os.write(2, b"compiler: ambiguous enum\n")
        raise RuntimeError("compile failed\n--- build diagnostic ---\nerror: ambiguous enum")

    monkeypatch.setattr(replay_isolated, "preflight", preflight)
    monkeypatch.setattr(replay_isolated.subprocess, "run", lambda *a, **k: pytest.fail("device child started"))
    with pytest.raises(SystemExit, match="70"):
        replay_isolated.run(args)
    report = json.loads(args.output.read_text())
    assert report["completed_cases"] == {}
    assert report["progress"] == dict(variant="reverse-incremental", phase="compile_preflight", step=None)
    assert "error: ambiguous enum" in report["error"]["message"]
    native_log = Path(report["native_log"])
    assert "compiler: ambiguous enum" in native_log.read_text()


def test_preflight_selects_pinned_production_before_build(tmp_path, monkeypatch):
    from kvip.diagnostics import scan_options, upstream

    args = arguments(tmp_path)
    retained, calls = {}, []

    def activate(root, revision):
        assert root == args.upstream_root and revision == args.upstream_revision
        calls.append("activate")

    def prebuild(arguments, state):
        assert calls == ["activate"] and arguments is args and state is retained
        calls.append("build")

    monkeypatch.setattr(upstream, "activate", activate)
    monkeypatch.setattr(scan_options, "prebuild", prebuild)
    replay_isolated.preflight(args, retained)
    assert calls == ["activate", "build"]
