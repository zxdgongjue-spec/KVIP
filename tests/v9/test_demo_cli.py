"""Standalone demo CLI regressions, including real host execution and child status."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/run_demo.py"


def run(*arguments, env=None):
    return subprocess.run(
        [sys.executable, str(TOOL), *arguments], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120
    )


def test_real_host_demo_uses_local_provider_and_checks_payload(tmp_path):
    poison = tmp_path / "other-provider/kvip"
    poison.mkdir(parents=True)
    (poison / "__init__.py").write_text('raise RuntimeError("must not import inherited provider")\n')
    output = tmp_path / "host demo"
    result = run("--output", str(output), env=dict(os.environ, PYTHONPATH=str(poison.parent)))
    assert result.returncode == 0, result.stdout + result.stderr
    summaries = list(output.glob("*/summary.json"))
    assert len(summaries) == 1
    reports = json.loads(summaries[0].read_text())
    assert len(reports) == 3
    assert all(report["status"] == "pass" for report in reports)
    assert all(report["backend"] == "cpu" and report["performance_status"] == "not_measured" for report in reports)


def test_host_dry_run_records_exact_benchmark_sources_and_output(tmp_path):
    output = tmp_path / "not-created"
    result = run("--dry-run", "--case-suite", "full", "--output", str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    plan = json.loads(result.stdout)
    assert plan["command"][:2] == [sys.executable, str(ROOT / "tests/v9/benchmark_contract.py")]
    assert plan["case_suite"] == "full"
    assert (
        plan["benchmark"]["sha256"]
        == hashlib.sha256((ROOT / "tests/v9/benchmark_contract.py").read_bytes()).hexdigest()
    )
    assert plan["operator_sources"]["sha256"]["contract_abi.h"]
    assert plan["kvip"]["package_version"]
    assert plan["output"] == str(output.resolve())
    assert plan["qualification"]["hardware_function_and_performance"] == "not_qualified_by_this_demo"
    assert not output.exists()


@pytest.mark.parametrize(
    "arguments,reason",
    [
        (("--profile", "unknown"), "invalid choice"),
        (("--profile", "host", "--execution", "graph"), "host cannot validate NPUGraph"),
        (("--profile", "npu-memory", "--dry-run"), "explicit external --kvaf-root"),
        (("--profile", "kvaf", "--kvaf-root", "/missing", "--dry-run"), "requires --kvaf-server"),
    ],
)
def test_invalid_modes_fail_instead_of_skipping(arguments, reason):
    result = run(*arguments)
    assert result.returncode == 2
    assert reason in result.stderr


def test_external_header_paths_and_missing_sdk_are_explicit(tmp_path):
    root = tmp_path / "KVAF"
    arguments = ("--profile", "npu-memory", "--kvaf-root", str(root))
    result = run(*arguments, "--dry-run")
    assert result.returncode != 0 and "KVAF directory does not exist" in result.stderr
    root.mkdir()
    result = run(*arguments, "--dry-run")
    assert result.returncode != 0 and "device header is missing" in result.stderr
    header = root / "access_operator/demo/aiv_batch_get/include/kvaf_batch_get_device.h"
    header.parent.mkdir(parents=True)
    header.write_text("// source-only fixture, never a runtime provider\n")
    result = run(*arguments, "--execution", "graph", "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    plan = json.loads(result.stdout)
    assert plan["kvaf"]["device_header_sha256"] == hashlib.sha256(header.read_bytes()).hexdigest()
    assert plan["kvaf"]["graph_scope"] == "standalone-single-output-NPUGraph"
    assert plan["kvaf"]["framework_graph_frame_required"] is False
    assert plan["kvaf"]["runtime_certified"] is False
    result = run(*arguments, "--cann", str(tmp_path / "missing-CANN"))
    assert result.returncode != 0
    assert "SDK/compiler/runtime files are missing" in result.stderr


def test_child_failure_exit_code_is_preserved():
    spec = importlib.util.spec_from_file_location("kvip_demo_cli", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.run_child([sys.executable, "-c", "raise SystemExit(23)"], ROOT) == 23


@pytest.mark.parametrize("execution", ["eager", "graph"])
def test_kvaf_dry_run_uses_shared_source_evidence_without_certifying_runtime(tmp_path, execution):
    from kvip.v9 import kvaf_compatibility as compatibility

    root = tmp_path / "KVAF"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    headers, exports = compatibility._requirements("v9-eager")
    for name in ("CMakeLists.txt", "build.sh", *headers):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// source-only test fixture\n")
    binding = root / "bindings/src/npu_python_runtime.cc"
    binding.parent.mkdir(parents=True)
    binding.write_text("\n".join(f'.def("{name}", &Runtime::{name})' for name in exports))
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Demo Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "source-only demo fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    arguments = (
        "--profile",
        "kvaf",
        "--execution",
        execution,
        "--kvaf-root",
        str(root),
        "--kvaf-server",
        "127.0.0.1:28960",
        "--dry-run",
    )
    result = run(*arguments, "--expected-kvaf-revision", revision)
    assert result.returncode == 0, result.stdout + result.stderr
    plan = json.loads(result.stdout)
    evidence = plan["kvaf"]["source_evidence"]
    assert evidence["kind"] == "source-only" and evidence["status"] == "pass"
    assert evidence["revision"] == revision and evidence["profile"] == "v9-eager"
    assert evidence["runtime_validated"] is False and evidence["hardware_certified"] is False
    assert "begin_fused_graph" not in evidence["required_exports"]
    assert plan["qualification"]["runtime_validated"] is False
    assert plan["qualification"]["runtime_surface_checked"] is False
    result = run(*arguments, "--expected-kvaf-revision", "0" * 40)
    assert result.returncode != 0 and "pinned commit" in result.stderr
    binding.write_text(binding.read_text().replace("acquire_fused_batch_get_context", "not_an_eager_export"))
    result = run(*arguments)
    assert result.returncode != 0 and "missing pybind exports" in result.stderr


def test_runtime_probe_passes_the_runtime_class_and_records_only_surface_evidence(tmp_path, monkeypatch):
    from kvip.v9 import kvaf_compatibility as compatibility

    bundle = tmp_path / "KVAF/build-npu"
    python_dir = bundle / "python"
    package = python_dir / "kvaf"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# explicitly fake CPU fixture\n")
    (package / "_kvaf_fake.so").touch()
    (bundle / "bin").mkdir()
    for name in ("kvaf_execute_batch_get_kernels.o", "kvaf_execute_batch_put_kernels.o"):
        (bundle / "bin" / name).touch()

    def forbidden(*args, **kwargs):
        raise AssertionError("surface probing must not acquire or initialize a runtime")

    _, exports = compatibility._requirements("v9-eager")
    runtime_type = type("FixtureRuntime", (), dict.fromkeys(exports, forbidden))
    runtime = SimpleNamespace(__file__=str(package / "__init__.py"), NpuRuntime=runtime_type)
    monkeypatch.setitem(sys.modules, "kvaf", runtime)
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location("kvip_demo_runtime_cli", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.require_kvaf_runtime(ROOT, SimpleNamespace(kvaf_server="test-only", kvaf_root=tmp_path / "KVAF"))
    assert result["module_path"] == str((package / "__init__.py").resolve())
    assert result["surface"]["kind"] == "runtime-surface-only"
    assert result["surface"]["runtime_validated"] is False
    assert result["surface"]["hardware_certified"] is False
