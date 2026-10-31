"""Regression checks for the verification entrypoints; no hardware benchmarks."""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"tools/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot(root):
    module = tool("verify_baseline")
    for name in module.PROTECTED:
        path = root / name
        path = path / "sample.cpp" if name == "operators" else path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# protected fixture\n")
    (root / "kvop/v9/launch.py").write_text(
        "class Launch:\n    def launch(self):\n        return 1\n"
    )


def git(root, *args):
    return subprocess.check_output(
        [
            "git",
            "-c",
            "user.name=Contract Test",
            "-c",
            "user.email=contract@example.invalid",
            *args,
        ],
        cwd=root,
        text=True,
    ).strip()


def test_git_independent_baseline_is_normalized_from_monorepo_subtree(
    tmp_path, monkeypatch
):
    module = tool("verify_baseline")
    snapshot(tmp_path)
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "test baseline")
    revision = git(tmp_path, "rev-parse", "HEAD")
    subtree = tmp_path / "third_party/KVOP"
    subtree.mkdir(parents=True)
    for name in ("operators", "kvop", "tests"):
        shutil.move(str(tmp_path / name), subtree / name)
    output = tmp_path / "export"
    output.mkdir()
    assert module.extract_git_baseline(subtree, revision, output) == revision
    monkeypatch.setattr(module, "oracle", lambda _: {"fixture": "same"})
    assert module.compare_sources(subtree, output)["identical_host_cases"] == 1
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "test subtree baseline")
    second = tmp_path / "second-export"
    second.mkdir()
    tree = module.extract_git_baseline(subtree, "HEAD", second)
    assert tree == "HEAD:third_party/KVOP"
    assert module.compare_sources(subtree, second)["identical_host_cases"] == 1


def test_no_git_snapshot_is_explicit_and_added_sources_are_not_skipped(
    tmp_path, monkeypatch
):
    module = tool("verify_baseline")
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    snapshot(baseline)
    shutil.copytree(baseline, candidate)
    monkeypatch.setattr(module, "oracle", lambda _: {"fixture": "same"})
    module.main(["--root", str(candidate), "--baseline-directory", str(baseline)])
    (candidate / "operators/untracked.cpp").write_text("new source")
    with pytest.raises(ValueError, match="untracked.cpp"):
        module.compare_sources(candidate, baseline)
    with pytest.raises(ValueError, match="Git/history"):
        module.extract_git_baseline(candidate, "HEAD", tmp_path / "unused")


def test_installed_package_scope_does_not_claim_source_or_hardware_equivalence():
    result = tool("verify_baseline").installed_package_checks()
    assert result["packaged_native_source_hashes"] > 0
    assert isinstance(result["schema_versions"], dict)
    assert result["source_baseline_and_host_equivalence"] == "not_checked_in_this_scope"
    assert result["npu_network_graph_performance"] == "not_verified"


def test_complete_tier_rejects_missing_torch_skips_and_missing_critical_suites():
    module = tool("run_conformance")

    def unavailable(_):
        raise ImportError("torch is unavailable")

    with pytest.raises(ValueError, match="cpu-test"):
        module.require_cpu_torch(unavailable)
    result = module.CompleteRun()
    with pytest.raises(ValueError, match="critical"):
        result.validate("cpu")
    result.collected.update(module.CRITICAL_SUITES)
    result.validate("cpu")
    result.pytest_runtest_logreport(
        SimpleNamespace(skipped=True, nodeid="critical_case")
    )
    with pytest.raises(ValueError, match="skipped"):
        result.validate("cpu")


def test_host_tier_deselects_torch_requirements_before_running_tests():
    module = tool("run_conformance")
    ignored, deselected = module.host_exclusions(ROOT / "tests/v9")
    assert "test_consumer_fake.py" in {path.name for path in ignored}
    assert any(name == "test_real_public_buffer_validation" for _, name in deselected)
    assert all(path.name != "test_public_session.py" for path, _ in deselected)
    with pytest.raises(ModuleNotFoundError, match="host conformance"):
        module.HostOnlyImports().find_spec("torch")
    assert module.HostOnlyImports().find_spec("numpy") is None


def test_host_subprocess_deselects_actual_parametrized_items_before_torch_skip(
    tmp_path,
):
    """Absolute source paths must work even with relative pytest node IDs."""
    tests = tmp_path / "tests/v9"
    tests.mkdir(parents=True)
    package = tmp_path / "kvop"
    package.mkdir()
    (package / "__init__.py").write_text(
        '"""Minimal provider for runner integration."""\n'
    )
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["tests/v9"]\n'
    )
    (tests / "test_mixed.py").write_text("""
import sys
import pytest

def test_host_case():
    assert "torch" not in sys.modules

@pytest.mark.parametrize("value", [1, 2])
def test_torch_case(value):
    pytest.importorskip("torch")
    raise AssertionError("Torch test must be deselected before execution")

def test_other_torch_case():
    import torch
    raise AssertionError("Torch test must be deselected before execution")
""")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/run_conformance.py"),
            "--tier",
            "host",
            "--root",
            str(tmp_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed, 3 deselected" in result.stdout
    assert '"skipped": 0' in result.stdout
    assert '"torch_deselected_cases": 3' in result.stdout
