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


def snapshot(root, namespace="kvip"):
    module = tool("verify_baseline")
    for name in module.PROTECTED:
        parts = Path(name).parts
        path = root.joinpath(namespace, *parts[1:]) if parts[0] == module.PACKAGE else root / name
        path = path / "sample.cpp" if name == "operators" else path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# protected fixture\n")
    (root / namespace / "v9/launch.py").write_text("class Launch:\n    def launch(self):\n        return 1\n")
    (root / namespace / "__init__.py").write_text("")
    (root / namespace / "v9/__init__.py").write_text("")


def rename_snapshot(root, namespace):
    """Represent the authorized package/token rename without altering history."""
    module = tool("verify_baseline")
    if namespace != module.PACKAGE:
        (root / namespace).rename(root / module.PACKAGE)
    for path in root.rglob("*"):
        if path.is_file():
            path.write_bytes(module.normalize_namespace(path.read_bytes(), namespace))


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


def test_git_independent_baseline_is_normalized_from_monorepo_subtree(tmp_path, monkeypatch):
    module = tool("verify_baseline")
    snapshot(tmp_path)
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qsm", "test baseline")
    revision = git(tmp_path, "rev-parse", "HEAD")
    subtree = tmp_path / "third_party/KVIP"
    subtree.mkdir(parents=True)
    for name in ("operators", "kvip", "tests"):
        shutil.move(str(tmp_path / name), subtree / name)
    output = tmp_path / "export"
    output.mkdir()
    assert module.extract_git_baseline(subtree, revision, output) == revision
    monkeypatch.setattr(module, "oracle", lambda _: {"fixture": "same"})
    assert module.compare_sources(subtree, output)["identical_host_cases"] == 1
    (tmp_path / ".gitignore").write_text("export/\nsecond-export/\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qsm", "test subtree baseline")
    second = tmp_path / "second-export"
    second.mkdir()
    tree = module.extract_git_baseline(subtree, "HEAD", second)
    assert tree == "HEAD:third_party/KVIP"
    assert module.compare_sources(subtree, second)["identical_host_cases"] == 1


def test_namespace_migration_preserves_original_oracle_imports_and_reports_normalization(tmp_path, monkeypatch):
    module = tool("verify_baseline")
    historical = "earlier_namespace"
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    snapshot(baseline, historical)
    (baseline / historical / "v9/marker.py").write_text("VALUE = 17\n")
    (baseline / "tests/v9/contract_payload.py").write_text(f"from {historical}.v9.marker import VALUE\n")
    (baseline / "operators/message.py").write_text(
        f'raise RuntimeError("Run from the {historical.upper()} repository")\n'
    )
    shutil.copytree(baseline, candidate)
    rename_snapshot(candidate, historical)
    # Execute two fresh interpreters, retaining each root's own original imports.
    monkeypatch.setattr(
        module, "ORACLE", 'import json\nfrom contract_payload import VALUE\nprint(json.dumps({"fixture": VALUE}))\n'
    )
    result = module.compare_sources(candidate, baseline)
    normalization = result["namespace_normalization"]
    assert normalization["baseline_package"] == historical
    assert normalization["candidate_package"] == "kvip" and normalization["applied"]
    assert normalization["normalized_text_files"] == ["operators/message.py", "tests/v9/contract_payload.py"]
    assert normalization["non_namespace_differences"] == "rejected"
    assert result["identical_host_cases"] == 1
    assert result["device_kernel_sources"] == "byte_identical"
    assert result["device_and_oracle_sources"] == "equivalent_after_namespace_normalization"
    assert historical in (baseline / "tests/v9/contract_payload.py").read_text()
    assert not (baseline / "kvip").exists()


@pytest.mark.parametrize(
    "path,contents,error",
    [
        ("kvip/v9/build.py", "# unrelated source change\n", "protected source difference"),
        ("tests/v9/contract_payload.py", "# numerical change\n", "protected source difference"),
        ("operators/sample.cpp", "// kernel change\n", "device kernel byte difference"),
        ("operators/new.cpp", "// added kernel\n", "device kernel byte difference"),
        ("operators/build_extra.cpp", "// added kernel\n", "device kernel byte difference"),
        ("operators/new.py", "# added operator helper\n", "protected source difference"),
        ("kvip/v9/launch.py", "class Launch:\n    def launch(self):\n        return 2\n", "execution changed"),
    ],
)
def test_namespace_normalization_cannot_hide_other_byte_or_ast_changes(tmp_path, monkeypatch, path, contents, error):
    module = tool("verify_baseline")
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    snapshot(baseline, "historical_namespace")
    shutil.copytree(baseline, candidate)
    rename_snapshot(candidate, "historical_namespace")
    monkeypatch.setattr(module, "oracle", lambda _: {"fixture": "same"})
    (candidate / path).write_text(contents)
    with pytest.raises(ValueError, match=error):
        module.compare_sources(candidate, baseline)


def test_only_standalone_namespace_tokens_are_normalized(tmp_path, monkeypatch):
    module = tool("verify_baseline")
    historical = "earlier_namespace"
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    snapshot(baseline, historical)
    (baseline / historical / "v9/build.py").write_text(f"{historical}_cache = 1\n")
    shutil.copytree(baseline, candidate)
    rename_snapshot(candidate, historical)
    (candidate / "kvip/v9/build.py").write_text("kvip_cache = 1\n")
    monkeypatch.setattr(module, "oracle", lambda _: {"fixture": "same"})
    with pytest.raises(ValueError, match="protected source difference"):
        module.compare_sources(candidate, baseline)


def test_kernel_bytes_remain_strict_even_for_namespace_tokens(tmp_path, monkeypatch):
    module = tool("verify_baseline")
    historical = "earlier_namespace"
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    snapshot(baseline, historical)
    (baseline / "operators/sample.cpp").write_text(f"// {historical.upper()}\n")
    shutil.copytree(baseline, candidate)
    rename_snapshot(candidate, historical)
    monkeypatch.setattr(module, "oracle", lambda _: {"fixture": "same"})
    with pytest.raises(ValueError, match="device kernel byte difference"):
        module.compare_sources(candidate, baseline)


def test_git_history_discovers_historical_subtree_prefix_and_gitfile_worktree(tmp_path, monkeypatch):
    module = tool("verify_baseline")
    historical = "earlier_namespace"
    repository = tmp_path / "framework"
    previous = repository / "third_party/EARLIER_REPOSITORY"
    snapshot(previous, historical)
    git(repository, "init", "-q")
    git(repository, "add", ".")
    git(repository, "commit", "-qsm", "test historical subtree")
    revision = git(repository, "rev-parse", "HEAD")
    current = repository / "third_party/KVIP"
    previous.rename(current)
    rename_snapshot(current, historical)
    git(repository, "add", "-A")
    git(repository, "commit", "-qsm", "test current namespace")
    linked = tmp_path / "linked worktree"
    git(repository, "worktree", "add", "--detach", str(linked), "HEAD")
    assert (linked / ".git").is_file()
    exported = tmp_path / "exported"
    exported.mkdir()
    tree = module.extract_git_baseline(linked / "third_party/KVIP", revision, exported)
    assert tree == f"{revision}:third_party/EARLIER_REPOSITORY"
    assert (exported / historical / "v9/launch.py").is_file()
    monkeypatch.setattr(module, "oracle", lambda _: {"fixture": "same"})
    result = module.compare_sources(linked / "third_party/KVIP", exported)
    assert result["namespace_normalization"]["applied"]


def test_ambiguous_historical_packages_are_rejected(tmp_path):
    module = tool("verify_baseline")
    snapshot(tmp_path / "one", "namespace_one")
    snapshot(tmp_path / "two", "namespace_two")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qsm", "test ambiguous baseline")
    with pytest.raises(ValueError, match="one operator package"):
        module.extract_git_baseline(tmp_path, "HEAD", tmp_path / "export")


def test_no_git_snapshot_is_explicit_and_added_sources_are_not_skipped(tmp_path, monkeypatch):
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
    result.pytest_runtest_logreport(SimpleNamespace(skipped=True, nodeid="critical_case"))
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
    package = tmp_path / "kvip"
    package.mkdir()
    (package / "__init__.py").write_text('"""Minimal provider for runner integration."""\n')
    (tmp_path / "pyproject.toml").write_text('[tool.pytest.ini_options]\ntestpaths = ["tests/v9"]\n')
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
