"""Temporary Git fixtures for source provenance and side-effect-free probing."""

import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kvip.v9 import kvaf_compatibility as compatibility


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "kvaf"
    root.mkdir()
    git(root, "init", "-q")
    headers, exports = compatibility._requirements("v9-eager")
    for name in ("CMakeLists.txt", "build.sh", *headers):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture source\n")
    binding = root / compatibility._BINDING
    binding.parent.mkdir(parents=True)
    binding.write_text("\n".join(f'.def("{name}", &Runtime::{name})' for name in exports))
    git(root, "add", ".")
    git(
        root,
        "-c",
        "user.name=Compatibility Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    return root


def test_source_evidence_checks_exact_commit_and_reports_only_source_provenance(checkout):
    sha = git(checkout, "rev-parse", "HEAD")
    evidence = compatibility.source_evidence(checkout, sha)
    assert evidence["status"] == "pass" and evidence["root"] == str(checkout)
    assert evidence["revision"] == sha and evidence["kind"] == "source-only"
    assert not evidence["dirty"] and evidence["source_tree_matches_revision"]
    assert not evidence["hardware_certified"] and not evidence["runtime_validated"]
    assert len(evidence["file_sha256"]) == 7
    with pytest.raises(compatibility.KvafCompatibilityError, match="pinned commit"):
        compatibility.source_evidence(checkout, "0" * 40)


@pytest.mark.parametrize("invalid", ["HEAD", "7c54bd5", "a" * 39, "x" * 40, 1, True])
def test_expected_commit_must_be_a_complete_sha(checkout, invalid):
    with pytest.raises(compatibility.KvafCompatibilityError, match="complete 40"):
        compatibility.source_evidence(checkout, invalid)


def test_real_gitfile_worktree_is_accepted(checkout, tmp_path):
    worktree = tmp_path / "linked-checkout"
    git(checkout, "worktree", "add", "--detach", "-q", str(worktree), "HEAD")
    assert (worktree / ".git").is_file()
    evidence = compatibility.source_evidence(worktree, git(checkout, "rev-parse", "HEAD"))
    assert evidence["root"] == str(worktree) and not evidence["dirty"]


@pytest.mark.parametrize("invalid", ["subdirectory", "no_git", "missing"])
def test_source_path_must_identify_the_actual_checkout_root(checkout, tmp_path, invalid):
    if invalid == "subdirectory":
        path = checkout / "bindings"
    else:
        path = tmp_path / invalid
        if invalid == "no_git":
            path.mkdir()
    with pytest.raises(compatibility.KvafCompatibilityError):
        compatibility.source_evidence(path)


def test_dirty_source_is_explicitly_fingerprinted_without_modification(checkout):
    target = checkout / "build.sh"
    target.write_text("changed fixture\n")
    before = git(checkout, "status", "--porcelain=v1")
    evidence = compatibility.source_evidence(checkout)
    assert evidence["dirty"] and evidence["tracked_changes"]
    assert not evidence["source_tree_matches_revision"]
    assert git(checkout, "status", "--porcelain=v1") == before
    assert target.read_text() == "changed fixture\n"


def test_raw_eager_source_does_not_claim_frame_graph_and_reports_bridge_requirement(checkout):
    with pytest.raises(compatibility.KvafCompatibilityError, match="fused_graph_abi") as failure:
        compatibility.source_evidence(checkout, profile="v9-frame-graph")
    assert failure.value.evidence["missing_exports"] == list(compatibility._GRAPH_EXPORTS)
    assert failure.value.evidence["graph_bridge_required"]


def test_frame_graph_source_exports_still_do_not_certify_the_sdk(checkout):
    path = checkout / compatibility._BINDING
    with path.open("a") as stream:
        for name in compatibility._GRAPH_EXPORTS:
            stream.write(f'\n.def_static("{name}", &Runtime::{name})')
    evidence = compatibility.source_evidence(checkout, profile="v9-frame-graph")
    assert evidence["dirty"] and not evidence["graph_bridge_required"]
    assert not evidence["runtime_validated"] and not evidence["hardware_certified"]


@pytest.mark.parametrize(
    "missing",
    ["build.sh", "CMakeLists.txt", compatibility._BINDING, compatibility._INCLUDE + "kvaf_batch_get_device.h"],
)
def test_missing_source_files_are_reported_before_any_build(checkout, missing):
    (checkout / missing).unlink()
    with pytest.raises(compatibility.KvafCompatibilityError) as failure:
        compatibility.source_evidence(checkout)
    assert missing in failure.value.evidence["missing_files"]


def test_commented_out_pybind_exports_are_not_treated_as_capabilities(checkout):
    path = checkout / compatibility._BINDING
    text = path.read_text().replace(
        '.def("acquire_fused_batch_get_context", &Runtime::acquire_fused_batch_get_context)',
        '/* .def("acquire_fused_batch_get_context", &Runtime::acquire_fused_batch_get_context) */',
    )
    text += '\n// .def("acquire_fused_batch_get_context", &Runtime::acquire_fused_batch_get_context)\n'
    path.write_text(text)
    with pytest.raises(compatibility.KvafCompatibilityError) as failure:
        compatibility.source_evidence(checkout)
    assert failure.value.evidence["missing_exports"] == ["acquire_fused_batch_get_context"]


def test_legacy_profile_does_not_require_fused_headers_or_methods(checkout):
    for name in compatibility._FUSED_HEADERS:
        (checkout / compatibility._INCLUDE / name).unlink()
    path = checkout / compatibility._BINDING
    path.write_text("\n".join(f'.def("{name}", &Runtime::{name})' for name in compatibility._LEGACY_EXPORTS))
    assert compatibility.source_evidence(checkout, profile="legacy")["status"] == "pass"
    with pytest.raises(compatibility.KvafCompatibilityError):
        compatibility.source_evidence(checkout)


def runtime_surface(profile="v9-eager"):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append("side effect")
        raise AssertionError("capability probing must not execute this method")

    _, exports = compatibility._requirements(profile)
    runtime = SimpleNamespace(**dict.fromkeys(exports, forbidden))
    runtime.fused_graph_abi = lambda: 1
    return runtime, calls


@pytest.mark.parametrize("profile", compatibility.PROFILES)
def test_runtime_surface_checks_never_execute_acquire_initialize_or_other_operations(profile):
    runtime, calls = runtime_surface(profile)
    evidence = compatibility.validate_runtime_surface(runtime, profile)
    assert evidence["status"] == "pass" and evidence["kind"] == "runtime-surface-only"
    assert not evidence["runtime_validated"] and not evidence["hardware_certified"]
    assert calls == []


def test_runtime_class_can_be_checked_without_constructing_a_runtime():
    runtime, calls = runtime_surface()
    runtime_type = type("Runtime", (), vars(runtime))
    assert compatibility.validate_runtime_surface(runtime_type)["status"] == "pass"
    assert calls == []


@pytest.mark.parametrize("abi", [True, 0, 2, 1.0, "1", None])
def test_runtime_graph_abi_is_checked_without_fallback(abi):
    runtime, calls = runtime_surface("v9-frame-graph")
    runtime.fused_graph_abi = lambda: abi
    with pytest.raises(compatibility.KvafCompatibilityError, match="integer 1"):
        compatibility.validate_runtime_surface(runtime, "v9-frame-graph")
    assert calls == []


def test_runtime_missing_or_noncallable_export_is_rejected():
    runtime, calls = runtime_surface()
    runtime.acquire_fused_batch_get_context = None
    with pytest.raises(compatibility.KvafCompatibilityError, match="acquire_fused"):
        compatibility.validate_runtime_surface(runtime)
    assert calls == []


def test_instance_bound_graph_abi_requires_an_existing_instance():
    runtime, _ = runtime_surface("v9-frame-graph")
    attributes = vars(runtime) | {"fused_graph_abi": lambda self: 1}
    runtime_type = type("Runtime", (), attributes)
    with pytest.raises(compatibility.KvafCompatibilityError, match="existing instance"):
        compatibility.validate_runtime_surface(runtime_type, "v9-frame-graph")
    assert compatibility.validate_runtime_surface(runtime_type(), "v9-frame-graph")["graph_transport_abi"] == 1


@pytest.mark.parametrize("profile", ["unsupported", "npu-memory", "v9-graph"])
def test_unsupported_profile_never_silently_falls_back(checkout, profile):
    with pytest.raises(compatibility.KvafCompatibilityError, match="unsupported"):
        compatibility.source_evidence(checkout, profile=profile)
    with pytest.raises(compatibility.KvafCompatibilityError, match="unsupported"):
        compatibility.validate_runtime_surface(object(), profile)


@pytest.mark.parametrize("profile,returncode", [("v9-eager", 0), ("v9-frame-graph", 1), ("unsupported", 1)])
def test_cli_returns_machine_readable_evidence_and_nonzero_on_failure(checkout, profile, returncode):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kvip.v9.kvaf_compatibility",
            "--root",
            str(checkout),
            "--profile",
            profile,
            "--expected-revision",
            git(checkout, "rev-parse", "HEAD"),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == returncode, result.stderr
    evidence = json.loads(result.stdout)
    assert evidence["status"] == ("pass" if returncode == 0 else "fail")


def test_helper_has_only_standard_library_imports_and_does_not_load_torch():
    source = Path(compatibility.__file__).read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] in sys.stdlib_module_names for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module.split(".")[0] in sys.stdlib_module_names
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from kvip.v9.kvaf_compatibility import source_evidence; import sys; "
            "assert not any(name == 'torch' or name.startswith(('torch_npu', 'kvaf', 'vllm')) for name in sys.modules)",
        ],
        check=True,
    )
