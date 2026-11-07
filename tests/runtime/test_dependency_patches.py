"""Exercise dependency upgrades with real, overlapping git patches, without CANN."""

import difflib
import importlib.util
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def dependency(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "scripts/build.py"
    spec = importlib.util.spec_from_file_location("kvip_dependency_build", source)
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    monkeypatch.setattr(build, "ROOT", tmp_path)
    patches = tmp_path / "scripts/patches"
    patches.mkdir(parents=True)
    root = tmp_path / "kvaf"
    root.mkdir()
    (root / "base.txt").write_text("base\n")
    (root / "context.txt").write_text("initial\n")
    (root / "page_mode.txt").write_text("huge\n")
    (root / "local.txt").write_text("unrelated local changes\n")

    names = (*build.PATCHES, build.UBMEM_PATCH, build.UBMEM_FUSED_PATCH, build.UBMEM_PAGE_PATCH)
    changes = (
        ("base.txt", "base", "reuse"),
        ("context.txt", "initial", "fused"),
        ("context.txt", "fused", "views"),
        ("context.txt", "views", "runtime"),
        ("context.txt", "runtime", "ubmem-fused"),
        ("context.txt", "ubmem-fused", "page-mode"),
    )

    def diff(path, before, after):
        return "".join(
            difflib.unified_diff([before + "\n"], [after + "\n"], fromfile=f"a/{path}", tofile=f"b/{path}")
        )

    for name, change in zip(names, changes):
        content = diff(*change)
        if name == build.UBMEM_PAGE_PATCH:
            content += diff("page_mode.txt", "huge", "configurable")
        (patches / (name + ".patch")).write_text(content)

    # Only the pinned-HEAD lookup is stubbed; patch checks/applications use Git.
    # No network checkout or test-only commit identity is needed.
    def pinned_head(command, **kwargs):
        assert command == ["git", "-C", str(root), "rev-parse", "HEAD"]
        return build.REVISION + "\n"

    monkeypatch.setattr(build.subprocess, "check_output", pinned_head)

    def apply_prefix(count):
        for name in names[:count]:
            subprocess.run(["git", "apply", str(patches / (name + ".patch"))], cwd=root, check=True)

    return build, root, apply_prefix


@pytest.mark.parametrize("applied", [0, 3, 4, 5, 6], ids=["fresh", "urma", "runtime", "fused", "page-mode"])
def test_ubmem_upgrade_and_repeat_preserve_local_edits(dependency, applied):
    build, root, apply_prefix = dependency
    apply_prefix(applied)

    build.prepare_dependency(root, "ubmem")
    first = {path.name: path.read_bytes() for path in root.iterdir()}
    build.prepare_dependency(root, "ubmem")

    assert {path.name: path.read_bytes() for path in root.iterdir()} == first
    assert (root / "context.txt").read_text() == "page-mode\n"
    assert (root / "page_mode.txt").read_text() == "configurable\n"
    assert (root / "local.txt").read_text() == "unrelated local changes\n"


@pytest.mark.parametrize("applied", [4, 5, 6], ids=["runtime", "fused", "page-mode"])
def test_urma_rejects_ubmem_tree_without_changes(dependency, applied):
    build, root, apply_prefix = dependency
    apply_prefix(applied)
    before = {path.name: path.read_bytes() for path in root.iterdir()}

    with pytest.raises(RuntimeError, match="separate URMA baseline tree"):
        build.prepare_dependency(root, "urma")

    assert {path.name: path.read_bytes() for path in root.iterdir()} == before


def test_urma_remains_idempotent_without_ubmem_patches(dependency):
    build, root, _ = dependency
    build.prepare_dependency(root, "urma")
    build.prepare_dependency(root, "urma")
    assert (root / "context.txt").read_text() == "views\n"
    assert (root / "page_mode.txt").read_text() == "huge\n"


def test_page_mode_conflict_preserves_fused_tree_and_local_edits(dependency):
    build, root, apply_prefix = dependency
    apply_prefix(5)
    (root / "page_mode.txt").write_text("local page policy\n")
    before = {path.name: path.read_bytes() for path in root.iterdir()}

    with pytest.raises(RuntimeError, match="kvaf-ubmem-page-mode.*checkout preserved"):
        build.prepare_dependency(root, "ubmem")

    assert {path.name: path.read_bytes() for path in root.iterdir()} == before
