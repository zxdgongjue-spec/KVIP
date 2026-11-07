"""Native dependency preparation preserves caller-owned checkouts."""
import importlib.util
import subprocess
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("dependency_build", Path(__file__).parents[2] / "scripts/build.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)

def repository(tmp_path):
    root = tmp_path / "KVAF"
    root.mkdir()
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "--allow-empty", "-m", "base"], check=True, capture_output=True)
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    return root, revision

def test_reuses_native_revision_and_preserves_edits(tmp_path, monkeypatch):
    root, revision = repository(tmp_path)
    monkeypatch.setattr(build, "REVISION", revision)
    (root / "experiment.cc").write_text("keep me")
    for transport in ("urma", "ubmem"):
        build.prepare_dependency(root, transport)
    assert (root / "experiment.cc").read_text() == "keep me"

def test_refuses_wrong_revision_without_checkout(tmp_path, monkeypatch):
    root, revision = repository(tmp_path)
    monkeypatch.setattr(build, "REVISION", "0" * 40)
    with pytest.raises(RuntimeError, match="left unchanged"):
        build.prepare_dependency(root, "ubmem")
    assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == revision
