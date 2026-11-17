"""Missing or mismatched bridge exports fail before any replay allocation."""

import subprocess
from types import SimpleNamespace

import pytest

from kvip.diagnostics import scan_options
from kvip.runtime import build


@pytest.mark.parametrize("actual,expected", [(0, False), (1, True)])
def test_real_export_value(tmp_path, actual, expected):
    source = tmp_path / "bridge.cpp"
    source.write_text(
        f'extern "C" unsigned kvip_incremental_inplace() {{ return {actual}; }}\n'
    )
    library = tmp_path / "bridge.so"
    subprocess.run(
        ["c++", "-shared", "-fPIC", str(source), "-o", str(library)], check=True
    )
    scan_options.verify_library(library, {"incremental_inplace": expected})
    with pytest.raises(RuntimeError, match="does not match requested incremental_inplace"):
        scan_options.verify_library(library, {"incremental_inplace": not expected})
    with pytest.raises(RuntimeError, match="kvip_missing"):
        scan_options.verify_library(library, {"missing": False})


def test_missing_abi_stops_preflight(tmp_path, monkeypatch):
    args = SimpleNamespace(build_dir=tmp_path, soc="A5", scan_variants=["forward", "reverse"])
    built = []
    monkeypatch.setattr(build, "kvaf_build_paths", lambda path: (path, path))
    monkeypatch.setattr(build, "build_libraries", lambda **kw: built.append(kw) or (tmp_path / "absent.so",))
    retained = {}
    with pytest.raises(RuntimeError, match="ABI preflight failed"):
        scan_options.prebuild(args, retained)
    assert len(built) == 1
    assert retained["progress"] == dict(variant="forward", phase="abi_preflight", step=None)
