"""Compile and call every builder option's actual host metadata export."""

import ctypes
import inspect
from pathlib import Path
import re
import subprocess

import pytest

from kvip.runtime.build import build_libraries


@pytest.mark.parametrize("enabled", [False, True])
def test_all_builder_options_are_exported_and_callable(tmp_path, enabled):
    root = Path(__file__).resolve().parents[2]
    source = (root / "csrc/operators/lightning_indexer_with_fetch/op_host/launch.cpp").read_text()
    options = {
        name for name, parameter in inspect.signature(build_libraries).parameters.items()
        if type(parameter.default) is bool
    }
    functions = re.findall(
        r'extern "C" uint32_t (kvip_\w+)\(\) \{ return (KVIP_\w+); \}', source
    )
    assert {name for name, _ in functions} == {"kvip_" + key for key in options}
    cpp = tmp_path / "options.cpp"
    cpp.write_text(
        "#include <cstdint>\n"
        + "\n".join(f'#define {macro} {int(enabled)}' for _, macro in functions)
        + "\n"
        + "\n".join(f'extern "C" uint32_t {name}() {{ return {macro}; }}' for name, macro in functions)
    )
    library = tmp_path / "options.so"
    subprocess.run(["c++", "-shared", "-fPIC", str(cpp), "-o", str(library)], check=True)
    loaded = ctypes.CDLL(str(library))
    for name, _ in functions:
        getter = getattr(loaded, name)
        getter.argtypes, getter.restype = [], ctypes.c_uint32
        assert getter() == int(enabled), name
