# SPDX-License-Identifier: Apache-2.0
"""Check kernel declaration ordering against the installed SDK's parser.

The host preprocessor supplies the SDK's attribute spellings. This exercises
host-stub extraction only, not CANN device compilation or scheduler behavior.
The SDK is an optional external dependency and is never copied into this repo.
"""

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest


def _extractor(repo):
    override = os.environ.get("ASCENDC_HOST_STUB_EXTRACTOR")
    if override:
        path = Path(override)
        assert path.is_file(), f"ASCENDC_HOST_STUB_EXTRACTOR is missing: {path}"
        return path
    candidates = [
        repo.parent / "asc-devkit/cmake/asc/legacy_modules/util/extract_host_stub.py",
    ]
    cann = os.environ.get("ASCEND_CANN_PACKAGE_PATH")
    if cann:
        for prefix in ("compiler/tikcpp", "aarch64-linux/tikcpp", "tikcpp"):
            candidates.append(Path(cann) / prefix / "ascendc_kernel_cmake/legacy_modules/util/extract_host_stub.py")
    for path in candidates:
        if path.is_file():
            return path
    pytest.skip("Set ASCENDC_HOST_STUB_EXTRACTOR to the SDK extract_host_stub.py")


@pytest.mark.parametrize("schedmode_attribute", [False, True], ids=["keyword", "attribute"])
def test_batch_kernel_signature_preserves_sdk_host_stub_extraction(tmp_path, schedmode_attribute):
    repo = Path(__file__).resolve().parents[4]
    extractor = _extractor(repo)
    compiler = shutil.which("clang++") or shutil.which("g++")
    assert compiler, "A C++ preprocessor is required"
    source = (repo / "csrc/operators/lightning_indexer_with_fetch/op_kernel/streaming.cpp").read_text()
    declarations = re.findall(
        r'^extern "C"[^\n]*\bvoid v9_streaming_resolve_update\(GM_ADDR arg\) \{',
        source,
        re.MULTILINE,
    )
    current = [line for line in declarations if "__schedmode__(1)" in line]
    assert len(current) == 1
    current = current[0]
    assert '__schedmode__(1) __global__ __aicore__ void' in current
    previous = current.replace(
        '__schedmode__(1) __global__ __aicore__',
        '__global__ __aicore__ __schedmode__(1)',
    )
    macros = """\
#define __global__ __attribute__((cce_kernel))
#define __aicore__ [aicore]
#define __gm__ __attribute__((cce_global))
#define GM_ADDR __gm__ uint8_t*
"""
    if schedmode_attribute:
        # Also exercise an equivalent expanded attribute, not only the keyword.
        macros += '#define __schedmode__(mode) __attribute__((cce_sched_mode(mode)))\n'
    paths = []
    for name, declaration in (("previous", previous), ("current", current)):
        path = tmp_path / f"{name}.cpp.o"
        result = subprocess.run(
            [compiler, "-E", "-P", "-x", "c++", "-"],
            input=macros + declaration + "}\n",
            capture_output=True,
            text=True,
            check=True,
        )
        path.write_text(result.stdout)
        paths.append(path)

    # Isolate generic SDK module names (constants, channel, ...) from pytest.
    check = r'''
import sys
sys.path.insert(0, sys.argv[1])
import extract_host_stub as parser
try:
    list(parser.parse_func_signature_by_filepaths([sys.argv[2]], "c310"))
except parser.FuncNameNotFound:
    pass
else:
    raise AssertionError("The old attribute order did not reproduce the SDK parser error")
groups = list(parser.parse_func_signature_by_filepaths([sys.argv[3]], "c310"))
assert len(groups) == 1
assert len(groups[0].func_signs) == 1
signature = groups[0].func_signs[0]
assert signature.func_name == "v9_streaming_resolve_update"
assert signature.return_type == "void"
assert len(signature.func_params) == 1
parameter = signature.func_params[0]
assert parameter.cce_global
assert parser.get_func_param_type(parameter) == "uint8_t*"
assert parser.get_func_param_name(parameter) == "arg"
print("HOST_STUB_SIGNATURE_PASS")
'''
    result = subprocess.run(
        [sys.executable, "-c", check, str(extractor.parent), *map(str, paths)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "HOST_STUB_SIGNATURE_PASS" in result.stdout
