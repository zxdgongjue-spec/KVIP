# SPDX-License-Identifier: Apache-2.0
"""Test actual SDK wrapper generation and its batch-schedule hook.

The host preprocessor models attributes and the generated inline origin.
This checks source transformation, not CANN device compilation or hardware
scheduling. The external SDK is never copied into this repository.
"""

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest


ENTRY = "v9_streaming_resolve_update"
WRAPPER = f"auto_gen_{ENTRY}_kernel"
OTHER_ENTRIES = ("v9_counter_calibration", "v9_query_fetch_only", "v9_native_fetch_only")


def _extractor(repo):
    override = os.environ.get("ASCENDC_HOST_STUB_EXTRACTOR")
    if override:
        path = Path(override)
        assert path.is_file(), f"ASCENDC_HOST_STUB_EXTRACTOR is missing: {path}"
        return path
    candidates = [repo.parent / "asc-devkit/cmake/asc/legacy_modules/util/extract_host_stub.py"]
    cann = os.environ.get("ASCEND_CANN_PACKAGE_PATH")
    if cann:
        for prefix in ("compiler/tikcpp", "aarch64-linux/tikcpp", "tikcpp"):
            candidates.append(Path(cann) / prefix / "ascendc_kernel_cmake/legacy_modules/util/extract_host_stub.py")
    for path in candidates:
        if path.is_file():
            return path
    pytest.skip("Set ASCENDC_HOST_STUB_EXTRACTOR to the SDK extract_host_stub.py")


def _preprocess(compiler, source):
    return subprocess.run([compiler, "-E", "-P", "-x", "c++", str(source)],
                          capture_output=True, text=True, check=True).stdout


@pytest.fixture(params=[False, True], ids=["keyword", "attribute"])
def generated(tmp_path, request):
    repo = Path(__file__).resolve().parents[4]
    extractor = _extractor(repo)
    compiler = shutil.which("clang++") or shutil.which("g++")
    assert compiler, "A C++ preprocessor is required"
    source = (repo / "csrc/operators/lightning_indexer_with_fetch/op_kernel/streaming.cpp").read_text()
    declarations = set(re.findall(
        r'^extern "C"[^\n]*\bvoid v9_streaming_resolve_update\(GM_ADDR arg\) \{', source, re.MULTILINE,
    ))
    assert declarations and all("__schedmode__" not in line for line in declarations)
    assert len(declarations) == 1
    declaration = declarations.pop()
    others = []
    for name in OTHER_ENTRIES:
        found = re.findall(r'^extern "C"[^\n]*\bvoid ' + name + r'\([^\n]*\) \{', source, re.MULTILINE)
        assert len(found) == 1
        others.append(found[0] + "}\n")
    # The SDK defines __global__ as inline before including this file. Preserve
    # that real transformation, rather than accidentally undoing it in a stub.
    macros = """\
#ifndef __global__
#define __global__ __attribute__((cce_kernel))
#endif
#define __aicore__ [aicore]
#define __gm__ __attribute__((cce_global))
#define GM_ADDR __gm__ uint8_t*
"""
    if request.param:
        macros += '#define __schedmode__(mode) __attribute__((cce_sched_mode(mode)))\n'
    paths = {}
    for name, text in (("current", declaration),
                       ("previous", declaration.replace("__global__", "__schedmode__(1) __global__", 1))):
        directory = tmp_path / name
        directory.mkdir()
        raw = directory / "streaming.cpp"
        raw.write_text(macros + text + "}\n" + "".join(others))
        preprocessed = directory / "streaming.cpp.o"
        preprocessed.write_text(_preprocess(compiler, raw))
        # Isolate generic SDK module names (constants/channel/...) from pytest.
        script = r'''
import sys
sys.path.insert(0, sys.argv[1])
import extract_host_stub as sdk
sdk.RUN_MODE = "npu"
sdk.IS_C310_MODE = True
groups = list(sdk.parse_func_signature_by_filepaths([sys.argv[2]], "c310"))
assert len(groups) == 1
group = groups[0]
assert {s.func_name for s in group.func_signs} == {
    "v9_streaming_resolve_update", "v9_counter_calibration", "v9_query_fetch_only", "v9_native_fetch_only"}
wrapped = sdk.FuncSignGroupWithModeBase(group.filepath, group.func_signs, group.dump_info,
                                      sdk.CodeMode.KERNEL_TYPE_MIX_AIV_1_0, 0)
sdk.save_device_kernel_function([wrapped], {sys.argv[2]: sys.argv[3]}, sys.argv[4], True)
'''
        subprocess.run([sys.executable, "-c", script, str(extractor.parent), str(preprocessed),
                        str(raw), str(directory)], capture_output=True, text=True, check=True)
        paths[name] = directory / "auto_gen_streaming.cpp"
        assert paths[name].is_file()
    return repo, compiler, paths


def _signature(text, name):
    found = re.findall(r'extern "C"[^{};]*\b' + re.escape(name) + r'\([^{};]*\)\s*\{', text)
    assert len(found) == 1, (name, found)
    return found[0]


def _scheduled(signature):
    return bool(re.search(r"(?:__schedmode__|cce_sched_mode)\s*\(\s*1\s*\)", signature))


def _patch(repo, path):
    cmake = shutil.which("cmake")
    assert cmake, "CMake is required to validate the generated-source hook"
    helper = repo / "csrc/operators/lightning_indexer_with_fetch/cmake/set_tail_schedule.cmake"
    assert helper.is_file()
    return subprocess.run([cmake, f"-DGENERATED_SOURCE={path}", "-P", str(helper)],
                          capture_output=True, text=True)


def test_actual_sdk_wrapper_reproduces_origin_attribute_leak(generated):
    _, compiler, paths = generated
    before = _preprocess(compiler, paths["previous"])
    origin, wrapper = _signature(before, ENTRY + "_origin"), _signature(before, WRAPPER)
    assert _scheduled(origin) and "cce_kernel" not in origin
    assert not _scheduled(wrapper) and "cce_kernel" in wrapper


def test_generated_kernel_alone_gets_batch_mode_and_hook_is_idempotent(generated):
    repo, compiler, paths = generated
    wrapper = paths["current"]
    outcome = _patch(repo, wrapper)
    assert outcome.returncode == 0, outcome.stdout + outcome.stderr
    after = _preprocess(compiler, wrapper)
    origin = _signature(after, ENTRY + "_origin")
    assert not _scheduled(origin) and "cce_kernel" not in origin
    actual = _signature(after, WRAPPER)
    assert _scheduled(actual) and "cce_kernel" in actual
    for name in OTHER_ENTRIES:
        assert not _scheduled(_signature(after, f"auto_gen_{name}_kernel"))
    before = wrapper.read_bytes()
    os.utime(wrapper, ns=(1_600_000_000_000_000_000, 1_600_000_000_000_000_000))
    stamp = wrapper.stat().st_mtime_ns
    outcome = _patch(repo, wrapper)
    assert outcome.returncode == 0, outcome.stdout + outcome.stderr
    assert wrapper.read_bytes() == before and wrapper.stat().st_mtime_ns == stamp


@pytest.mark.parametrize("fault", ["missing", "duplicate", "wrong-mode"])
def test_generated_source_hook_rejects_missing_or_ambiguous_kernel(generated, fault):
    repo, _, paths = generated
    wrapper = paths["current"]
    original = wrapper.read_text()
    if fault == "missing":
        invalid = original.replace(WRAPPER, "auto_gen_another_kernel")
    elif fault == "duplicate":
        invalid = original + original
    else:
        invalid = original.replace('extern "C" __global__ [aicore] void ' + WRAPPER,
                                   'extern "C" __schedmode__(2) __global__ [aicore] void ' + WRAPPER)
        assert invalid != original
    wrapper.write_text(invalid)
    outcome = _patch(repo, wrapper)
    assert outcome.returncode != 0
    assert wrapper.read_text() == invalid
