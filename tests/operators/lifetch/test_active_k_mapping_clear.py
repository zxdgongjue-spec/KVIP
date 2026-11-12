# SPDX-License-Identifier: Apache-2.0
"""P0-A1: host-executed regression of the exact device mapping clear bound."""

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
HEADER_DIR = ROOT / "csrc/operators/lightning_indexer_with_fetch/include"


def test_mapping_clear_bound_host_cpp(tmp_path):
    compiler = shutil.which("c++") or shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable for host-only mapping guard test")
    source = tmp_path / "mapping_bound_test.cpp"
    source.write_text(r"""
#include <cassert>
#include <cstdint>
#include "mapping_clear_bounds.h"

using rotating::ActiveKMappingClearTokens;
int main() {
    constexpr uint32_t H = 131072, R = 8192;
    // Normal mixed-Q per-request K, and the 128-token partial tail.
    assert(ActiveKMappingClearTokens(H,R,true,true,32768,true,32768,false,0,0) == 32768);
    assert(ActiveKMappingClearTokens(H,R,true,true,32769,true,32769,false,0,0) == 32896);
    assert(ActiveKMappingClearTokens(H,R,true,true,8192,true,8192,false,0,0) == R);
    assert(ActiveKMappingClearTokens(H,R,true,true,0,true,0,false,0,0) == R);
    assert(ActiveKMappingClearTokens(H,R,true,true,H,true,H,false,0,0) == H);
    // Missing/inconsistent ABI metadata must disable partial clearing.
    assert(ActiveKMappingClearTokens(H,R,false,true,32768,true,32768,false,0,0) == H);
    assert(ActiveKMappingClearTokens(H,R,true,false,32768,true,32768,false,0,0) == H);
    assert(ActiveKMappingClearTokens(H,R,true,true,32768,false,32768,false,0,0) == H);
    assert(ActiveKMappingClearTokens(H,R,true,true,32768,true,32769,false,0,0) == H);
    assert(ActiveKMappingClearTokens(H,R,true,true,H+1,true,H+1,false,0,0) == H);
    // ABC tail mapping writes must remain in the freshly invalidated span.
    assert(ActiveKMappingClearTokens(H,R,true,true,32768,true,32768,true,32700,33000) == 33024);
    assert(ActiveKMappingClearTokens(H,R,true,true,32768,true,32768,true,50000,49999) == H);
    assert(ActiveKMappingClearTokens(H,R,true,true,32768,true,32768,true,H-128,H+1) == H);
    // Long -> short -> long and row reuse must be independently bounded.
    assert(ActiveKMappingClearTokens(H,R,true,true,98304,true,98304,false,0,0) == 98304);
    assert(ActiveKMappingClearTokens(H,R,true,true,16384,true,16384,false,0,0) == 16384);
    assert(ActiveKMappingClearTokens(H,R,true,true,98304,true,98304,false,0,0) == 98304);
}
""")
    binary = tmp_path / "mapping_bound_test"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
         "-I", str(HEADER_DIR), str(source), "-o", str(binary)],
        check=True,
    )
    subprocess.run([str(binary)], check=True)


def test_fixture_can_explicitly_build_optimized_kernel():
    # A source-level integration guard: the URMA runner must propagate the
    # CLI flag into the same CMake option wired by the service-side builder.
    test_entry = (ROOT / "tests/operators/lifetch/test_lifetch.py").read_text()
    runner = (ROOT / "tests/operators/lifetch/support/lifetch.py").read_text()
    cmake = (ROOT / "csrc/operators/lightning_indexer_with_fetch/CMakeLists.txt").read_text()
    runtime = (ROOT / "python/kvip/runtime/build.py").read_text()
    assert test_entry.count('add_argument("--active-k-map-clear"') == 2
    assert 'common.append("--active-k-map-clear")' in test_entry
    assert 'V9_ACTIVE_K_MAP_CLEAR=int(getattr(args, "active_k_map_clear", False))' in runner
    assert "-DV9_ACTIVE_K_MAP_CLEAR=${V9_ACTIVE_K_MAP_CLEAR}" in cmake
    assert 'V9_ACTIVE_K_MAP_CLEAR=active_k_map_clear' in runtime


def test_resolver_checks_actual_per_query_bounds_when_enabled():
    src = (ROOT / "csrc/operators/common/lifetch/resident_selection.h").read_text()
    assert "QueryEnd(rt_history,ctx,batch*4U+pos/kTop)" in src
    assert "if(cfg[1]) return false;" in src
    assert "#if V9_ACTIVE_K_MAP_CLEAR" in src
