# SPDX-License-Identifier: Apache-2.0
"""Exercise the real schedule-fix build step before parallel device consumers."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_schedule_step_runs_after_every_regeneration_before_parallel_compile(tmp_path):
    cmake = shutil.which("cmake")
    if not cmake:
        pytest.skip("CMake is required for the build-order integration test")
    repo = Path(__file__).resolve().parents[4]
    operator = repo / "csrc/operators/lightning_indexer_with_fetch"
    source = (operator / "CMakeLists.txt").read_text()
    start = source.index("ExternalProject_Add_Step(v9_kernel_preprocess tail_batch_schedule")
    depth, finish = 0, start
    for finish in range(source.index("(", start), len(source)):
        depth += (source[finish] == "(") - (source[finish] == ")")
        if not depth:
            break
    step = source[start:finish + 1].replace(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/set_tail_schedule.cmake",
        str(operator / "cmake/set_tail_schedule.cmake"),
    )
    (tmp_path / "generate.cmake").write_text(r'''
file(MAKE_DIRECTORY "${ROOT}/auto_gen/v9_kernel")
file(WRITE "${ROOT}/auto_gen/v9_kernel/auto_gen_streaming.cpp" [=[
extern "C" __aicore__ inline void v9_streaming_resolve_update_origin(GM_ADDR arg) {}
extern "C" __global__ [aicore] void auto_gen_v9_streaming_resolve_update_kernel(
    GM_ADDR arg, GM_ADDR overflow_status) {
    v9_streaming_resolve_update_origin(arg);
}
]=])
file(APPEND "${ROOT}/generations.txt" "generated\n")
''')
    (tmp_path / "check.cmake").write_text(r'''
file(READ "${ROOT}/auto_gen/v9_kernel/auto_gen_streaming.cpp" text)
if(NOT text MATCHES "__schedmode__\\(1\\)[^;{}]*auto_gen_v9_streaming_resolve_update_kernel")
  message(FATAL_ERROR "Device compile observed an unpatched wrapper")
endif()
if(text MATCHES "__schedmode__[^;{}]*v9_streaming_resolve_update_origin")
  message(FATAL_ERROR "The non-kernel origin was annotated")
endif()
file(APPEND "${ROOT}/${SIDE}.txt" "checked\n")
''')
    (tmp_path / "CMakeLists.txt").write_text(r'''
cmake_minimum_required(VERSION 3.16)
project(schedule_order NONE)
include(ExternalProject)
ExternalProject_Add(v9_kernel_preprocess SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  DOWNLOAD_COMMAND "" CONFIGURE_COMMAND "" INSTALL_COMMAND "" BUILD_ALWAYS TRUE
  BUILD_COMMAND "${CMAKE_COMMAND}" -DROOT=${CMAKE_BINARY_DIR} -P "${CMAKE_CURRENT_SOURCE_DIR}/generate.cmake")
''' + step + r'''
foreach(side aic aiv)
  add_custom_target(${side} ALL
    COMMAND "${CMAKE_COMMAND}" -DROOT=${CMAKE_BINARY_DIR} -DSIDE=${side}
      -P "${CMAKE_CURRENT_SOURCE_DIR}/check.cmake"
    DEPENDS v9_kernel_preprocess)
endforeach()
''')
    build = tmp_path / "build"
    subprocess.run([cmake, "-S", str(tmp_path), "-B", str(build)], check=True, capture_output=True, text=True)
    for _ in range(2):
        subprocess.run([cmake, "--build", str(build), "--parallel", "4"], check=True, capture_output=True, text=True)
    assert (build / "generations.txt").read_text().splitlines() == ["generated"] * 2
    for side in ("aic", "aiv"):
        assert (build / f"{side}.txt").read_text().splitlines() == ["checked"] * 2
