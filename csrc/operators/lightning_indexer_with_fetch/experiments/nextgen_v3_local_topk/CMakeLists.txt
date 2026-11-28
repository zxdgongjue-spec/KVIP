cmake_minimum_required(VERSION 3.16)
project(kvip_nextgen_v3_local_topk LANGUAGES C CXX)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(RUN_MODE npu CACHE STRING "NPU build")
set(SOC_VERSION Ascend950DT_9582 CACHE STRING "SoC")
set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/cann-9.2.1 CACHE PATH "CANN")
set(KVAF_INCLUDE "" CACHE PATH "Existing KVAF headers required by reused LI templates")
if(NOT EXISTS "${KVAF_INCLUDE}/kvaf_batch_get_device.h")
  message(FATAL_ERROR "Use the frozen NextGen KVAF device headers")
endif()
get_filename_component(LI_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
include(${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
ascendc_library(nextgen_v3_local_kernel STATIC b16_local_topk.cpp)
ascendc_include_directories(nextgen_v3_local_kernel PRIVATE
  ${LI_ROOT}/include ${LI_ROOT}/op_kernel ${LI_ROOT}/op_kernel/indexer
  ${KVAF_INCLUDE} ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include)
ascendc_compile_options(nextgen_v3_local_kernel PRIVATE -O2 -g --cce-auto-sync=off
  -mllvm -user-reserved-ub-size=223232 -DV9_USER_UB_BYTES=223232
  -DV9_RESOLVER_COMPARE=0 -DV9_PARTIAL_SCAN_SIMD=1 -DV9_PROBE_EPOCH=0
  -DV9_INTERMEDIATE_PROFILING=0 -DV9_BATCH_POST=2 -DSIM_Q=4
  -DSIM_CAPTURE_HITS=1 -DV9_SIMULATION=0 -DROTATION_PUBLISH_TILES=1
  -DV9_S2_SCORE_PROBE=0 -DV9_S2_SCORE_CORES=1)
add_library(nextgen_v3_local_bridge SHARED launch.cpp)
target_link_libraries(nextgen_v3_local_bridge PRIVATE nextgen_v3_local_kernel host_intf_pub ascendcl)
