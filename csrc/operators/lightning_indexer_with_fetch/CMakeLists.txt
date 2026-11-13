cmake_minimum_required(VERSION 3.16)
project(kvip_lifetch LANGUAGES C CXX)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(RUN_MODE npu CACHE STRING "NPU build")
set(SOC_VERSION Ascend950PR_9599 CACHE STRING "Target SoC, not runtime device ordinal")
set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/cann CACHE PATH "CANN installation")
set(KVAF_INCLUDE "" CACHE PATH "KVAF device interface headers")
if(NOT EXISTS "${KVAF_INCLUDE}/kvaf_batch_get_device.h")
  message(FATAL_ERROR "Set KVAF_INCLUDE to the patched KVAF aiv_batch_get/include directory")
endif()
set(RESOLVER_COMPARE 0 CACHE STRING "Include resolver diagnostic entry points")
# Test-only Event timing/candidate telemetry. Keep the production default off.
set(V9_INTERMEDIATE_PROFILING 0 CACHE STRING "Profile V9 Rotation Epoch events (0 or 1)")
if(NOT V9_INTERMEDIATE_PROFILING MATCHES "^[01]$")
  message(FATAL_ERROR "V9_INTERMEDIATE_PROFILING must be 0 or 1")
endif()
# Opt-in varlen optimization. The logical mapping buffer retains full History capacity.
set(V9_ACTIVE_K_MAP_CLEAR 0 CACHE STRING "Clear only the active per-request K prefix in v9_init (0 or 1)")
if(NOT V9_ACTIVE_K_MAP_CLEAR MATCHES "^[01]$")
  message(FATAL_ERROR "V9_ACTIVE_K_MAP_CLEAR must be 0 or 1")
endif()
# Test-only TopK attribution: 0 keeps the already-validated Q2 16K Trunk.
# 1 sets Q2 to the existing Q4 8K Trunk without changing Q-core mapping.
set(V9_Q2_TRUNK8K_CONTROL 0 CACHE STRING "LI-only P0-C3 Q2 TopK Trunk=8K (0 or 1)")
if(NOT V9_Q2_TRUNK8K_CONTROL MATCHES "^[01]$")
  message(FATAL_ERROR "V9_Q2_TRUNK8K_CONTROL must be 0 or 1")
endif()
# P0-C5a is opt-in: build a separate Score-only diagnostic kernel,
# while keeping the original v9_li_kernel implementation intact.
set(V9_S2_SCORE_PROBE 0 CACHE STRING "Build isolated S2 Score-only diagnostic (0 or 1)")
set(V9_S2_SCORE_CORES 1 CACHE STRING "Score-only diagnostic groups: 1, 2, or 4")
if(NOT V9_S2_SCORE_PROBE MATCHES "^[01]$" OR
   NOT V9_S2_SCORE_CORES MATCHES "^(1|2|4)$")
  message(FATAL_ERROR "V9_S2_SCORE_PROBE=0/1 and V9_S2_SCORE_CORES=1/2/4 required")
endif()
include(${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
set(V9_KERNEL_SOURCES op_kernel/indexer.cpp op_kernel/lifecycle.cpp)
if(V9_S2_SCORE_PROBE)
  list(APPEND V9_KERNEL_SOURCES op_kernel/indexer_s2_score.cpp)
endif()
ascendc_library(v9_kernel STATIC ${V9_KERNEL_SOURCES})
ascendc_include_directories(v9_kernel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
  ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/indexer ${KVAF_INCLUDE}
  ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include)
ascendc_compile_options(v9_kernel PRIVATE -DV9_RESOLVER_COMPARE=${RESOLVER_COMPARE} -O2 -g --cce-auto-sync=off
  -mllvm -user-reserved-ub-size=223232 -DV9_USER_UB_BYTES=223232
  -DV9_PARTIAL_SCAN_SIMD=1 -DV9_PROBE_EPOCH=0 -DV9_INTERMEDIATE_PROFILING=${V9_INTERMEDIATE_PROFILING}
  -DV9_BATCH_POST=2 -DSIM_Q=4 -DSIM_CAPTURE_HITS=1 -DV9_SIMULATION=0 -DROTATION_PUBLISH_TILES=1
  -DV9_ACTIVE_K_MAP_CLEAR=${V9_ACTIVE_K_MAP_CLEAR} -DV9_Q2_TRUNK8K_CONTROL=${V9_Q2_TRUNK8K_CONTROL}
  -DV9_S2_SCORE_PROBE=${V9_S2_SCORE_PROBE} -DV9_S2_SCORE_CORES=${V9_S2_SCORE_CORES})
add_library(v9_bridge SHARED op_host/launch.cpp)
target_include_directories(v9_bridge PRIVATE ${KVAF_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(v9_bridge PRIVATE V9_RESOLVER_COMPARE=${RESOLVER_COMPARE}
  V9_S2_SCORE_PROBE=${V9_S2_SCORE_PROBE} V9_S2_SCORE_CORES=${V9_S2_SCORE_CORES})
target_link_libraries(v9_bridge PRIVATE v9_kernel host_intf_pub ascendcl)
# Independent AIV target: does not change the LI/Fetch compilation options.
ascendc_library(kvop_cache_update_kernel STATIC op_kernel/index_cache_update.cpp)
ascendc_include_directories(kvop_cache_update_kernel PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include)
ascendc_compile_options(kvop_cache_update_kernel PRIVATE -O2
  -mllvm -user-reserved-ub-size=223232)
add_library(kvop_cache_update SHARED op_host/index_cache_update.cpp)
target_link_libraries(kvop_cache_update PRIVATE kvop_cache_update_kernel host_intf_pub ascendcl)

target_link_libraries(v9_bridge PRIVATE kvop_cache_update)

# CANN may suppress the automatic build RPATH. The two libraries are deployed
# together; resolve the update dependency relative to the bridge, not the cwd.
set_target_properties(v9_bridge PROPERTIES
  SKIP_BUILD_RPATH FALSE
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "$ORIGIN")

if(RESOLVER_COMPARE)
  ascendc_include_directories(v9_kernel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/operators/lifetch)
endif()
