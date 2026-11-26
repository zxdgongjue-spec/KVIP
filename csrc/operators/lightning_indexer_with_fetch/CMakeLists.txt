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
set(KVIP_FUSE_QUERY_WORK ON)
include(${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
set(KVIP_OUTPUT_WORKER ON)
if(KVIP_OUTPUT_WORKER AND NOT KVIP_FUSE_QUERY_WORK)
  message(FATAL_ERROR "KVIP_OUTPUT_WORKER requires KVIP_FUSE_QUERY_WORK")
endif()
set(_base "${CMAKE_CURRENT_SOURCE_DIR}")
ascendc_library(v9_kernel STATIC op_kernel/indexer.cpp op_kernel/indexer_fixed.cpp op_kernel/indexer_large.cpp op_kernel/lifecycle.cpp op_kernel/streaming.cpp)
ascendc_include_directories(v9_kernel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
  ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/indexer ${KVAF_INCLUDE}
  ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include)
ascendc_compile_options(v9_kernel PRIVATE -DKVIP_STREAMING_LI=1 -DKVIP_OUTPUT_WORKER=$<BOOL:${KVIP_OUTPUT_WORKER}> -DKVIP_FUSE_QUERY_WORK=$<BOOL:${KVIP_FUSE_QUERY_WORK}> -O2 -g --cce-auto-sync=off
  -mllvm -user-reserved-ub-size=223232 -DV9_USER_UB_BYTES=223232
  -DV9_PARTIAL_SCAN_SIMD=1 -DV9_PROBE_EPOCH=0 -DV9_INTERMEDIATE_PROFILING=0
  -DV9_BATCH_POST=2 -DSIM_Q=4 -DSIM_CAPTURE_HITS=1 -DV9_SIMULATION=0 -DROTATION_PUBLISH_TILES=1)
add_library(v9_bridge SHARED op_host/launch.cpp)
target_include_directories(v9_bridge PRIVATE ${KVAF_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR}/include ${_base}/include)
target_compile_definitions(v9_bridge PRIVATE KVIP_OUTPUT_WORKER=$<BOOL:${KVIP_OUTPUT_WORKER}>)
target_link_libraries(v9_bridge PRIVATE v9_kernel host_intf_pub ascendcl)

# Independent compile options preserve the retained four-AIV terminal code.
ascendc_library(parallel_terminal_kernel STATIC op_kernel/terminal/kernel.cpp)
ascendc_include_directories(parallel_terminal_kernel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/indexer
  ${KVAF_INCLUDE} ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include)
ascendc_compile_options(parallel_terminal_kernel PRIVATE -DKVIP_PARALLEL_TERMINAL=1
  -O2 -g --cce-auto-sync=off -mllvm -user-reserved-ub-size=223232
  -DV9_USER_UB_BYTES=223232 -DV9_BATCH_POST=2 -DSIM_Q=4 -DSIM_CAPTURE_HITS=1
  -DV9_SIMULATION=0 -DKVIP_STREAMING_LI=1 -DKVIP_OUTPUT_WORKER=1 -DROTATION_PUBLISH_TILES=1)
target_link_libraries(v9_bridge PRIVATE parallel_terminal_kernel)
