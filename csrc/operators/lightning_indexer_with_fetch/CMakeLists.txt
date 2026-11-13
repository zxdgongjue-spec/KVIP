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
option(KVIP_RECENT_PIPELINE "Keep query state across recent/resident seed ranges" ON)
option(KVIP_RECENT_UNIFIED_TOPK "Select the resident/recent seed TopK in one stream" ON)
option(KVIP_RECENT_SKIP "Reuse recent scores at their original Fetch windows" OFF)
option(KVIP_RECENT_THRESHOLD_ONLY "Compute the exact seed threshold without materializing unused TopK lists" OFF)
option(KVIP_RECENT_SEED_UB "Keep early scores in the existing TopK scratch until threshold selection" OFF)
option(KVIP_RECENT_PROFILE "Compile optional early/final TopK diagnostic markers" ON)
option(KVIP_RECENT_LOCAL_VICTIMS "Classify hot-slot protection from the resident scores already in UB" OFF)
option(KVIP_RECENT_REGISTER_KTH "Keep the exact early-Kth histogram selection in vector registers" OFF)
if(KVIP_RECENT_SEED_UB AND NOT KVIP_RECENT_THRESHOLD_ONLY)
  message(FATAL_ERROR "KVIP_RECENT_SEED_UB requires KVIP_RECENT_THRESHOLD_ONLY")
endif()
if(KVIP_RECENT_LOCAL_VICTIMS AND NOT KVIP_RECENT_THRESHOLD_ONLY)
  message(FATAL_ERROR "KVIP_RECENT_LOCAL_VICTIMS requires KVIP_RECENT_THRESHOLD_ONLY")
endif()
if(KVIP_RECENT_REGISTER_KTH AND NOT KVIP_RECENT_THRESHOLD_ONLY)
  message(FATAL_ERROR "KVIP_RECENT_REGISTER_KTH requires KVIP_RECENT_THRESHOLD_ONLY")
endif()
include(${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
set(KVIP_OUTPUT_WORKER ON)
if(KVIP_OUTPUT_WORKER AND NOT KVIP_FUSE_QUERY_WORK)
  message(FATAL_ERROR "KVIP_OUTPUT_WORKER requires KVIP_FUSE_QUERY_WORK")
endif()
set(_base "${CMAKE_CURRENT_SOURCE_DIR}")
ascendc_library(v9_kernel STATIC op_kernel/indexer.cpp op_kernel/lifecycle.cpp op_kernel/streaming.cpp)
ascendc_include_directories(v9_kernel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
  ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/indexer ${KVAF_INCLUDE}
  ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include)
ascendc_compile_options(v9_kernel PRIVATE -DKVIP_STREAMING_LI=1 -DKVIP_OUTPUT_WORKER=$<BOOL:${KVIP_OUTPUT_WORKER}> -DKVIP_FUSE_QUERY_WORK=$<BOOL:${KVIP_FUSE_QUERY_WORK}> -O2 -g --cce-auto-sync=off
  -DKVIP_RECENT_PIPELINE=$<BOOL:${KVIP_RECENT_PIPELINE}> -DKVIP_RECENT_UNIFIED_TOPK=$<BOOL:${KVIP_RECENT_UNIFIED_TOPK}>
  -DKVIP_RECENT_SKIP=$<BOOL:${KVIP_RECENT_SKIP}>
  -DKVIP_RECENT_THRESHOLD_ONLY=$<BOOL:${KVIP_RECENT_THRESHOLD_ONLY}>
  -DKVIP_RECENT_SEED_UB=$<BOOL:${KVIP_RECENT_SEED_UB}> -DKVIP_RECENT_PROFILE=$<BOOL:${KVIP_RECENT_PROFILE}>
  -DKVIP_RECENT_LOCAL_VICTIMS=$<BOOL:${KVIP_RECENT_LOCAL_VICTIMS}>
  -DKVIP_RECENT_REGISTER_KTH=$<BOOL:${KVIP_RECENT_REGISTER_KTH}>
  -mllvm -user-reserved-ub-size=223232 -DV9_USER_UB_BYTES=223232
  -DV9_PARTIAL_SCAN_SIMD=1 -DV9_PROBE_EPOCH=0 -DV9_INTERMEDIATE_PROFILING=0
  -DV9_BATCH_POST=2 -DSIM_Q=4 -DSIM_CAPTURE_HITS=1 -DV9_SIMULATION=0 -DROTATION_PUBLISH_TILES=1)
add_library(v9_bridge SHARED op_host/launch.cpp)
target_include_directories(v9_bridge PRIVATE ${KVAF_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR}/include ${_base}/include)
target_compile_definitions(v9_bridge PRIVATE KVIP_OUTPUT_WORKER=$<BOOL:${KVIP_OUTPUT_WORKER}>
  KVIP_RECENT_PIPELINE=$<BOOL:${KVIP_RECENT_PIPELINE}> KVIP_RECENT_UNIFIED_TOPK=$<BOOL:${KVIP_RECENT_UNIFIED_TOPK}>
  KVIP_RECENT_SKIP=$<BOOL:${KVIP_RECENT_SKIP}>
  KVIP_RECENT_THRESHOLD_ONLY=$<BOOL:${KVIP_RECENT_THRESHOLD_ONLY}>
  KVIP_RECENT_SEED_UB=$<BOOL:${KVIP_RECENT_SEED_UB}> KVIP_RECENT_PROFILE=$<BOOL:${KVIP_RECENT_PROFILE}>
  KVIP_RECENT_LOCAL_VICTIMS=$<BOOL:${KVIP_RECENT_LOCAL_VICTIMS}>
  KVIP_RECENT_REGISTER_KTH=$<BOOL:${KVIP_RECENT_REGISTER_KTH}>)
target_link_libraries(v9_bridge PRIVATE v9_kernel host_intf_pub ascendcl)
