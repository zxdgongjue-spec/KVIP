cmake_minimum_required(VERSION 3.16)
project(kvip_test_two_cold LANGUAGES C CXX)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(RUN_MODE npu)
include(${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
get_filename_component(OP "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../csrc/operators/lightning_indexer_with_fetch" ABSOLUTE)
option(TRACE_SIBLING "Record the non-posting lane at each predecessor check" OFF)
ascendc_library(two_cold_kernel STATIC kernel.cpp)
ascendc_include_directories(two_cold_kernel PRIVATE ${OP}/include ${OP}/op_kernel
  ${OP}/op_kernel/indexer ${KVAF_INCLUDE} ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include)
ascendc_compile_options(two_cold_kernel PRIVATE -DKVIP_STREAMING_LI=1 -DKVIP_OUTPUT_WORKER=1
  -DKVIP_FUSE_QUERY_WORK=1 -O2 -g --cce-auto-sync=off -mllvm -user-reserved-ub-size=223232
  -DV9_USER_UB_BYTES=223232 -DV9_PARTIAL_SCAN_SIMD=1 -DV9_PROBE_EPOCH=0 -DV9_INTERMEDIATE_PROFILING=0
  -DV9_BATCH_POST=2 -DSIM_Q=4 -DSIM_CAPTURE_HITS=1 -DV9_SIMULATION=0 -DROTATION_PUBLISH_TILES=1)
add_library(two_cold_bridge SHARED bridge.cpp)
if(TRACE_SIBLING)
  ascendc_compile_options(two_cold_kernel PRIVATE -DKVIP_TEST_SIBLING_TRACE=1)
  target_compile_definitions(two_cold_bridge PRIVATE KVIP_TEST_SIBLING_TRACE=1)
endif()
target_include_directories(two_cold_bridge PRIVATE ${OP}/include ${KVAF_INCLUDE})
target_link_libraries(two_cold_bridge PRIVATE two_cold_kernel host_intf_pub ascendcl)
