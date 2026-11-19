cmake_minimum_required(VERSION 3.16)
project(lifetch_predicate_probe LANGUAGES C CXX)
set(CMAKE_BUILD_TYPE Release)
set(RUN_MODE sim CACHE STRING "CANN Model or npu")
set(SOC_VERSION Ascend950PR_9599 CACHE STRING "SoC")
set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/cann CACHE PATH "CANN")
set(KVAF_INCLUDE "" CACHE PATH "Patched KVAF device headers")
include(${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(OP "${REPO_ROOT}/csrc/operators/lightning_indexer_with_fetch")
ascendc_library(predicate_kernel STATIC ../predicate_probe.cpp)
ascendc_include_directories(predicate_kernel PRIVATE ${OP}/include ${OP}/op_kernel ${OP}/op_kernel/indexer
  ${KVAF_INCLUDE} ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include)
ascendc_compile_options(predicate_kernel PRIVATE -DKVIP_STREAMING_LI=1 -DKVIP_OUTPUT_WORKER=1 -DKVIP_FUSE_QUERY_WORK=1
  -O2 -g --cce-auto-sync=off -mllvm -user-reserved-ub-size=223232 -DV9_USER_UB_BYTES=223232
  -DV9_PARTIAL_SCAN_SIMD=1 -DV9_PROBE_EPOCH=0 -DV9_INTERMEDIATE_PROFILING=0
  -DV9_BATCH_POST=2 -DSIM_Q=4 -DSIM_CAPTURE_HITS=1 -DV9_SIMULATION=0 -DROTATION_PUBLISH_TILES=1)
add_executable(predicate_smoke ../predicate_smoke.cpp)
target_include_directories(predicate_smoke PRIVATE ${OP}/include ${KVAF_INCLUDE})
target_link_libraries(predicate_smoke PRIVATE predicate_kernel host_intf_pub ascendcl)
if(RUN_MODE STREQUAL "sim")
  target_link_libraries(predicate_smoke PRIVATE "-Wl,-rpath-link,${ASCEND_CANN_PACKAGE_PATH}/tools/simulator/${SOC_VERSION}/lib")
endif()
