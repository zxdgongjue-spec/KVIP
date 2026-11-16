cmake_minimum_required(VERSION 3.16)
project(kvip_score_routes LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(RUN_MODE sim CACHE STRING "sim or npu")
set(SOC_VERSION Ascend950PR_9599 CACHE STRING "Actual target")
set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/cann-9.1.0 CACHE PATH "Matching CANN")
set(SCORE_SOURCE "" CACHE PATH "prepare_score_source.py output")
set(SCORE_VERSION current CACHE STRING "current or historical")
set(KVAF_INCLUDE "" CACHE PATH "Pinned patched KVAF headers")
if(NOT EXISTS "${SCORE_SOURCE}/score-source.json" OR NOT EXISTS "${KVAF_INCLUDE}/kvaf_batch_get_device.h")
  message(FATAL_ERROR "Prepare versioned score source and provide pinned KVAF headers")
endif()
file(READ "${SCORE_SOURCE}/score-source.json" _manifest)
string(JSON _version GET "${_manifest}" version)
if(NOT _version STREQUAL SCORE_VERSION)
  message(FATAL_ERROR "SCORE_VERSION disagrees with prepared source")
endif()
if(SCORE_VERSION STREQUAL current)
  set(STREAMING 1)
elseif(SCORE_VERSION STREQUAL historical)
  set(STREAMING 0)
else()
  message(FATAL_ERROR "Unsupported SCORE_VERSION")
endif()
include(${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
set(OP ${SCORE_SOURCE}/csrc/operators/lightning_indexer_with_fetch)
ascendc_library(score_kernel STATIC ../score_kernel.cpp)
ascendc_include_directories(score_kernel PRIVATE ${OP}/include ${OP}/op_kernel ${OP}/op_kernel/indexer
  ${KVAF_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR}/../historical/full
  ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include)
ascendc_compile_options(score_kernel PRIVATE -O2 -g --cce-auto-sync=off -mllvm -user-reserved-ub-size=223232
  -DV9_USER_UB_BYTES=223232 -DKVIP_STREAMING_LI=${STREAMING} -DSIM_Q=4 -DSIM_CAPTURE_HITS=0
  -DKVIP_OUTPUT_WORKER=1 -DKVIP_FUSE_QUERY_WORK=1
  -DV9_SIMULATION=0 -DROTATION_PUBLISH_TILES=1 -DKVIP_FUSION3=0 -DKVIP_FUSION3_TEST_TRANSPORT=1
  -DKVIP_OPT_LI_REGISTER_PREDICATE=0 -DV9_PARTIAL_SCAN_SIMD=1 -DV9_PROBE_EPOCH=0
  -DV9_INTERMEDIATE_PROFILING=0 -DV9_BATCH_POST=2)
add_executable(score_routes ../score_host.cpp)
target_include_directories(score_routes PRIVATE ${SCORE_SOURCE}/csrc/operators/common/lifetch)
# Historical source predates score_schedule.h; the new planner is explicit input.
target_include_directories(score_routes PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../csrc/operators/common/lifetch)
target_compile_definitions(score_routes PRIVATE SCORE_VERSION="${SCORE_VERSION}" SCORE_RUN_MODE="${RUN_MODE}")
target_link_libraries(score_routes PRIVATE score_kernel host_intf_pub ascendcl)
if(RUN_MODE STREQUAL sim)
  # Match the repository's existing CA Model driver link contract.
  target_link_options(score_routes PRIVATE -Wl,--allow-shlib-undefined)
endif()
