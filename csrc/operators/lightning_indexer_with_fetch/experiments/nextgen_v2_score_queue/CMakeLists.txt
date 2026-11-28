cmake_minimum_required(VERSION 3.16)
project(kvip_nextgen_v2_score_queue LANGUAGES C CXX)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(RUN_MODE npu CACHE STRING "NPU build")
set(SOC_VERSION Ascend950DT_9582 CACHE STRING "SoC")
set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/cann-9.2.1 CACHE PATH "CANN")
set(KVAF_INCLUDE "" CACHE PATH "Frozen KVAF device headers")
if(NOT EXISTS "${KVAF_INCLUDE}/kvaf_batch_get_device.h")
  message(FATAL_ERROR "Missing frozen KVAF device headers")
endif()
get_filename_component(LI_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(INDEXER_DIR "${LI_ROOT}/op_kernel/indexer")
set(FIX_DIR "${CMAKE_CURRENT_BINARY_DIR}/isolated_indexer")
file(MAKE_DIRECTORY "${FIX_DIR}")
# Only the score-only experiment uses the historical end-of-task QScale
# release. Preserve the production source and all other release behavior.
file(READ "${INDEXER_DIR}/indexer_vector.h" VEC_ORIG)
set(OLD_RELEASE [=[#if !V9_UBMEM
    if (info.LoadQuery()) {
#else
    if (info.isFirstS2InnerLoop) {
#endif
        SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + qScalepingpong);
    }]=])
set(NEW_RELEASE [=[#if !V9_UBMEM
    if (scoreOnly ? info.ReleaseQuery() : info.LoadQuery()) {
#else
    if (info.isFirstS2InnerLoop) {
#endif
        SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + qScalepingpong);
    }]=])
string(FIND "${VEC_ORIG}" "${OLD_RELEASE}" ANCHOR_POS)
if(ANCHOR_POS EQUAL -1)
  message(FATAL_ERROR "Frozen QScale release anchor missing")
endif()
string(REPLACE "${OLD_RELEASE}" "${NEW_RELEASE}" VEC_FIXED "${VEC_ORIG}")
if(VEC_FIXED STREQUAL VEC_ORIG)
  message(FATAL_ERROR "Isolated QScale lifetime patch not applied")
endif()
file(WRITE "${FIX_DIR}/indexer_vector.h" "${VEC_FIXED}")
include(${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
ascendc_library(nextgen_v2_score_kernel STATIC b16_score_queue.cpp)
ascendc_include_directories(nextgen_v2_score_kernel PRIVATE
  ${FIX_DIR} ${LI_ROOT}/include ${LI_ROOT}/op_kernel ${INDEXER_DIR}
  ${KVAF_INCLUDE} ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include)
ascendc_compile_options(nextgen_v2_score_kernel PRIVATE
  -DKVIP_STREAMING_LI=1 -DKVIP_OUTPUT_WORKER=1 -DKVIP_FUSE_QUERY_WORK=1
  -DKVIP_URMA_WINDOW_LI=1 -DV9_UBMEM=0
  -O2 -g --cce-auto-sync=off
  -mllvm -user-reserved-ub-size=223232 -DV9_USER_UB_BYTES=223232
  -DV9_PARTIAL_SCAN_SIMD=1 -DV9_PROBE_EPOCH=0 -DV9_INTERMEDIATE_PROFILING=0
  -DV9_BATCH_POST=2 -DSIM_Q=4 -DSIM_CAPTURE_HITS=1 -DV9_SIMULATION=0
  -DROTATION_PUBLISH_TILES=1)
add_library(nextgen_v2_score_bridge SHARED launch.cpp)
target_link_libraries(nextgen_v2_score_bridge PRIVATE nextgen_v2_score_kernel host_intf_pub ascendcl)
