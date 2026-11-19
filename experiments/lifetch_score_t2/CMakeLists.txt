cmake_minimum_required(VERSION 3.16)
project(cold_contract LANGUAGES C CXX)
set(CMAKE_BUILD_TYPE Release)
set(RUN_MODE sim CACHE STRING "sim or npu")
set(SOC_VERSION Ascend950PR_9599 CACHE STRING "Target SoC")
set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/cann CACHE PATH "CANN")
set(KVAF_INCLUDE "" CACHE PATH "Patched KVAF device headers")
include(${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
set(_op ${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/operators/lightning_indexer_with_fetch)
ascendc_library(cold_device STATIC cold_contract.cpp)
ascendc_include_directories(cold_device PRIVATE ${_op}/include ${_op}/op_kernel ${_op}/op_kernel/indexer ${KVAF_INCLUDE} ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include)
ascendc_compile_options(cold_device PRIVATE -DKVIP_STREAMING_LI=1 -DKVIP_OUTPUT_WORKER=1 -DKVIP_FUSE_QUERY_WORK=1 -O2 -g --cce-auto-sync=off
  -mllvm -user-reserved-ub-size=223232 -DV9_USER_UB_BYTES=223232 -DV9_PARTIAL_SCAN_SIMD=1 -DV9_PROBE_EPOCH=0
  -DV9_INTERMEDIATE_PROFILING=0 -DV9_BATCH_POST=2 -DSIM_Q=4 -DSIM_CAPTURE_HITS=1 -DV9_SIMULATION=0 -DROTATION_PUBLISH_TILES=1)
add_executable(cold_contract_host cold_contract_host.cpp)
target_include_directories(cold_contract_host PRIVATE ${_op}/include ${KVAF_INCLUDE})
target_link_libraries(cold_contract_host PRIVATE cold_device host_intf_pub ascendcl)
# CANN Model resolves its simulator shared-library peers when loading.
target_link_options(cold_contract_host PRIVATE -Wl,--allow-shlib-undefined)
