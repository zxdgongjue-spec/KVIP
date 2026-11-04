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
include(${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
ascendc_library(v9_kernel STATIC op_kernel/indexer.cpp op_kernel/lifecycle.cpp op_kernel/transport.cpp)
ascendc_include_directories(v9_kernel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
  ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/indexer ${KVAF_INCLUDE}
  ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include)
ascendc_compile_options(v9_kernel PRIVATE -DV9_RESOLVER_COMPARE=${RESOLVER_COMPARE} -O2 -g --cce-auto-sync=off
  -mllvm -user-reserved-ub-size=223232 -DV9_USER_UB_BYTES=223232
  -DV9_PARTIAL_SCAN_SIMD=1 -DV9_PROBE_EPOCH=0 -DV9_INTERMEDIATE_PROFILING=0
  -DV9_BATCH_POST=2 -DSIM_Q=4 -DSIM_CAPTURE_HITS=1 -DV9_SIMULATION=0 -DROTATION_PUBLISH_TILES=1)
add_library(v9_bridge SHARED op_host/launch.cpp)
target_include_directories(v9_bridge PRIVATE ${KVAF_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(v9_bridge PRIVATE V9_RESOLVER_COMPARE=${RESOLVER_COMPARE})
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
