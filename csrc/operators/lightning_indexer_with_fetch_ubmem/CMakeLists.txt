cmake_minimum_required(VERSION 3.16)
project(kvip_lifetch_ubmem LANGUAGES C CXX)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(RUN_MODE npu CACHE STRING "UBmem requires real NPU execution")
set(SOC_VERSION "" CACHE STRING "Exact active Ascend950 SoC")
set(ASCEND_CANN_PACKAGE_PATH "" CACHE PATH "Matching CANN installation")
set(KVAF_INCLUDE "" CACHE PATH "KVAF access_operator/include")
if(NOT RUN_MODE STREQUAL "npu" OR NOT SOC_VERSION MATCHES "^Ascend950[A-Za-z0-9_]+$")
  message(FATAL_ERROR "UBmem fusion needs RUN_MODE=npu and explicit active Ascend950 SOC_VERSION")
endif()
if(NOT EXISTS "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
  message(FATAL_ERROR "Matching CANN AscendC CMake package is absent")
endif()
if(NOT EXISTS "${KVAF_INCLUDE}/ubmem/fused_batch_get_device.h")
  message(FATAL_ERROR "KVAF UBmem device Get header is missing: ${KVAF_INCLUDE}")
endif()
include("${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
set(KVIP_STREAMING_UBMEM ON)
set(_streaming "${CMAKE_CURRENT_SOURCE_DIR}/../lightning_indexer_with_fetch")
set(_definitions
  TOKEN_BYTES=656 SIM_Q=4 SIM_CAPTURE_HITS=1
  V9_UBMEM=1 V9_UBPIPE_DEPTH=8 V9_UBPIPE_ROWS=4 V9_WINDOW_UBPIPE_DEPTH=8
  V9_WINDOW_COPY_ENGINE=0 V9_WINDOW_SCAN_FUSED=0 V9_WINDOW_PREPARED_GROUPS=0
  V9_PHASE_PROFILING=0 V9_INTERMEDIATE_PROFILING=0 V9_FUSED_LIFECYCLE=0
  V9_DEDICATED_SERVICE=0 V9_SIMULATION=0 V9_LI2_PREFETCH=0 V9_BATCH_POST=2
  V9_PARTIAL_SCAN_SIMD=1 V9_USER_UB_BYTES=223232 ROTATION_PUBLISH_TILES=1)
if(KVIP_STREAMING_UBMEM)
 list(APPEND _definitions KVIP_STREAMING_LI=1 KVIP_STREAMING_UBMEM=1 KVIP_FUSE_QUERY_WORK=1 KVIP_OUTPUT_WORKER=1)
endif()
set(_device_options)
foreach(_definition IN LISTS _definitions)
  list(APPEND _device_options "-D${_definition}")
endforeach()
# Compile the two fixed KV row pipelines into the same library. The LI
# geometry and host launches remain runtime-driven; only row-copy tiling differs.
set(_kernel_libraries)
foreach(_width IN ITEMS 656 2624)
  set(_target "kvip_ubmem_kernel_${_width}")
  set(_width_options ${_device_options})
  list(FILTER _width_options EXCLUDE REGEX "^-DTOKEN_BYTES=")
  list(APPEND _width_options "-DTOKEN_BYTES=${_width}")
  foreach(_name IN ITEMS init)
    list(APPEND _width_options "-Dkvip_ubmem_${_name}_kernel=kvip_ubmem_${_width}_${_name}_kernel")
  endforeach()
  set(_sources op_kernel/lifecycle.cpp)
  if(KVIP_STREAMING_UBMEM)
    list(APPEND _sources "${_streaming}/op_kernel/indexer.cpp" "${_streaming}/op_kernel/streaming.cpp")
    foreach(_suffix IN ITEMS "" _1 _2 _3)
      list(APPEND _width_options "-Dv9_li_kernel${_suffix}=kvip_ubmem_${_width}_li_kernel${_suffix}")
    endforeach()
    foreach(_entry IN ITEMS v9_streaming_resolve_update v9_counter_calibration v9_query_fetch_only)
      list(APPEND _width_options "-D${_entry}=${_entry}_${_width}")
    endforeach()
  endif()
  ascendc_library(${_target} STATIC ${_sources})
  if(KVIP_STREAMING_UBMEM)
    ascendc_include_directories(${_target} PRIVATE "${_streaming}/op_kernel" "${_streaming}/op_kernel/indexer"
      "${CMAKE_CURRENT_SOURCE_DIR}/../lightning_indexer_with_fetch/op_kernel")
  endif()
  ascendc_include_directories(${_target} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel"
    "${CMAKE_CURRENT_SOURCE_DIR}/../lightning_indexer_with_fetch/op_kernel/indexer"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/controller"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/copy"
    "${KVAF_INCLUDE}"
    "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include")
  ascendc_compile_options(${_target} PRIVATE ${_width_options} -O2 -g --cce-auto-sync=off
    -mllvm -user-reserved-ub-size=223232)
  list(APPEND _kernel_libraries ${_target})
endforeach()
add_library(kvip_lifetch_ubmem SHARED op_host/launch.cpp)
target_include_directories(kvip_lifetch_ubmem PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_definitions(kvip_lifetch_ubmem PRIVATE ${_definitions})
target_link_libraries(kvip_lifetch_ubmem PRIVATE ${_kernel_libraries} host_intf_pub ascendcl)
