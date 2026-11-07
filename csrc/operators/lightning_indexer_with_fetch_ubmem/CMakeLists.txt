cmake_minimum_required(VERSION 3.16)
project(kvip_lifetch_ubmem LANGUAGES C CXX)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(RUN_MODE npu CACHE STRING "UBmem requires real NPU execution")
set(SOC_VERSION "" CACHE STRING "Exact active Ascend950 SoC")
set(ASCEND_CANN_PACKAGE_PATH "" CACHE PATH "Matching CANN installation")
set(SIM_BATCH 1 CACHE STRING "Compiled active batch, 1..16")
set(SIM_HISTORY 33024 CACHE STRING "Compiled Index capacity")
set(SIM_RESIDENT 8192 CACHE STRING "Compiled hot capacity")
set(INDEX_HEADS 64 CACHE STRING "Index heads: 32 or 64")
set(TOKEN_BYTES 656 CACHE STRING "KV token bytes: 656 or 2624")
set(AIC_COUNT "" CACHE STRING "Physical cube cores from aclrtGetDeviceInfo")
set(INTERMEDIATE_FETCHES 8 CACHE STRING "Requested windows, clipped to active LI cores")
set(LI_TILING auto CACHE STRING "auto split Q, or q4")
set(FINAL_AIVS 0 CACHE STRING "Final workers per batch: 0 auto, 1, 2, or 4")
if(NOT RUN_MODE STREQUAL "npu" OR NOT SOC_VERSION MATCHES "^Ascend950[A-Za-z0-9_]+$")
  message(FATAL_ERROR "UBmem fusion needs RUN_MODE=npu and explicit active Ascend950 SOC_VERSION")
endif()
foreach(_number SIM_BATCH SIM_HISTORY SIM_RESIDENT INDEX_HEADS TOKEN_BYTES AIC_COUNT INTERMEDIATE_FETCHES FINAL_AIVS)
  if(NOT "${${_number}}" MATCHES "^[0-9]+$")
    message(FATAL_ERROR "${_number} must be supplied as an unsigned integer")
  endif()
endforeach()
if(SIM_BATCH LESS 1 OR SIM_BATCH GREATER 16 OR AIC_COUNT LESS 1 OR AIC_COUNT GREATER 32 OR SIM_BATCH GREATER AIC_COUNT)
  message(FATAL_ERROR "UBmem grouped controller requires B1..16 and enough physical AICs")
endif()
if(NOT INDEX_HEADS MATCHES "^(32|64)$" OR NOT TOKEN_BYTES MATCHES "^(656|2624)$" OR
   NOT LI_TILING MATCHES "^(auto|q4)$" OR NOT FINAL_AIVS MATCHES "^(0|1|2|4)$")
  message(FATAL_ERROR "Unsupported UBmem static specialization")
endif()
math(EXPR _history_remainder "${SIM_HISTORY}%256")
math(EXPR _resident_remainder "${SIM_RESIDENT}%256")
math(EXPR _cold_tiles "(${SIM_HISTORY}-${SIM_RESIDENT})/128")
math(EXPR _split_tasks "${SIM_BATCH}*2")
set(_split 0)
if(LI_TILING STREQUAL "auto" AND _split_tasks LESS_EQUAL AIC_COUNT)
  set(_split 1)
endif()
math(EXPR _li_cores "${SIM_BATCH}*(1+${_split})")
set(_rounds ${INTERMEDIATE_FETCHES})
if(_rounds GREATER _li_cores)
  set(_rounds ${_li_cores})
endif()
if(NOT _history_remainder EQUAL 0 OR NOT _resident_remainder EQUAL 0 OR
   SIM_RESIDENT LESS 8192 OR SIM_HISTORY GREATER 210944 OR
   _cold_tiles LESS_EQUAL _rounds OR _cold_tiles LESS 1 OR INTERMEDIATE_FETCHES GREATER 32)
  message(FATAL_ERROR "Requires 256-aligned H>R>=8192, bounded LI scratch, and distinct cold window boundaries")
endif()
if(NOT EXISTS "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
  message(FATAL_ERROR "Matching CANN AscendC CMake package is absent")
endif()
include("${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
set(_definitions
  SIM_BATCH=${SIM_BATCH} SIM_HISTORY=${SIM_HISTORY} SIM_RESIDENT=${SIM_RESIDENT}
  INDEX_HEADS=${INDEX_HEADS} TOKEN_BYTES=${TOKEN_BYTES} SIM_Q=4 SIM_CAPTURE_HITS=1
  V9_AIC_COUNT=${AIC_COUNT} V9_SPLIT_Q=${_split} V9_INTERMEDIATE_FETCHES=${_rounds}
  V9_FINAL_AIVS=${FINAL_AIVS} V9_SERVICE_AICS=0 V9_FINAL_REPLICATED=1
  V9_UBMEM=1 V9_UBPIPE_DEPTH=8 V9_UBPIPE_ROWS=4 V9_WINDOW_UBPIPE_DEPTH=8
  V9_WINDOW_COPY_ENGINE=0 V9_WINDOW_SCAN_FUSED=0 V9_WINDOW_PREPARED_GROUPS=0
  V9_PHASE_PROFILING=0 V9_INTERMEDIATE_PROFILING=0 V9_FUSED_LIFECYCLE=0
  V9_DEDICATED_SERVICE=0 V9_SIMULATION=0 V9_LI2_PREFETCH=0 V9_BATCH_POST=2
  V9_PARTIAL_SCAN_SIMD=1 V9_USER_UB_BYTES=223232 ROTATION_PUBLISH_TILES=1)
set(_device_options)
foreach(_definition IN LISTS _definitions)
  list(APPEND _device_options "-D${_definition}")
endforeach()
ascendc_library(kvip_ubmem_kernel STATIC op_kernel/indexer.cpp op_kernel/lifecycle.cpp)
ascendc_include_directories(kvip_ubmem_kernel PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel"
  "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/indexer"
  "${CMAKE_CURRENT_SOURCE_DIR}/../common/lightning_indexer"
  "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/controller"
  "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/copy"
  "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/resolver"
  "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include")
ascendc_compile_options(kvip_ubmem_kernel PRIVATE ${_device_options} -O2 -g --cce-auto-sync=off
  -mllvm -user-reserved-ub-size=223232)
ascendc_library(kvip_ubmem_update_kernel STATIC op_kernel/index_cache_update.cpp)
ascendc_include_directories(kvip_ubmem_update_kernel PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/resolver"
  "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include")
ascendc_compile_options(kvip_ubmem_update_kernel PRIVATE ${_device_options} -O2 -mllvm -user-reserved-ub-size=223232)
add_library(kvip_lifetch_ubmem SHARED op_host/launch.cpp)
target_include_directories(kvip_lifetch_ubmem PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_definitions(kvip_lifetch_ubmem PRIVATE ${_definitions})
target_link_libraries(kvip_lifetch_ubmem PRIVATE kvip_ubmem_kernel kvip_ubmem_update_kernel host_intf_pub ascendcl)
