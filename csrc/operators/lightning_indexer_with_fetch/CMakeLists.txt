cmake_minimum_required(VERSION 3.16)
project(kvip_lifetch LANGUAGES C CXX)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(RUN_MODE npu CACHE STRING "NPU build")
set(SOC_VERSION Ascend950PR_9599 CACHE STRING "Target SoC, not runtime device ordinal")
set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/cann CACHE PATH "CANN installation")
set(KVAF_INCLUDE "" CACHE PATH "KVAF device interface headers")
option(KVIP_INCREMENTAL_TOPK "Producer compact pairs, incremental TopK and direct pending lists" OFF)
option(KVIP_INCREMENTAL_WINDOW "Fixed-address per-window incremental score staging" OFF)
option(KVIP_INCREMENTAL_REGISTER "Register Kth for incremental window selection" OFF)
option(KVIP_INCREMENTAL_PREFILTER "Fixed tile filtering and boundary-only sparse compaction" OFF)
option(KVIP_INCREMENTAL_DIRECT "Materialize selected scores and physical IDs in one SIMD VF" OFF)
option(KVIP_INCREMENTAL_INPLACE "Update only the boundary-tie suffix in place" OFF)
option(KVIP_FETCH_BITMAP "Fetch-owned per-request ranking" OFF)
option(KVIP_FETCH_DENSE "Fetch-owned per-request ranking" OFF)
option(KVIP_FETCH_PACKETS "Producer-packed candidates and demand-driven Fetch ranking" OFF)
option(KVIP_WINDOW_PROBE "Measure producer stages in a separate diagnostic binary" OFF)
option(KVIP_TRUNK_PRUNED "Skip scores below the retained native Kth at each window boundary" OFF)
option(KVIP_NATIVE_TRUNK "Use native fixed-block TopK at Fetch boundaries" OFF)
option(KVIP_SEED_PARALLEL "Prepare resident predicates on independent init blocks" OFF)
option(KVIP_SEED_MASK "Classify the initial winners from a shared resident bitmap" OFF)
option(KVIP_CONSUMED_PREFIX "Skip proven consumed list prefixes at Fetch" OFF)
option(KVIP_INCREMENTAL_SEGMENTS "Retain ranked prefixes across Fetch windows" OFF)
option(KVIP_INCREMENTAL_PROGRESS "Keep progress in an independent existing UB bank" OFF)
if(KVIP_INCREMENTAL_PROGRESS AND NOT KVIP_INCREMENTAL_WINDOW)
  message(FATAL_ERROR "Independent progress requires fixed-window incremental staging")
endif()
option(KVIP_INCREMENTAL_LIGHT "Map only new IDs and prepare list headers on vector" OFF)
option(KVIP_INCREMENTAL_TIES "Exact boundary-tie replacement without Kth selection" OFF)
if(KVIP_INCREMENTAL_INPLACE AND (NOT KVIP_INCREMENTAL_LIGHT OR NOT KVIP_INCREMENTAL_TIES))
  message(FATAL_ERROR "In-place boundary updates require light state and exact tie updates")
endif()
if(KVIP_INCREMENTAL_LIGHT AND NOT KVIP_INCREMENTAL_DIRECT)
  message(FATAL_ERROR "Light incremental state requires direct selection")
endif()
if(KVIP_INCREMENTAL_DIRECT AND NOT KVIP_INCREMENTAL_PREFILTER)
  message(FATAL_ERROR "Direct incremental state requires window prefilter")
endif()
if(KVIP_INCREMENTAL_TIES AND NOT KVIP_INCREMENTAL_DIRECT)
  message(FATAL_ERROR "Boundary tie updates require direct incremental state")
endif()
if(KVIP_INCREMENTAL_PREFILTER AND NOT KVIP_INCREMENTAL_REGISTER)
  message(FATAL_ERROR "Window prefilter requires register window selection")
endif()
if(KVIP_INCREMENTAL_WINDOW AND NOT KVIP_INCREMENTAL_TOPK)
  message(FATAL_ERROR "Window staging requires incremental TopK")
endif()
if(KVIP_INCREMENTAL_REGISTER AND NOT KVIP_INCREMENTAL_WINDOW)
  message(FATAL_ERROR "Register selection requires incremental window staging")
endif()
option(KVIP_REVERSE_COLD "Reverse complete 2K cold Index blocks in non-sequence-split URMA LI" OFF)
if(KVIP_FETCH_PACKETS AND (NOT KVIP_FETCH_BITMAP OR KVIP_FETCH_DENSE OR NOT KVIP_REVERSE_COLD))
  message(FATAL_ERROR "Producer packets require the reverse bitmap Fetch-rank path")
endif()
option(KVIP_ROLLING_THRESHOLD "Capped BF16 CDF in the native LI score VF" OFF)
option(KVIP_ROLLING_WINDOWED "Accumulate raw LI scores at existing window boundaries" OFF)
option(KVIP_ROLLING_COMPACT "One comparison per tile and early QK-buffer release" OFF)
option(KVIP_ROLLING_REUSE_SEED "Reuse the exact seed selector CDF within its byte range" OFF)
option(KVIP_ROLLING_FAST_PATH "Packed byte histogram and visibility-first admission" OFF)
option(KVIP_ROLLING_BOUNDARY "Read existing score GM only at window boundaries" OFF)
option(KVIP_ROLLING_SPARSE "Compact above-seed scores before window histogram updates" OFF)
if(KVIP_ROLLING_SPARSE AND NOT KVIP_ROLLING_FAST_PATH)
  message(FATAL_ERROR "Sparse window counts require packed seed reuse")
endif()
if(KVIP_ROLLING_FAST_PATH AND NOT KVIP_ROLLING_REUSE_SEED)
  message(FATAL_ERROR "Packed byte accumulation requires seed histogram reuse")
endif()
if(KVIP_ROLLING_REUSE_SEED AND NOT KVIP_ROLLING_COMPACT)
  message(FATAL_ERROR "Seed reuse requires compact window accumulation")
endif()
option(KVIP_ROLLING_FIRST_WINDOW "Correct Kth after the first cold window, then freeze" OFF)
if((KVIP_ROLLING_COMPACT OR KVIP_ROLLING_FIRST_WINDOW) AND NOT KVIP_ROLLING_WINDOWED)
  message(FATAL_ERROR "Compact and first-window options require window accumulation")
endif()
if(KVIP_ROLLING_FIRST_WINDOW AND NOT KVIP_ROLLING_COMPACT)
  message(FATAL_ERROR "First-window correction requires compact admission")
endif()
option(KVIP_ROLLING_BATCHED "Batch eight LI score tiles per CDF update" OFF)
option(KVIP_WINDOW_REFILTER "Recheck cold candidates against rolling Kth before admission" OFF)
option(KVIP_COMPACT_OUT_OF_PLACE "Compact admitted IDs into disjoint existing UB scratch" OFF)
option(KVIP_REFILTER_SAFE "Keep candidates with a causal query score not yet published" OFF)
option(KVIP_REFILTER_REFILL "Refill filtered candidate batches within the frozen window" OFF)
if(KVIP_REFILTER_SAFE AND NOT KVIP_WINDOW_REFILTER)
  message(FATAL_ERROR "Visibility-safe admission requires window refilter")
endif()
if(KVIP_REFILTER_REFILL AND NOT KVIP_REFILTER_SAFE)
  message(FATAL_ERROR "Same-window refill requires visibility-safe admission")
endif()
if((KVIP_WINDOW_REFILTER OR KVIP_ROLLING_BATCHED OR KVIP_ROLLING_WINDOWED) AND NOT KVIP_ROLLING_THRESHOLD)
  message(FATAL_ERROR "Window refilter and batched counts require rolling threshold")
endif()
if(KVIP_ROLLING_BATCHED AND KVIP_ROLLING_WINDOWED)
  message(FATAL_ERROR "Select either batch or window accumulation")
endif()
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
ascendc_library(v9_kernel STATIC op_kernel/indexer.cpp op_kernel/lifecycle.cpp op_kernel/streaming.cpp)
ascendc_include_directories(v9_kernel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
  ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/indexer ${KVAF_INCLUDE}
  ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include)
ascendc_compile_options(v9_kernel PRIVATE -DKVIP_STREAMING_LI=1 -DKVIP_OUTPUT_WORKER=$<BOOL:${KVIP_OUTPUT_WORKER}> -DKVIP_FUSE_QUERY_WORK=$<BOOL:${KVIP_FUSE_QUERY_WORK}> -O2 -g --cce-auto-sync=off
  -DKVIP_INCREMENTAL_WINDOW=$<BOOL:${KVIP_INCREMENTAL_WINDOW}>
  -DKVIP_INCREMENTAL_REGISTER=$<BOOL:${KVIP_INCREMENTAL_REGISTER}>
  -DKVIP_INCREMENTAL_PREFILTER=$<BOOL:${KVIP_INCREMENTAL_PREFILTER}>
  -DKVIP_INCREMENTAL_INPLACE=$<BOOL:${KVIP_INCREMENTAL_INPLACE}>
  -DKVIP_FETCH_BITMAP=$<BOOL:${KVIP_FETCH_BITMAP}>
  -DKVIP_FETCH_DENSE=$<BOOL:${KVIP_FETCH_DENSE}>
  -DKVIP_FETCH_PACKETS=$<BOOL:${KVIP_FETCH_PACKETS}>
  -DKVIP_WINDOW_PROBE=$<BOOL:${KVIP_WINDOW_PROBE}>
  -DKVIP_TRUNK_PRUNED=$<BOOL:${KVIP_TRUNK_PRUNED}>
  -DKVIP_NATIVE_TRUNK=$<BOOL:${KVIP_NATIVE_TRUNK}>
  -DKVIP_SEED_PARALLEL=$<BOOL:${KVIP_SEED_PARALLEL}>
  -DKVIP_SEED_MASK=$<BOOL:${KVIP_SEED_MASK}>
  -DKVIP_CONSUMED_PREFIX=$<BOOL:${KVIP_CONSUMED_PREFIX}>
  -DKVIP_INCREMENTAL_SEGMENTS=$<BOOL:${KVIP_INCREMENTAL_SEGMENTS}>
  -DKVIP_INCREMENTAL_PROGRESS=$<BOOL:${KVIP_INCREMENTAL_PROGRESS}>
  -DKVIP_INCREMENTAL_LIGHT=$<BOOL:${KVIP_INCREMENTAL_LIGHT}>
  -DKVIP_INCREMENTAL_DIRECT=$<BOOL:${KVIP_INCREMENTAL_DIRECT}>
  -DKVIP_INCREMENTAL_TIES=$<BOOL:${KVIP_INCREMENTAL_TIES}>
  -DKVIP_INCREMENTAL_TOPK=$<BOOL:${KVIP_INCREMENTAL_TOPK}>
  -DKVIP_REVERSE_COLD=$<BOOL:${KVIP_REVERSE_COLD}>
  -DKVIP_ROLLING_THRESHOLD=$<BOOL:${KVIP_ROLLING_THRESHOLD}>
  -DKVIP_ROLLING_COMPACT=$<BOOL:${KVIP_ROLLING_COMPACT}>
  -DKVIP_ROLLING_REUSE_SEED=$<BOOL:${KVIP_ROLLING_REUSE_SEED}>
  -DKVIP_ROLLING_BOUNDARY=$<BOOL:${KVIP_ROLLING_BOUNDARY}>
  -DKVIP_ROLLING_SPARSE=$<BOOL:${KVIP_ROLLING_SPARSE}>
  -DKVIP_ROLLING_FAST_PATH=$<BOOL:${KVIP_ROLLING_FAST_PATH}>
  -DKVIP_ROLLING_FIRST_WINDOW=$<BOOL:${KVIP_ROLLING_FIRST_WINDOW}>
  -DKVIP_ROLLING_WINDOWED=$<BOOL:${KVIP_ROLLING_WINDOWED}>
  -DKVIP_ROLLING_BATCHED=$<BOOL:${KVIP_ROLLING_BATCHED}>
  -DKVIP_COMPACT_OUT_OF_PLACE=$<BOOL:${KVIP_COMPACT_OUT_OF_PLACE}>
  -DKVIP_REFILTER_SAFE=$<BOOL:${KVIP_REFILTER_SAFE}>
  -DKVIP_REFILTER_REFILL=$<BOOL:${KVIP_REFILTER_REFILL}>
  -DKVIP_WINDOW_REFILTER=$<BOOL:${KVIP_WINDOW_REFILTER}>
  -mllvm -user-reserved-ub-size=223232 -DV9_USER_UB_BYTES=223232
  -DV9_PARTIAL_SCAN_SIMD=1 -DV9_PROBE_EPOCH=0 -DV9_INTERMEDIATE_PROFILING=0
  -DV9_BATCH_POST=2 -DSIM_Q=4 -DSIM_CAPTURE_HITS=1 -DV9_SIMULATION=0 -DROTATION_PUBLISH_TILES=1)
add_library(v9_bridge SHARED op_host/launch.cpp)
target_include_directories(v9_bridge PRIVATE ${KVAF_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR}/include ${_base}/include)
target_compile_definitions(v9_bridge PRIVATE KVIP_OUTPUT_WORKER=$<BOOL:${KVIP_OUTPUT_WORKER}>)
target_compile_definitions(v9_bridge PRIVATE KVIP_REVERSE_COLD=$<BOOL:${KVIP_REVERSE_COLD}>)
target_compile_definitions(v9_bridge PRIVATE KVIP_ROLLING_WINDOWED=$<BOOL:${KVIP_ROLLING_WINDOWED}> KVIP_ROLLING_BATCHED=$<BOOL:${KVIP_ROLLING_BATCHED}> KVIP_ROLLING_THRESHOLD=$<BOOL:${KVIP_ROLLING_THRESHOLD}> KVIP_WINDOW_REFILTER=$<BOOL:${KVIP_WINDOW_REFILTER}>)
target_compile_definitions(v9_bridge PRIVATE KVIP_ROLLING_COMPACT=$<BOOL:${KVIP_ROLLING_COMPACT}> KVIP_ROLLING_FIRST_WINDOW=$<BOOL:${KVIP_ROLLING_FIRST_WINDOW}>)
target_link_libraries(v9_bridge PRIVATE v9_kernel host_intf_pub ascendcl)
target_compile_definitions(v9_bridge PRIVATE KVIP_REFILTER_SAFE=$<BOOL:${KVIP_REFILTER_SAFE}> KVIP_REFILTER_REFILL=$<BOOL:${KVIP_REFILTER_REFILL}>)
target_compile_definitions(v9_bridge PRIVATE KVIP_COMPACT_OUT_OF_PLACE=$<BOOL:${KVIP_COMPACT_OUT_OF_PLACE}>)

target_compile_definitions(v9_bridge PRIVATE KVIP_ROLLING_REUSE_SEED=$<BOOL:${KVIP_ROLLING_REUSE_SEED}>)
target_compile_definitions(v9_bridge PRIVATE KVIP_ROLLING_FAST_PATH=$<BOOL:${KVIP_ROLLING_FAST_PATH}>)
target_compile_definitions(v9_bridge PRIVATE KVIP_ROLLING_SPARSE=$<BOOL:${KVIP_ROLLING_SPARSE}>)
target_compile_definitions(v9_bridge PRIVATE KVIP_ROLLING_BOUNDARY=$<BOOL:${KVIP_ROLLING_BOUNDARY}>)

target_compile_definitions(v9_bridge PRIVATE KVIP_INCREMENTAL_TOPK=$<BOOL:${KVIP_INCREMENTAL_TOPK}>)

target_compile_definitions(v9_bridge PRIVATE KVIP_INCREMENTAL_WINDOW=$<BOOL:${KVIP_INCREMENTAL_WINDOW}> KVIP_INCREMENTAL_REGISTER=$<BOOL:${KVIP_INCREMENTAL_REGISTER}>)

target_compile_definitions(v9_bridge PRIVATE KVIP_INCREMENTAL_PREFILTER=$<BOOL:${KVIP_INCREMENTAL_PREFILTER}>)

target_compile_definitions(v9_bridge PRIVATE KVIP_INCREMENTAL_DIRECT=$<BOOL:${KVIP_INCREMENTAL_DIRECT}> KVIP_INCREMENTAL_TIES=$<BOOL:${KVIP_INCREMENTAL_TIES}>)

target_compile_definitions(v9_bridge PRIVATE KVIP_INCREMENTAL_LIGHT=$<BOOL:${KVIP_INCREMENTAL_LIGHT}>)

target_compile_definitions(v9_bridge PRIVATE KVIP_INCREMENTAL_INPLACE=$<BOOL:${KVIP_INCREMENTAL_INPLACE}>)
target_compile_definitions(v9_bridge PRIVATE KVIP_INCREMENTAL_PROGRESS=$<BOOL:${KVIP_INCREMENTAL_PROGRESS}>)

target_compile_definitions(v9_bridge PRIVATE KVIP_FETCH_BITMAP=$<BOOL:${KVIP_FETCH_BITMAP}>)
target_compile_definitions(v9_bridge PRIVATE KVIP_FETCH_DENSE=$<BOOL:${KVIP_FETCH_DENSE}>)
target_compile_definitions(v9_bridge PRIVATE KVIP_FETCH_PACKETS=$<BOOL:${KVIP_FETCH_PACKETS}>)
target_compile_definitions(v9_bridge PRIVATE KVIP_WINDOW_PROBE=$<BOOL:${KVIP_WINDOW_PROBE}>)
target_compile_definitions(v9_bridge PRIVATE KVIP_TRUNK_PRUNED=$<BOOL:${KVIP_TRUNK_PRUNED}>)
target_compile_definitions(v9_bridge PRIVATE KVIP_NATIVE_TRUNK=$<BOOL:${KVIP_NATIVE_TRUNK}>)
target_compile_definitions(v9_bridge PRIVATE KVIP_SEED_PARALLEL=$<BOOL:${KVIP_SEED_PARALLEL}>)
target_compile_definitions(v9_bridge PRIVATE KVIP_SEED_MASK=$<BOOL:${KVIP_SEED_MASK}>)
target_compile_definitions(v9_bridge PRIVATE KVIP_CONSUMED_PREFIX=$<BOOL:${KVIP_CONSUMED_PREFIX}>)
target_compile_definitions(v9_bridge PRIVATE KVIP_INCREMENTAL_SEGMENTS=$<BOOL:${KVIP_INCREMENTAL_SEGMENTS}>)
