cmake_minimum_required(VERSION 3.16)
project(kvip_update_sim LANGUAGES C CXX)
set(CMAKE_BUILD_TYPE Release)
set(RUN_MODE sim CACHE STRING "Execution target")
set(SOC_VERSION Ascend950PR_9599 CACHE STRING "Target SoC")
set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/cann-9.1.0 CACHE PATH "CANN")
set(OP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../csrc/operators/lightning_indexer_with_fetch" CACHE PATH "Operator source")
include(${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
ascendc_library(update_kernel STATIC ${OP_ROOT}/op_kernel/index_cache_update.cpp)
ascendc_include_directories(update_kernel PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include)
ascendc_compile_options(update_kernel PRIVATE -O2 -mllvm -user-reserved-ub-size=223232)
add_executable(update_sim update_sim.cpp ${OP_ROOT}/op_host/index_cache_update.cpp)
target_link_libraries(update_sim PRIVATE update_kernel host_intf_pub ascendcl)
# Simulator driver symbols are provided by msprof's process launcher.
target_link_options(update_sim PRIVATE -Wl,--allow-shlib-undefined)

set(KVAF_INCLUDE "" CACHE PATH "Patched KVAF headers for resolver validation")
if(KVAF_INCLUDE)
  ascendc_library(resolver_kernel STATIC resolver_kernel.cpp)
  ascendc_include_directories(resolver_kernel PRIVATE ${OP_ROOT}/include ${OP_ROOT}/op_kernel ${KVAF_INCLUDE}
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include)
  ascendc_compile_options(resolver_kernel PRIVATE -O2 --cce-auto-sync=off -mllvm -user-reserved-ub-size=223232
    -DV9_USER_UB_BYTES=223232 -DV9_SIMULATION=0 -DV9_BATCH_POST=2 -DSIM_CAPTURE_HITS=1)
  add_executable(resolver_sim resolver_sim.cpp)
  target_include_directories(resolver_sim PRIVATE ${OP_ROOT}/include ${KVAF_INCLUDE})
  target_link_libraries(resolver_sim PRIVATE resolver_kernel host_intf_pub ascendcl)
  target_link_options(resolver_sim PRIVATE -Wl,--allow-shlib-undefined)
endif()

if(KVAF_INCLUDE)
  add_executable(resolver_batch_sim resolver_batch_sim.cpp)
  target_include_directories(resolver_batch_sim PRIVATE ${OP_ROOT}/include ${KVAF_INCLUDE})
  target_link_libraries(resolver_batch_sim PRIVATE resolver_kernel host_intf_pub ascendcl)
  target_link_options(resolver_batch_sim PRIVATE -Wl,--allow-shlib-undefined)

  # Isolated experimental Q-by-Q resolver; the production operator is untouched.
  ascendc_library(staged_dma2_kernel STATIC resolver_staged_dma2_kernel.cpp)
  ascendc_include_directories(staged_dma2_kernel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${OP_ROOT}/include ${OP_ROOT}/op_kernel ${KVAF_INCLUDE}
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include)
  ascendc_compile_options(staged_dma2_kernel PRIVATE -O2 --cce-auto-sync=off -mllvm -user-reserved-ub-size=223232
    -DV9_USER_UB_BYTES=223232 -DV9_SIMULATION=0 -DV9_BATCH_POST=2 -DSIM_CAPTURE_HITS=1)
  add_executable(staged_dma2_batch_sim resolver_batch_sim.cpp)
  target_compile_definitions(staged_dma2_batch_sim PRIVATE STAGED_RESOLVER=1)
  target_include_directories(staged_dma2_batch_sim PRIVATE ${OP_ROOT}/include ${KVAF_INCLUDE})
  target_link_libraries(staged_dma2_batch_sim PRIVATE staged_dma2_kernel host_intf_pub ascendcl)
  target_link_options(staged_dma2_batch_sim PRIVATE -Wl,--allow-shlib-undefined)
endif()

ascendc_library(topk_classify_kernel STATIC topk_classify_kernel.cpp)
ascendc_include_directories(topk_classify_kernel PRIVATE ${OP_ROOT}/op_kernel/indexer ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include)
ascendc_compile_options(topk_classify_kernel PRIVATE -O2 --cce-auto-sync=off -mllvm -user-reserved-ub-size=196608)
add_executable(topk_classify_sim topk_classify_sim.cpp)
target_link_libraries(topk_classify_sim PRIVATE topk_classify_kernel host_intf_pub ascendcl)
target_link_options(topk_classify_sim PRIVATE -Wl,--allow-shlib-undefined)
