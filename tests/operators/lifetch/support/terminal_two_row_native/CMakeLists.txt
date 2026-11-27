cmake_minimum_required(VERSION 3.16)
project(kvip_terminal_two_rows LANGUAGES CXX C)
set(CMAKE_BUILD_TYPE Release)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/prepare_overlay.py
    ${OP}/op_kernel/terminal ${CMAKE_CURRENT_BINARY_DIR} RESULT_VARIABLE overlay_result)
if(overlay_result)
    message(FATAL_ERROR "Pinned row preparation does not match")
endif()
set(RUN_MODE npu)
include(${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
ascendc_library(two_row_kernel STATIC kernel.cpp)
ascendc_include_directories(two_row_kernel PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR} ${OP}/op_kernel/terminal
  ${OP}/include ${OP}/op_kernel ${OP}/op_kernel/indexer ${KVAF_INCLUDE}
  ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include)
ascendc_compile_options(two_row_kernel PRIVATE -DKVIP_PARALLEL_TERMINAL=1
  -O2 -g --cce-auto-sync=off -mllvm -user-reserved-ub-size=223232
  -DV9_USER_UB_BYTES=223232 -DV9_BATCH_POST=2 -DSIM_Q=4 -DSIM_CAPTURE_HITS=1
  -DV9_SIMULATION=0 -DKVIP_STREAMING_LI=1 -DKVIP_OUTPUT_WORKER=1 -DROTATION_PUBLISH_TILES=1)
add_library(two_row_bridge SHARED bridge.cpp)
target_link_libraries(two_row_bridge PRIVATE two_row_kernel host_intf_pub ascendcl)
