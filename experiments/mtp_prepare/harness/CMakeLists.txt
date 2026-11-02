cmake_minimum_required(VERSION 3.16)
project(mtp_prepare_sim LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE Release)
set(RUN_MODE sim CACHE STRING "Simulator execution")
set(SOC_VERSION Ascend950PR_9599 CACHE STRING "SoC")
set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/cann-9.1.0 CACHE PATH "CANN")
set(MTP_VARIANT_DIR "" CACHE PATH "Archived MTP operator source directory")

if(NOT EXISTS "${MTP_VARIANT_DIR}/a5_lidu_fused_batch_get_prepare_mtp_implementation.inc")
  message(FATAL_ERROR "MTP_VARIANT_DIR does not contain the archived kernel")
endif()

include(${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)

ascendc_library(
  mtp_prepare_kernel STATIC kernel.cpp)
ascendc_include_directories(mtp_prepare_kernel PRIVATE ${MTP_VARIANT_DIR})
ascendc_compile_options(
  mtp_prepare_kernel PRIVATE -O2 -g --cce-auto-sync=off
  -I${MTP_VARIANT_DIR})

add_executable(mtp_prepare_sim main.cpp)
target_include_directories(
  mtp_prepare_sim PRIVATE
  ${MTP_VARIANT_DIR}
  ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
  ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime)
target_link_libraries(
  mtp_prepare_sim PRIVATE mtp_prepare_kernel host_intf_pub ascendcl)
target_link_options(
  mtp_prepare_sim PRIVATE
  "-Wl,-rpath-link,${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/simulator/dav_3510/lib")
