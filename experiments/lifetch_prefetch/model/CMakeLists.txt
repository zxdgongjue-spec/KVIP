cmake_minimum_required(VERSION 3.16)
project(kvip_prefetch_service_model LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(RUN_MODE sim)
set(SOC_VERSION Ascend950PR_9599)
set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/cann CACHE PATH "CANN")
set(KVAF_INCLUDE "" CACHE PATH "KVAF URMA headers")
set(KVIP_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../.." CACHE PATH "KVIP checkout")
include(${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
set(op "${KVIP_SOURCE}/csrc/operators/lightning_indexer_with_fetch")
ascendc_library(service_kernel STATIC service.cpp)
ascendc_include_directories(service_kernel PRIVATE "${op}/include" "${op}/op_kernel"
  "${KVAF_INCLUDE}" "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include")
ascendc_compile_options(service_kernel PRIVATE -O2 --cce-auto-sync=off -mllvm -user-reserved-ub-size=223232
  -DV9_USER_UB_BYTES=223232 -DV9_SIMULATION=1 -DV9_BATCH_POST=2
  -DKVIP_SERVICE_UB_CANDIDATES=${UB_CANDIDATES} -DKVIP_SERVICE_POST_CHUNK=${POST_CHUNK})
add_executable(service_model host.cpp)
target_include_directories(service_model PRIVATE "${op}/include" "${KVAF_INCLUDE}")
target_link_libraries(service_model PRIVATE service_kernel host_intf_pub ascendcl)
target_link_options(service_model PRIVATE -Wl,--allow-shlib-undefined
  "-Wl,-rpath-link,${ASCEND_CANN_PACKAGE_PATH}/tools/simulator/${SOC_VERSION}/lib")
