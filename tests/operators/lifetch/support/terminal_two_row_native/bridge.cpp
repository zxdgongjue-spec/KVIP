// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
extern "C" uint32_t aclrtlaunch_v9_test_two_row_terminal(uint32_t,void*,void*,void*,uint64_t);
extern "C" uint32_t kvip_test_two_rows(uint32_t cores,void* stream,void* binding,void* workspace,uint64_t ticks){
    if(cores!=64 || !stream || !binding || !workspace || !ticks) return 1;
    return aclrtlaunch_v9_test_two_row_terminal(cores,stream,binding,workspace,ticks);
}
