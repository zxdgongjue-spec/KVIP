// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
extern "C" uint32_t aclrtlaunch_v9_terminal_first_cq(uint32_t, void*, void*);
extern "C" uint32_t kvip_test_terminal_first_cq(uint32_t cores, void* stream, void* binding) {
    return aclrtlaunch_v9_terminal_first_cq(cores, stream, binding);
}
