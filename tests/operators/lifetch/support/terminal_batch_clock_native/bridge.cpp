// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
extern "C" uint32_t aclrtlaunch_v9_terminal_private_batch_clocks(uint32_t, void*, void*);
extern "C" uint32_t kvip_test_terminal_clocks(uint32_t cores, void* stream, void* binding) {
    return aclrtlaunch_v9_terminal_private_batch_clocks(cores, stream, binding);
}
