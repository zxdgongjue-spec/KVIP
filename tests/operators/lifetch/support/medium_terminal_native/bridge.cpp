// SPDX-License-Identifier: Apache-2.0
#include <cstdint>

extern "C" uint32_t aclrtlaunch_v9_parallel_terminal_large(uint32_t, void*, void*);

extern "C" uint32_t kvip_private_medium_tail(uint32_t blocks, void* stream, void* binding) {
    return aclrtlaunch_v9_parallel_terminal_large(blocks, stream, binding);
}
