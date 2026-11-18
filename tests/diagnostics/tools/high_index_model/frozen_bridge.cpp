// SPDX-License-Identifier: Apache-2.0
#include <cstdint>

extern "C" uint32_t aclrtlaunch_kvip_frozen_high_tail(uint32_t blocks, void* stream, void* arguments);
extern "C" uint32_t kvip_replay_fixed_tail(uint32_t blocks, void* stream, void* arguments) {
    return aclrtlaunch_kvip_frozen_high_tail(blocks, stream, arguments);
}
