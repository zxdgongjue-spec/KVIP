// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace rotating {
// Speculative admission is bounded independently of final TopK repair.
__aicore__ inline uint32_t PrefetchAllowance(uint32_t slots, uint32_t requests,
    uint32_t slotLimit, uint32_t prefetchLimit, uint32_t transportLimit, uint32_t windowLimit) {
    uint32_t available=windowLimit;
    uint32_t room=slots<slotLimit?slotLimit-slots:0;
    if(available>room)available=room;
    room=requests<prefetchLimit?prefetchLimit-requests:0;
    if(available>room)available=room;
    room=requests<transportLimit?transportLimit-requests:0;
    if(available>room)available=room;
    return available;
}
}
