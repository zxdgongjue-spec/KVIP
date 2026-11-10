#pragma once
#include <stdint.h>

#ifndef V9_SKIP_INACTIVE
#define V9_SKIP_INACTIVE 1
#endif
static_assert(V9_SKIP_INACTIVE == 0 || V9_SKIP_INACTIVE == 1,
              "Inactive LI skipping must be enabled with 0 or 1");
#if defined(__CCE_AICORE__)
#define KVIP_IDLE_DEVICE __aicore__
#else
#define KVIP_IDLE_DEVICE
#endif
namespace QLICommon {
KVIP_IDLE_DEVICE constexpr bool InactiveQueryRow(uint32_t q0,uint32_t q1,uint32_t q2,uint32_t q3) {
    return V9_SKIP_INACTIVE && !(q0|q1|q2|q3);
}
KVIP_IDLE_DEVICE constexpr uint32_t InactiveWindowTarget(
    uint32_t trigger,uint32_t resident,uint32_t queryEnd,bool hasLru) {
    if(!queryEnd)return 0;
    if(hasLru && queryEnd<resident)queryEnd=resident;
    return queryEnd<trigger?queryEnd:trigger;
}
}
#undef KVIP_IDLE_DEVICE
