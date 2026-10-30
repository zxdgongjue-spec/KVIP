#pragma once
#include <stdint.h>
#if defined(__CCE_AICORE__)
#define V9_CQ_INLINE __aicore__ inline
#else
#define V9_CQ_INLINE inline
#endif
namespace rotating {
struct CqDrainResult { uint64_t error; uint32_t completed; uint64_t polls; };
// Exclusive CQ consumer. Inspect every status; never infer success from the tail.
// pending < depth ensures no CQ space must be reclaimed before the tail arrives.
template<class Access>
V9_CQ_INLINE CqDrainResult DrainCq(Access& access, uint64_t consumer,
    uint32_t depth, uint32_t pending, uint32_t maxPolls,
    uint64_t invalid, uint64_t cqError, uint64_t timeout) {
    CqDrainResult result{0,0,0};
    if (!pending) return result;
    if (depth < 2 || pending >= depth) { result.error=invalid; return result; }
    const uint64_t initial=consumer;
    for (uint32_t i=0;i<pending;++i) {
        bool ready=false;
        for (uint32_t poll=0;poll<maxPolls;++poll) {
            uint32_t word=access.Read(uint32_t(consumer%depth));
            ++result.polls;
            if (((word>>2)&1U)==((consumer/depth)&1U)) continue;
            ready=true;
            ++consumer;
            if ((word>>16)!=0) result.error=cqError;
            else ++result.completed;
            break;
        }
        if (!ready) result.error=timeout;
        if (result.error) break;
    }
    // Also reclaim an observed error CQE or the successful prefix on timeout.
    if (consumer!=initial) access.Commit(consumer);
    return result;
}
}
#undef V9_CQ_INLINE
