// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

namespace terminal_credit {
#if defined(__CCE_AICORE__)
#define KVIP_CREDIT_INLINE __aicore__ inline
#else
#define KVIP_CREDIT_INLINE inline
#endif

struct Receipt {
    uint64_t error;
    bool consumed;
};

struct State {
    uint64_t error;
    uint32_t issued;
    uint32_t consumed;
    uint32_t successful;
};

// The first prefix was posted before Index. Only this query writes its SQ or
// consumes its CQ. On failure, quiesce the already-published suffix if the
// consumer advanced. A timeout leaves the remaining count for quarantine.
template<class Queue>
KVIP_CREDIT_INLINE State CompletePair(Queue& queue, uint32_t count) {
    State state{0, 1, 0, 0};
    while (state.consumed < state.issued) {
        if (!state.error && state.issued < count && state.issued - state.consumed < 2) {
            queue.Issue(state.issued++);
        }
        const auto receipt = queue.Consume(state.consumed);
        if (receipt.error && !state.error) state.error = receipt.error;
        if (!receipt.consumed) break;
        ++state.consumed;
        if (!receipt.error) ++state.successful;
    }
    return state;
}

#undef KVIP_CREDIT_INLINE
}
