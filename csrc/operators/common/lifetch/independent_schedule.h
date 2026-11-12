#pragma once
#include <stdint.h>

namespace rotating {
// This local LI implementation uses MIX1:2. The subsequent AIV-only tail
// has its own physical budget; publication records are not resident workers.
inline uint32_t ValidateStreamingBudget(uint32_t batch, uint32_t aic,
                                       uint32_t aiv, uint32_t li,
                                       uint32_t fetchPerQuery) {
    if (!batch || batch > 16 || !aic || aic > 32 || !aiv || aiv > 64 ||
        !li || li > aic || 2 * li > aiv ||
        (fetchPerQuery != 1 && fetchPerQuery != 2 && fetchPerQuery != 4) ||
        3 * batch + 4 * fetchPerQuery > aiv) return 100000;
    return 0;
}
}
