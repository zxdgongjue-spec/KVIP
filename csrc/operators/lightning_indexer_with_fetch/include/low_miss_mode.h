// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
#include "pipeline_options.h"

namespace rotating {
// The existing control tensor is stable for the whole graph invocation. All
// cooperating cores must make the same decision: a single positive or invalid
// row retains the original pipeline. Zero is a caller-selected policy, not an
// assertion about this round's exact miss count; final Get remains unbounded
// by the speculative prefetch quota.
template <typename ContextType>
__aicore__ inline bool NoPrefetchRound(const __gm__ ContextType* ctx) {
#if KVIP_LOW_MISS_SKIP && !(defined(V9_UBMEM) && V9_UBMEM)
    if (!ctx->admissionControls || !ctx->shape.batch || ctx->shape.batch > 16) return false;
    auto controls = reinterpret_cast<const __gm__ int32_t*>(ctx->admissionControls);
    for (uint32_t b = 0; b < ctx->shape.batch; ++b) {
        auto row = controls + b * 4;
        if (row[0] != 0 || row[1] < 1 || row[1] > 480 ||
            row[2] < 0 || row[2] > 65535 || (row[3] != 0 && row[3] != 1)) return false;
    }
    return true;
#else
    (void)ctx;
    return false;
#endif
}
}  // namespace rotating
