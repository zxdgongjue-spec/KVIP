// SPDX-License-Identifier: Apache-2.0
#pragma once

// Large batches have one LI service lane per request. All four completion
// owners have joined before this copy, including this request's prefetch CQ.
__simt_callee__ inline void SaveRequestEvents(__gm__ pipeline_ab::Args* a,
    __gm__ rotating::Context* ctx, uint32_t row, uint32_t q, uint32_t thread) {
    constexpr uint32_t eventWords = sizeof(rotating::Event) / sizeof(uint32_t);
    if (!a->completedEvents || thread >= eventWords) return;

    auto live = reinterpret_cast<__gm__ uint32_t*>(ctx->events);
    auto saved = reinterpret_cast<__gm__ uint32_t*>(a->completedEvents);
    for (uint32_t epoch = q; epoch < ctx->shape.rounds; epoch += 4) {
        const uint32_t word = (epoch * ctx->shape.serviceLanes + row) * eventWords + thread;
        asc_stcg(saved + word, asc_ldcg(live + word));
        asc_stcg(live + word, 0U);
    }
}
