// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace post_index {
constexpr uint32_t kThreads = 512;
constexpr uint32_t kMaxEvents = 8192;

// Run before posting any Get, then join every AIV's error result. The compact
// Q4 resolver guarantees unique sources and destinations: missing resident
// rows refill themselves, and cold rows use unique, unprotected hot slots.
// A legal non-self event therefore cannot share either endpoint with any
// other event. This check rejects unsupported plans before Index mutation.
__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads)
inline void ValidatePlan(
    __gm__ uint32_t* sources, __gm__ uint32_t* destinations,
    uint32_t begin, uint32_t count, uint32_t resident, uint32_t history,
    uint32_t lane, uint32_t lanes, __ubuf__ uint32_t* work) {
    if (!threadIdx.x)
        work[0] = count > kMaxEvents || !lanes || lane >= lanes ||
                  resident > history ? 1U : 0U;
    asc_syncthreads();
    if (work[0]) return;
    uint32_t first = count * lane / lanes, end = count * (lane + 1) / lanes;
    for (uint32_t i = first + threadIdx.x; i < end; i += kThreads) {
        uint32_t src = asc_ldcg(sources + begin + i);
        uint32_t dst = asc_ldcg(destinations + begin + i);
        if (src >= history || dst >= resident || (src != dst && src < resident))
            asc_atomic_or(work, 1U);
    }
    asc_syncthreads();
}

// Preconditions: all plans passed ValidatePlan; Q4 dedup/protection invariants
// hold; every WQE address and TND output has consumed the original metadata.
// CQ completion may still be pending: it writes hot KV, not these Index rows.
// work holds 2*ceil(count/lanes) uint32 IDs (at most 64 KiB). Invalid resident
// self-refills still have a Get/CQ event but need no Index copy or permutation.
__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads)
inline void ApplyPlan(
    __gm__ uint32_t* sources, __gm__ uint32_t* destinations,
    uint32_t begin, uint32_t count, uint32_t resident, uint32_t history,
    __gm__ uint64_t* keys, __gm__ uint32_t* scales,
    __gm__ uint64_t* addresses, __gm__ uint64_t* logical,
    uint32_t lane, uint32_t lanes, __ubuf__ uint32_t* work) {
    uint32_t first = count * lane / lanes, end = count * (lane + 1) / lanes;
    uint32_t localCount = end - first;
    auto sourceRows = work, destRows = work + localCount;
    for (uint32_t i = threadIdx.x; i < localCount; i += kThreads) {
        sourceRows[i] = asc_ldcg(sources + begin + first + i);
        destRows[i] = asc_ldcg(destinations + begin + first + i);
    }
    asc_syncthreads();
    for (uint32_t x = threadIdx.x; x < localCount * 16; x += kThreads) {
        uint32_t i = x / 16, field = x % 16;
        uint32_t src = sourceRows[i], dst = destRows[i];
        if (src == dst) continue;
        uint64_t source = uint64_t(src) * 16 + field;
        uint64_t destination = uint64_t(dst) * 16 + field;
        uint64_t a = asc_ldcg(keys + source), b = asc_ldcg(keys + destination);
        asc_stcg(keys + source, b);
        asc_stcg(keys + destination, a);
    }
    for (uint32_t i = threadIdx.x; i < localCount; i += kThreads) {
        uint32_t src = sourceRows[i], dst = destRows[i];
        if (src == dst) continue;
        uint32_t a = asc_ldcg(scales + src), b = asc_ldcg(scales + dst);
        uint64_t x = asc_ldcg(addresses + src), y = asc_ldcg(addresses + dst);
        uint64_t u = asc_ldcg(logical + src), v = asc_ldcg(logical + dst);
        asc_stcg(scales + src, b); asc_stcg(scales + dst, a);
        asc_stcg(addresses + src, y); asc_stcg(addresses + dst, x);
        asc_stcg(logical + src, v); asc_stcg(logical + dst, u);
    }
    // hotValid remains untouched until the transport's successful CQ drain.
    asc_threadfence();
    asc_syncthreads();
}
}  // namespace post_index
