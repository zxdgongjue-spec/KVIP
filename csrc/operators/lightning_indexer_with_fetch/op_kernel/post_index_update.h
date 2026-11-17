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
// Split the whole batch's event list across all AIVs. Prefix is only batch+1
// words, built during the existing pre-Post error reduction; no event list is
// copied or compacted. Self-refills need Get/CQ but no Index permutation.
// Profile counters use 64 B of UB; headline invocations use no UB staging.
template<bool Profile = false>
__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads)
inline void ApplyPlan(
    __gm__ uint32_t* sources, __gm__ uint32_t* destinations,
    uint32_t eventOffset, uint32_t eventStride, uint32_t history,
    __gm__ uint32_t* prefix, uint32_t batch, uint32_t first, uint32_t end,
    __gm__ uint32_t* rowOwners, __gm__ uint64_t* records, uint32_t cores,
    __gm__ uint64_t* keys, __gm__ uint32_t* scales,
    __gm__ uint64_t* addresses, __gm__ uint64_t* logical,
    uint32_t addressStride, __ubuf__ uint32_t* work) {
    if constexpr (Profile) {
        if (!threadIdx.x) work[0] = 0;
        asc_syncthreads();
    }
    const uint32_t field = threadIdx.x % 16;
    uint32_t moved = 0, row = batch, rowBegin = 0, rowEnd = 0, owner = 0;
    uint64_t rowError = 0;
    for (uint32_t i = first + threadIdx.x / 16; i < end; i += kThreads / 16) {
        if (i >= rowEnd) {
            // First locate a row by binary search; later events only advance
            // across boundaries (including empty requests). Cache both bounds
            // so an event never reloads the row's prefix just to get its ID.
            if (row == batch) {
                uint32_t lo = 0, hi = batch;
                while (lo < hi) {
                    uint32_t mid = (lo + hi) / 2;
                    if (i >= asc_ldcg(prefix + mid + 1)) lo = mid + 1;
                    else hi = mid;
                }
                row = lo;
            } else {
                while (row < batch && i >= asc_ldcg(prefix + row + 1)) ++row;
            }
            rowBegin = asc_ldcg(prefix + row);
            rowEnd = asc_ldcg(prefix + row + 1);
            owner = rowOwners ? asc_ldcg(rowOwners + row) : row;
            rowError = 0;
            // A failed Post cancels this request's swaps, even if this AIV
            // originally posted for a different request. All posted queues
            // are still drained by the scalar caller after the Index barrier.
            for (uint32_t peer = row; peer < cores; peer += batch)
                rowError |= asc_ldcg(records + uint64_t(peer) * 16);
        }
        if (rowError) continue;
        const uint64_t event = uint64_t(row) * eventStride + eventOffset + i - rowBegin;
        uint32_t src = asc_ldcg(sources + event), dst = asc_ldcg(destinations + event);
        if (src == dst) continue;
        uint64_t source = (uint64_t(owner) * history + src) * 16 + field;
        uint64_t destination = (uint64_t(owner) * history + dst) * 16 + field;
        uint64_t a = asc_ldcg(keys + source), b = asc_ldcg(keys + destination);
        asc_stcg(keys + source, b);
        asc_stcg(keys + destination, a);
        if (!field) {
            uint64_t sourceRow = uint64_t(owner) * history + src;
            uint64_t destRow = uint64_t(owner) * history + dst;
            uint64_t sourceAddress = uint64_t(owner) * addressStride + src;
            uint64_t destAddress = uint64_t(owner) * addressStride + dst;
            uint32_t a = asc_ldcg(scales + sourceRow), b = asc_ldcg(scales + destRow);
            uint64_t x = asc_ldcg(addresses + sourceAddress), y = asc_ldcg(addresses + destAddress);
            uint64_t u = asc_ldcg(logical + sourceRow), v = asc_ldcg(logical + destRow);
            asc_stcg(scales + sourceRow, b); asc_stcg(scales + destRow, a);
            asc_stcg(addresses + sourceAddress, y); asc_stcg(addresses + destAddress, x);
            asc_stcg(logical + sourceRow, v); asc_stcg(logical + destRow, u);
            if constexpr (Profile) ++moved;
        }
    }
    if constexpr (Profile) {
        // One addition per 16-thread group, not one atomic per event.
        if (!field && moved) asc_atomic_add(work, moved);
    }
    // hotValid remains untouched until the transport's successful CQ drain.
    asc_threadfence();
    asc_syncthreads();
}
}  // namespace post_index
