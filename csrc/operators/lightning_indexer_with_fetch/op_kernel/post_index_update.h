// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace post_index {
constexpr uint32_t kThreads = 256;
constexpr uint32_t kEventThreads = 8;
constexpr uint32_t kMaxEvents = 8192;
constexpr uint32_t kFailureMaskWord = 31;

// Clear before the producer join; errors publish before the existing Post join.
// On failure only, finish this VF and its V_S edge before the Post/CQ join.
__simt_vf__ __aicore__ LAUNCH_BOUND(32)
inline void MarkPostFailure(__gm__ uint32_t* prefix, uint32_t row, uint64_t error) {
    if (!threadIdx.x && error) asc_atomic_or(prefix + kFailureMaskWord, uint32_t(1) << row);
    asc_threadfence();
}

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
// words, built inside block0 Pack; no event list is
// copied or compacted. Self-refills need Get/CQ but no Index permutation.
// The caller stages the prefix, failure mask and owners in UB after the
// collective Post join. Source/destination event arrays remain in GM.
template<bool Profile = false>
__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads)
inline void ApplyPlan(
    __gm__ uint32_t* sources, __gm__ uint32_t* destinations,
    uint32_t eventOffset, uint32_t eventStride, uint32_t history,
    __ubuf__ uint32_t* prefix, uint32_t batch, uint32_t first, uint32_t end,
    __ubuf__ uint32_t* rowOwners,
    __gm__ uint64_t* keys, __gm__ uint32_t* scales,
    __gm__ uint64_t* addresses, __gm__ uint64_t* logical,
    uint32_t addressStride, __ubuf__ uint32_t* work) {
    if constexpr (Profile) {
        if (!threadIdx.x) work[0] = 0;
        asc_syncthreads();
    }
    const uint32_t field = threadIdx.x % kEventThreads;
    uint32_t moved = 0, row = batch, rowBegin = 0, rowEnd = 0, owner = 0;
    uint64_t rowError = 0;
    // Each eight-lane group covers B<=16 with two register banks. This
    // retains four independent event groups per warp without uint4 ABI values.
    const uint32_t initialEvent = first + threadIdx.x / kEventThreads;
    uint32_t initialRow = batch, initialBegin = 0, initialEnd = 0;
    uint64_t initialError = 0;
    uint32_t initialOwner = 0;
    if (initialEvent < end) {
        const uint32_t boundaryLo = field < batch ? prefix[field + 1] : UINT32_MAX;
        const uint32_t boundaryHi = field + 8 < batch ? prefix[field + 9] : UINT32_MAX;
        const uint32_t ownerLo = field < batch && rowOwners ? rowOwners[field] : field;
        const uint32_t ownerHi = field + 8 < batch && rowOwners ? rowOwners[field + 8] : field + 8;
        const uint32_t shift = (threadIdx.x % 32) & (32 - kEventThreads);
        const uint32_t lo = (asc_ballot(field < batch && initialEvent >= boundaryLo) >> shift) & 0xffU;
        const uint32_t hi = (asc_ballot(field + 8 < batch && initialEvent >= boundaryHi) >> shift) & 0xffU;
        initialRow = __popc(lo) + __popc(hi);
        const uint32_t previous = initialRow ? initialRow - 1 : 0;
        initialBegin = asc_shfl(previous < 8 ? boundaryLo : boundaryHi, previous & 7, kEventThreads);
        if (!initialRow) initialBegin = 0;
        initialEnd = asc_shfl(initialRow < 8 ? boundaryLo : boundaryHi, initialRow & 7, kEventThreads);
        initialOwner = asc_shfl(initialRow < 8 ? ownerLo : ownerHi, initialRow & 7, kEventThreads);
        initialError = (prefix[kFailureMaskWord] >> initialRow) & 1U;
    }
    for (uint32_t i = first + threadIdx.x / kEventThreads; i < end; i += kThreads / kEventThreads) {
        uint32_t src = 0, dst = 0;
        if (!field) {
            if (i >= rowEnd) {
                // Reuse the collectively located first row. Later events only
                // advance across boundaries, including empty requests. Keep both
                // bounds so the same row never reloads its prefix per event.
                if (row == batch) {
                    row = initialRow;
                    rowBegin = initialBegin;
                    rowEnd = initialEnd;
                    rowError = initialError;
                    owner = initialOwner;
                } else {
                    while (row < batch && i >= prefix[row + 1]) ++row;
                    rowBegin = prefix[row];
                    rowEnd = prefix[row + 1];
                    rowError = (prefix[kFailureMaskWord] >> row) & 1U;
                    owner = rowOwners ? rowOwners[row] : row;
                }
            }
            if (!rowError) {
                const uint64_t event = uint64_t(row) * eventStride + eventOffset + i - rowBegin;
                src = asc_ldcg(sources + event);
                dst = asc_ldcg(destinations + event);
            }
        }
        // Every active eight-thread group reaches both broadcasts together. Its
        // lane 0 has classified the event; failed rows use a no-swap sentinel.
        src = asc_shfl(src, 0, kEventThreads);
        dst = asc_shfl(dst, 0, kEventThreads);
        if (src == dst) continue;
        const uint32_t eventOwner = asc_shfl(owner, 0, kEventThreads);
        uint64_t source = (uint64_t(eventOwner) * history + src) * 16 + field;
        uint64_t destination = (uint64_t(eventOwner) * history + dst) * 16 + field;
        uint64_t a = asc_ldcg(keys + source), b = asc_ldcg(keys + destination);
        uint64_t c = asc_ldcg(keys + source + 8), d = asc_ldcg(keys + destination + 8);
        asc_stcg(keys + source, b);
        asc_stcg(keys + destination, a);
        asc_stcg(keys + source + 8, d);
        asc_stcg(keys + destination + 8, c);
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
        // One addition per eight-thread group, not one atomic per event.
        if (!field && moved) asc_atomic_add(work, moved);
    }
    // hotValid remains untouched until the transport's successful CQ drain.
    asc_threadfence();
    asc_syncthreads();
}
}  // namespace post_index
