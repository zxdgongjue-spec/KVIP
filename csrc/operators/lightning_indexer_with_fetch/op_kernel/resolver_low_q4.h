// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace rotating {

// Only the exact, zero-prefetch path uses locations as temporary hash handles.
// Valid resident/ABC slots fit below these marker bits. Before publication all
// handles are replaced with the same final slots as ResolveCompactQ4<true>.
constexpr uint32_t kLowQ4Bucket = 0x80000000U;
constexpr uint32_t kLowQ4First = 0x40000000U;
constexpr uint32_t kLowQ4BucketMask = 8191U;

__simt_vf__ __aicore__ LAUNCH_BOUND(kCompactThreads)
inline void ResolveLowQ4(
    StagedContextPointer ctx, uint32_t history, uint32_t resident, uint32_t batch,
    __ubuf__ uint32_t* selected, __ubuf__ uint32_t* loc,
    __ubuf__ uint32_t* keys, __ubuf__ uint32_t* ranks, __ubuf__ uint32_t* slots,
    __ubuf__ uint32_t* scratch, __ubuf__ uint32_t* ctl,
    __gm__ uint32_t* eventSource, __gm__ uint32_t* eventDest) {
    static_assert(kSlots <= 8192, "low-Q4 ranks and bucket handles use 13 bits");
    // The preceding initialization VF may reject geometry before touching
    // locations or hash storage. Never consume that stale scratch on failure.
    const uint32_t initializationError = ctl[1];
    // Join these reads before any thread can publish a new hash error below;
    // otherwise late threads could exit while their peers await a barrier.
    asc_syncthreads();
    if (initializationError) return;
    const uint32_t tid = threadIdx.x, buckets = ctl[7], per = kSlots / kCompactThreads;
    auto bitmap = ctl + 1024, lru = scratch + 4096;
    // Cache the successful lookup once. Each selected position owns its loc
    // word, while atomic_min preserves the first occurrence across all Qs.
    for (uint32_t i = tid; i < kSlots; i += kCompactThreads) {
        const uint32_t token = selected[i];
        if (token == UINT32_MAX) continue;
        if (loc[i] != UINT32_MAX) {
            if (loc[i] & (kLowQ4Bucket | kLowQ4First)) asc_atomic_or(ctl + 1, 1U);
            continue;
        }
        if (token >= history) { asc_atomic_or(ctl + 1, 1U); continue; }
        uint32_t bucket = SparseHash(token) & (buckets - 1), probe = 0;
        for (; probe < buckets; ++probe) {
            const uint32_t previous = asc_atomic_cas(keys + bucket, UINT32_MAX, token);
            if (previous == UINT32_MAX || previous == token) {
                asc_atomic_min(ranks + bucket, i);
                loc[i] = kLowQ4Bucket | bucket;
                break;
            }
            bucket = (bucket + 1) & (buckets - 1);
        }
        if (probe == buckets) asc_atomic_or(ctl + 1, 8U);
    }
    asc_syncthreads();
    if (ctl[1]) return;
    uint32_t local = 0, coldLocal = 0;
    for (uint32_t i = tid * per; i < (tid + 1) * per; ++i) {
        if (selected[i] == UINT32_MAX || !(loc[i] & kLowQ4Bucket)) continue;
        if (ranks[loc[i] & kLowQ4BucketMask] == i) {
            loc[i] |= kLowQ4First;
            ++local;
            coldLocal += selected[i] >= resident;
        }
    }
    const uint32_t offset = StagedPrefix(local | (coldLocal << 16), ctl);
    const uint32_t total = ctl[544 + kCompactThreads / warpSize - 1];
    const uint32_t needed = total & 0xffffU, neededSlots = total >> 16;
    uint32_t rankOffset = offset & 0xffffU, coldOffset = offset >> 16;
    if (ctl[1]) return;
    if (tid < 4) {
        const uint32_t end = ctl[544 + (tid + 1) * (kCompactThreads / 4 / warpSize) - 1];
        const uint32_t begin = tid ? ctl[544 + tid * (kCompactThreads / 4 / warpSize) - 1] : 0;
        ctl[2 + tid] = (end - begin) & 0xffffU;
    }
    if (!tid) ctl[0] = needed;
    for (uint32_t i = tid * per; i < (tid + 1) * per; ++i) {
        if (selected[i] == UINT32_MAX || !(loc[i] & kLowQ4First)) continue;
        const uint32_t bucket = loc[i] & kLowQ4BucketMask;
        ranks[bucket] = kStagedRankTag | (rankOffset++ << 13) | coldOffset;
        coldOffset += selected[i] >= resident;
    }
    asc_syncthreads();
    if (neededSlots) {
        uint32_t available = 0;
        constexpr uint32_t candidatesPerThread = kLruSlots / kCompactThreads;
        for (uint32_t i = tid * candidatesPerThread; i < (tid + 1) * candidatesPerThread; ++i) {
            const uint32_t slot = lru[i];
            if (slot >= resident) { asc_atomic_or(ctl + 1, 4U); continue; }
            available += !(bitmap[slot / 32] & (1U << (slot % 32)));
        }
        uint32_t rank = StagedPrefix(available, ctl);
        uint32_t allocated = ctl[544 + kCompactThreads / warpSize - 1];
        if (allocated > neededSlots) allocated = neededSlots;
        if (ctl[1]) return;
        for (uint32_t i = tid * candidatesPerThread; i < (tid + 1) * candidatesPerThread; ++i) {
            const uint32_t slot = lru[i];
            if (!(bitmap[slot / 32] & (1U << (slot % 32)))) {
                if (rank < allocated) slots[rank] = slot;
                ++rank;
            }
        }
        asc_syncthreads();
        for (uint32_t i = tid; i < allocated; i += kCompactThreads)
            asc_atomic_or(bitmap + slots[i] / 32, 1U << (slots[i] % 32));
        asc_syncthreads();
        if (allocated < neededSlots) {
            // Keep the original fallback order, including protection for late
            // query hits. LRU scratch is dead after this point.
            for (uint32_t i = tid; i < kSlots; i += kCompactThreads)
                if (loc[i] < resident) asc_atomic_or(bitmap + loc[i] / 32, 1U << (loc[i] % 32));
            asc_syncthreads();
            const uint32_t words = (resident + 31) / 32;
            auto counts = reinterpret_cast<__ubuf__ int32_t*>(scratch), offsets = counts + words;
            kvip_hot_slots::CountAvailableBitmapWordsBody(bitmap, counts, resident, 0, words, 0);
            asc_syncthreads();
            const uint32_t wordsPerThread = (words + kCompactThreads - 1) / kCompactThreads;
            uint32_t localWords = 0;
            for (uint32_t i = tid * wordsPerThread; i < (tid + 1) * wordsPerThread && i < words; ++i)
                localWords += counts[i];
            uint32_t prefix = StagedPrefix(localWords, ctl);
            if (!tid && ctl[544 + kCompactThreads / warpSize - 1] < neededSlots - allocated) ctl[1] = 4;
            for (uint32_t i = tid * wordsPerThread; i < (tid + 1) * wordsPerThread && i < words; ++i) {
                offsets[i] = prefix;
                prefix += counts[i];
            }
            asc_syncthreads();
            if (ctl[1]) return;
            kvip_hot_slots::WriteAvailableBitmapSlotsBody(bitmap, counts, offsets,
                reinterpret_cast<__ubuf__ int32_t*>(slots), resident, 0, words, 0,
                neededSlots - allocated, allocated);
            asc_syncthreads();
        }
    }
    auto reserved = reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot) + batch * resident;
    auto changes = reinterpret_cast<__gm__ uint32_t*>(ctx->changes) + batch * kChangeCapacity;
    auto changeSlots = reinterpret_cast<__gm__ uint32_t*>(ctx->changeSlots) + batch * kChangeCapacity;
    // Resolve duplicates and publish each unique event in this same pass.
    // This replaces repeated hash probes and the final full-bucket scan.
    for (uint32_t i = tid; i < kSlots; i += kCompactThreads) {
        const uint32_t token = selected[i], handle = loc[i];
        if (token == UINT32_MAX || !(handle & kLowQ4Bucket)) continue;
        const uint32_t rankPair = ranks[handle & kLowQ4BucketMask] & ~kStagedRankTag;
        const uint32_t slot = token < resident ? token : slots[rankPair & 8191U];
        loc[i] = slot;
        if (!(handle & kLowQ4First)) continue;
        const uint32_t rank = rankPair >> 13;
        const int32_t ordinal = asc_ldcg(reserved + slot);
        const uint32_t record = ordinal >= 0 ? uint32_t(ordinal) : kLruSlots + rank;
        asc_stcg(eventSource + 2048 + rank, token);
        asc_stcg(eventDest + 2048 + rank, slot);
        asc_stcg(changeSlots + record, slot);
        asc_stcg(changes + record, token);
    }
    if (!tid) {
        asc_stcg(reinterpret_cast<__gm__ uint32_t*>(ctx->changeCount) + batch, kLruSlots + needed);
        ctl[9] = 0;
    }
    asc_threadfence();
    asc_syncthreads();
}
}  // namespace rotating
