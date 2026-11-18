// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace rotating {

// Only the exact, zero-prefetch path uses locations as temporary hash handles.
// Valid resident/ABC slots fit below these marker bits. Before publication all
// handles are replaced with the same final slots as ResolveCompactQ4<true>.
constexpr uint32_t kLowQ4Bucket = 0x80000000U;
constexpr uint32_t kLowQ4BucketMask = 8191U;

__simt_vf__ __aicore__ LAUNCH_BOUND(kCompactThreads)
inline void ResolveLowQ4(
    StagedContextPointer ctx, uint32_t history, uint32_t resident, uint32_t batch,
    __ubuf__ uint32_t* selected, __ubuf__ uint32_t* loc,
    __ubuf__ uint32_t* keys, __ubuf__ uint32_t* ranks, __ubuf__ uint32_t* slots,
    __ubuf__ uint32_t* scratch, __ubuf__ uint32_t* ctl,
    __gm__ uint32_t* eventSource, __gm__ uint32_t* eventDest) {
    static_assert(kSlots <= 8192, "low-Q4 ranks and bucket handles use 13 bits");
    const uint32_t tid = threadIdx.x;
    constexpr uint32_t per = kSlots / kCompactThreads;
    static_assert(per <= 32, "one lane mask covers its contiguous positions");
    const uint32_t begin = tid * per;
    auto bitmap = ctl + 1024, lru = scratch + 4096;
    const uint32_t words = (resident + 31) / 32;
    // The other Qs' three hit bitmaps temporarily occupy keys. Fold them here
    // before reusing that storage; LOW never needs per-query output masks.
    for (uint32_t w = tid; w < words; w += kCompactThreads)
        bitmap[w] |= keys[w] | keys[words + w] | keys[2 * words + w];
    if (tid < 10) ctl[tid] = 0;
    uint32_t pending = 0, inputError = 0;
    for (uint32_t bit = 0; bit < per; ++bit) {
        const uint32_t i = begin + bit, token = selected[i], slot = loc[i];
        if (token == UINT32_MAX) continue;
        if (slot == UINT32_MAX) {
            pending |= 1U << bit;
            inputError |= token >= history;
        } else {
            inputError |= (slot >> 30) != 0;
        }
    }
    StagedPrefix(__builtin_popcount(pending), ctl);
    const uint32_t pendingCount = ctl[544 + kCompactThreads / warpSize - 1];
    const uint32_t buckets = pendingCount <= 128 ? 256 : pendingCount <= 512 ? 1024 :
                             pendingCount <= 1024 ? 2048 : pendingCount <= 2048 ? 4096 : 8192;
    if (inputError) asc_atomic_or(ctl + 1, 1U);
    for (uint32_t i = tid; i < buckets; i += kCompactThreads) {
        keys[i] = UINT32_MAX;
        ranks[i] = UINT32_MAX;
    }
    asc_syncthreads();
    if (ctl[1]) return;
    if (!pendingCount) {
        if (!tid) asc_stcg(reinterpret_cast<__gm__ uint32_t*>(ctx->changeCount) + batch, kLruSlots);
        asc_threadfence();
        asc_syncthreads();
        return;
    }
    // A register bit names the original Q4 position. The remaining passes
    // visit only misses, retaining first-occurrence and victim order.
    for (uint32_t bits = pending; bits; bits &= bits - 1) {
        const uint32_t i = begin + __builtin_ctz(bits), token = selected[i];
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
    uint32_t first = 0, coldLocal = 0;
    for (uint32_t bits = pending; bits; bits &= bits - 1) {
        const uint32_t i = begin + __builtin_ctz(bits);
        if (ranks[loc[i] & kLowQ4BucketMask] == i) {
            first |= bits & -bits;
            coldLocal += selected[i] >= resident;
        }
    }
    const uint32_t offset = StagedPrefix(__builtin_popcount(first) | (coldLocal << 16), ctl);
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
    for (uint32_t bits = first; bits; bits &= bits - 1) {
        const uint32_t i = begin + __builtin_ctz(bits);
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
        if (allocated < neededSlots) {
            // The bitmap is read again only in the fallback allocation.
            for (uint32_t i = tid; i < allocated; i += kCompactThreads)
                asc_atomic_or(bitmap + slots[i] / 32, 1U << (slots[i] % 32));
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
    for (uint32_t bits = pending; bits; bits &= bits - 1) {
        const uint32_t i = begin + __builtin_ctz(bits);
        const uint32_t token = selected[i], handle = loc[i];
        const uint32_t rankPair = ranks[handle & kLowQ4BucketMask] & ~kStagedRankTag;
        const uint32_t slot = token < resident ? token : slots[rankPair & 8191U];
        loc[i] = slot;
        if (!(first & (bits & -bits))) continue;
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
