// SPDX-License-Identifier: Apache-2.0
#pragma once
__simt_callee__ inline uint32_t OffsetBucket(uint32_t token, uint32_t mask) {
    uint32_t mixed = token * 0x9e3779b9U;
    return (mixed ^ (mixed >> 16)) & mask;
}

// Hash stores the lowest index in the concatenation Q1,Q2,Q3,Q4. Comparing
// through the immutable input lets one word represent both key and priority.
__simt_callee__ inline uint32_t FindFirst(__ubuf__ uint32_t *table, __ubuf__ uint32_t *ids, uint32_t mask,
                                          uint32_t token) {
    uint32_t h = OffsetBucket(token, mask);
    while (ids[table[h]] != token)
        h = (h + 1) & mask;
    return table[h];
}
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void PrefixRanks(__ubuf__ uint32_t *ids, uint32_t n,
                                                                 uint32_t begin, uint32_t end, uint32_t buckets,
                                                                 __ubuf__ uint32_t *table, __ubuf__ uint32_t *rank,
                                                                 __ubuf__ uint32_t *ctl) {
    uint32_t t = threadIdx.x, mask = buckets - 1;
    for (uint32_t i = t; i < buckets; i += 512)
        table[i] = UINT32_MAX;
    if (!t) {
        ctl[1] = 0;
        ctl[2] = 0;
        ctl[3] = 0;
        ctl[4] = 0;
        ctl[5] = 0;
        ctl[6] = 0;
    }
    asc_syncthreads();
    for (uint32_t i = t; i < n; i += 512) {
        uint32_t h = OffsetBucket(ids[i], mask);
        for (;;) {
            uint32_t prior = asc_atomic_cas(table + h, UINT32_MAX, i);
            if (prior == UINT32_MAX)
                break;
            if (ids[prior] == ids[i]) {
                asc_atomic_min(table + h, i);
                break;
            }
            h = (h + 1) & mask;
        }
    }
    asc_syncthreads();
    uint32_t chunk = (n + 511) / 512, start = t * chunk, stop = (n < start + chunk ? n : start + chunk);
    uint32_t own = 0;
    for (uint32_t i = start; i < stop; ++i) {
        uint32_t first = FindFirst(table, ids, mask, ids[i]);
        rank[i] = first;
        own += first == i;
    }
    uint32_t count = StagedPrefix(own, ctl);
    for (uint32_t i = start; i < stop; ++i) {
        if (rank[i] == i)
            rank[i] = kStagedRankTag | count++;
        if (i + 1 == begin)
            ctl[2] = count;
        if (i + 1 == end)
            ctl[3] = count;
    }
    asc_syncthreads();
    if (!t)
        ctl[4] = ctl[559];
}
__simt_callee__ inline bool Protected(__gm__ uint32_t *bits, uint32_t words, uint32_t slot) {
    uint32_t w = slot / 32;
    return ((asc_ldcg(bits + w) | asc_ldcg(bits + words + w) | asc_ldcg(bits + 2 * words + w) |
             asc_ldcg(bits + 3 * words + w)) >>
            (slot % 32)) &
           1U;
}

// A single deterministic sequence for every prefix: unused legal LRU entries,
// then unprotected physical slots from the entire resident cache.
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void ChooseSlots(
    uint32_t resident, uint32_t used, bool malformedPrefix, __gm__ int32_t *inv, __gm__ uint32_t *bits,
    __ubuf__ uint32_t *lru, __ubuf__ uint32_t *slots, __ubuf__ uint32_t *bitmap, __ubuf__ uint32_t *ctl) {
    uint32_t t = threadIdx.x, need = ctl[4], words = (resident + 31) / 32;
    if (!need)
        return;
    for (uint32_t i = t; i < need; i += 512) {
        uint32_t ord = used + i;
        if (ord >= kLruSlots) {
            asc_atomic_or(ctl + 5, 1U);
            continue;
        }
        uint32_t slot = lru[ord];
        if (slot >= resident || asc_ldcg(inv + slot) != int32_t(ord) || Protected(bits, words, slot))
            asc_atomic_or(ctl + 5, 1U);
        else
            slots[i] = slot;
    }
    asc_syncthreads();
    if (!ctl[5] && !malformedPrefix)
        return;
    if (!t)
        ctl[5] = 1;
    asc_syncthreads();
    for (uint32_t w = t; w < words; w += 512)
        bitmap[w] = asc_ldcg(bits + w) | asc_ldcg(bits + words + w) | asc_ldcg(bits + 2 * words + w) |
                    asc_ldcg(bits + 3 * words + w);
    asc_syncthreads();
    for (uint32_t i = t; i < used; i += 512) {
        uint32_t s = lru[i];
        if (s >= resident)
            asc_atomic_or(ctl + 1, 4U);
        else
            asc_atomic_or(bitmap + s / 32, 1U << (s % 32));
    }
    asc_syncthreads();
    uint32_t permitted = 0;
    for (uint32_t j = 0; j < 4; ++j) {
        uint32_t i = 4 * t + j, s = lru[i];
        if (s >= resident)
            continue; // Unfilled LRU tail entries are not victims.
        if (i >= used && asc_ldcg(inv + s) == int32_t(i) && !(bitmap[s / 32] & (1U << (s % 32))))
            permitted |= 1U << j;
    }
    uint32_t off = StagedPrefix(__popc(permitted), ctl);
    const uint32_t suffix = ctl[559];
    for (uint32_t j = 0; j < 4; ++j) {
        if (permitted & (1U << j)) {
            if (off < need) {
                uint32_t slot = lru[4 * t + j];
                slots[off] = slot;
                // Mark the destination on its own writer. The following barrier
                // makes every mark visible before any lane enumerates free slots.
                if (suffix < need)
                    asc_atomic_or(bitmap + slot / 32, 1U << (slot % 32));
            }
            ++off;
        }
    }
    if (ctl[1] || suffix >= need)
        return;
    asc_syncthreads();
    // Enumerate physical slots with LIDU's protected-bitmap rule. Each lane
    // owns consecutive words, so an exclusive count gives ascending physical
    // order across lanes without a shared eviction cursor.
    uint32_t chunk = (words + 511) / 512, first = t * chunk;
    uint32_t last = words < first + chunk ? words : first + chunk;
    uint32_t free = 0;
    for (uint32_t w = first; w < last; ++w) {
        uint32_t available = ~bitmap[w];
        if (w + 1 == words && resident % 32)
            available &= (1U << (resident % 32)) - 1;
        bitmap[w] = available;
        free += __popc(available);
    }
    uint32_t cursor = suffix + StagedPrefix(free, ctl);
    if (ctl[559] < need - suffix) {
        // LIDU likewise checks capacity after walking the entire hot cache.
        // Valid 16K/Q4/1024 rows always have enough space.
        if (!t)
            ctl[1] = 4;
        return;
    }
    for (uint32_t w = first; w < last && cursor < need; ++w) {
        uint32_t available = bitmap[w];
        while (available && cursor < need) {
            slots[cursor++] = w * 32 + __builtin_ctz(available);
            available &= available - 1;
        }
    }
}
