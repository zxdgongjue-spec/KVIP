// SPDX-License-Identifier: Apache-2.0
#pragma once

// A row holds at most two changed Index tuples per event. Compact position
// pairs grow from the front of the list, long-cycle edges from its back.
// The existing host allocation also covers the rare oversized lane spill.
namespace pipeline_ab {
namespace high_post_index {
constexpr uint32_t kCapacity = 2 * (kPrefetch + kSlots);
struct Edge {
    uint32_t src, dst;
};
struct Tuple {
    uint64_t key[16], address, logical;
    uint32_t scale, padding;
};
static_assert(sizeof(Edge) + sizeof(Tuple) == 160, "workspace allocation ABI");
constexpr uint32_t kLocalTuples = 1024;
static_assert(kLocalTuples * sizeof(Tuple) <= 223232, "Index spill threshold exceeds UB");
constexpr uint32_t kRowBytes = kCapacity * (sizeof(Edge) + sizeof(Tuple));

__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS)
inline void EmitEdges(__gm__ Edge* edges, __gm__ uint32_t* dense, uint32_t history,
                      bool denseMode, __ubuf__ uint32_t* work) {
    const uint32_t tid = threadIdx.x, size = denseMode ? history : MAP;
    uint32_t pairs = 0, singles = 0;
    for (uint32_t base = 0; base < size; base += THREADS) {
        uint32_t i = base + tid, dst = i, src = i;
        if (i < size) {
            if (denseMode) src = asc_ldcg(dense + i);
            else {
                dst = work[KEYS_POS + i];
                src = dst == EMPTY ? dst : work[VALUES_POS + i];
            }
        }
        bool changed = dst != src;
        if (changed && (dst >= history || src >= history)) asc_atomic_or(work + CONTROL, 1U);
        asc_syncthreads();
        if (work[CONTROL]) return;
        // Test the final composed bijection, never raw prefetch/final edges.
        bool reciprocal = changed && (denseMode ? asc_ldcg(dense + src) :
            get(work + KEYS_POS, work + VALUES_POS, src)) == dst;
        uint32_t pair = reciprocal && dst < src;
        uint32_t single = changed && !reciprocal;
        uint32_t ranks = prefix(pair | (single << 16), work + CONTROL + 32);
        uint32_t found = work[CONTROL + 32 + 15];
        uint32_t newPairs = pairs + (found & 65535U), newSingles = singles + (found >> 16);
        if (2 * newPairs + newSingles > kCapacity) {
            if (!tid) work[CONTROL] = 1;
            return;
        }
        if (pair || single) {
            uint32_t index = pair ? pairs + (ranks & 65535U) :
                kCapacity - 1 - singles - (ranks >> 16);
            asc_stcg(&edges[index].src, src);
            asc_stcg(&edges[index].dst, dst);
        }
        pairs = newPairs;
        singles = newSingles;
        asc_syncthreads();
    }
    if (!tid) { work[CONTROL + 1] = pairs; work[CONTROL + 2] = singles; }
    asc_threadfence();
}

template<class T>
__simt_callee__ inline void ExchangeTwoColumns(__gm__ T* column,
        uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    T x = asc_ldcg(column + a), y = asc_ldcg(column + b);
    T z = asc_ldcg(column + c), w = asc_ldcg(column + d);
    asc_stcg(column + a, y); asc_stcg(column + b, x);
    asc_stcg(column + c, w); asc_stcg(column + d, z);
}

// The plan only places disjoint two-cycles in edges[p0:p1]. Issue both
// records before their stores so an independent key can be in flight.
__simt_callee__ inline void ExchangeTwoPairs(__gm__ Edge* edges, uint32_t i, uint32_t field,
        __gm__ uint64_t* keys, __gm__ uint32_t* scales, __gm__ uint64_t* addr,
        __gm__ uint64_t* logical) {
    uint32_t a = asc_ldcg(&edges[i].src), b = asc_ldcg(&edges[i].dst);
    uint32_t c = asc_ldcg(&edges[i+16].src), d = asc_ldcg(&edges[i+16].dst);
    auto ka = keys + uint64_t(a)*16 + field, kb = keys + uint64_t(b)*16 + field;
    auto kc = keys + uint64_t(c)*16 + field, kd = keys + uint64_t(d)*16 + field;
    uint64_t x = asc_ldcg(ka), y = asc_ldcg(kb);
    uint64_t z = asc_ldcg(kc), w = asc_ldcg(kd);
    if (field < 2) ExchangeTwoColumns(field == 0 ? addr : logical, a, b, c, d);
    else if (field == 2) ExchangeTwoColumns(scales, a, b, c, d);
    asc_stcg(ka, y); asc_stcg(kb, x); asc_stcg(kc, w); asc_stcg(kd, z);
}

__simt_callee__ inline void TransferPairSlice(__gm__ Edge* edges, uint32_t p0, uint32_t p1,
        __gm__ uint64_t* keys, __gm__ uint32_t* scales, __gm__ uint64_t* addr, __gm__ uint64_t* logical) {
    const uint32_t tuple=threadIdx.x/16, field=threadIdx.x%16;
    uint32_t i = p0 + tuple;
    for (; i + 16 < p1; i += 32)
        ExchangeTwoPairs(edges, i, field, keys, scales, addr, logical);
    if (i < p1) {
        uint32_t a = asc_ldcg(&edges[i].src), b = asc_ldcg(&edges[i].dst);
        auto ka = keys + uint64_t(a) * 16 + field;
        auto kb = keys + uint64_t(b) * 16 + field;
        uint64_t left = asc_ldcg(ka), right = asc_ldcg(kb);
        // Keep one 16-lane group per key. Two lanes load the independent
        // 64-bit columns together; the third exchanges the 32-bit scale.
        auto column = field == 0 ? addr : logical;
        if (field < 2) {
            uint64_t x = asc_ldcg(column + a), y = asc_ldcg(column + b);
            asc_stcg(column + a, y);
            asc_stcg(column + b, x);
        } else if (field == 2) {
            uint32_t x = asc_ldcg(scales + a), y = asc_ldcg(scales + b);
            asc_stcg(scales + a, y);
            asc_stcg(scales + b, x);
        }
        asc_stcg(ka, right);
        asc_stcg(kb, left);
    }
}

template<bool Scatter, bool Spill>
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void Transfer(__gm__ Edge* edges, uint32_t p0, uint32_t p1, uint32_t s0, uint32_t s1,
                     __gm__ uint64_t* keys, __gm__ uint32_t* scales,
                     __gm__ uint64_t* addr, __gm__ uint64_t* logical,
                     __gm__ Tuple* spill, __ubuf__ Tuple* local) {
    const uint32_t tuple = threadIdx.x / 16, field = threadIdx.x % 16;
    if constexpr (!Scatter) TransferPairSlice(edges,p0,p1,keys,scales,addr,logical);
    for (uint32_t i = s0 + tuple; i < s1; i += 16) {
        auto edge = edges + kCapacity - 1 - i;
        uint32_t pos = Scatter ? asc_ldcg(&edge->dst) : asc_ldcg(&edge->src);
        __gm__ uint64_t* key = keys + uint64_t(pos) * 16 + field;
        if constexpr (Spill) {
            auto t = spill + i;
            if constexpr (Scatter) {
                asc_stcg(key, asc_ldcg(t->key + field));
            } else {
                asc_stcg(t->key + field, asc_ldcg(key));
            }
            if (field < 2) {
                auto column = field == 0 ? addr + pos : logical + pos;
                auto cached = field == 0 ? &t->address : &t->logical;
                if constexpr (Scatter) asc_stcg(column, asc_ldcg(cached));
                else asc_stcg(cached, asc_ldcg(column));
            } else if (field == 2) {
                if constexpr (Scatter) asc_stcg(scales + pos, asc_ldcg(&t->scale));
                else asc_stcg(&t->scale, asc_ldcg(scales + pos));
            }
        } else {
            auto t = local + i - s0;
            if constexpr (Scatter) {
                asc_stcg(key, t->key[field]);
            } else {
                t->key[field] = asc_ldcg(key);
            }
            if (field < 2) {
                auto column = field == 0 ? addr + pos : logical + pos;
                auto cached = field == 0 ? &t->address : &t->logical;
                if constexpr (Scatter) asc_stcg(column, *cached);
                else *cached = asc_ldcg(column);
            } else if (field == 2) {
                if constexpr (Scatter) asc_stcg(scales + pos, t->scale);
                else t->scale = asc_ldcg(scales + pos);
            }
        }
    }
    asc_threadfence();
}

__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void CommitInvalidations(__gm__ uint32_t* log, uint32_t first, uint32_t end,
                                __gm__ uint32_t* valid) {
    for (uint32_t i = first + threadIdx.x; i < end; i += 256)
        asc_stcg(valid + asc_ldcg(log + i * (sizeof(Tuple) / sizeof(uint32_t))), uint32_t(0));
    asc_threadfence();
}
} // namespace high_post_index
} // namespace pipeline_ab
