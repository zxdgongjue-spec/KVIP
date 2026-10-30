#pragma once
#include <stdint.h>
#include "contract_abi.h"
#if defined(__CCE_AICORE__)
#define V9_PLAN __simt_callee__ inline
#define V9_LOCAL __ubuf__
#define V9_GLOBAL __gm__
#else
#define V9_PLAN inline
#define V9_LOCAL
#define V9_GLOBAL
#endif

namespace rotating {
constexpr uint32_t kPlanHash = 16384;
constexpr uint32_t kPlanEntries = 8192;
constexpr uint32_t kPlanBitmap = 512;
// Scalar result words: status, unique count, misses, supplement, cursor.
enum PlanWord : uint32_t { kPlanError, kPlanUnique, kPlanMisses, kPlanSupplement, kPlanCursor };
V9_PLAN uint32_t PlanHash(uint32_t token) {
    return (token * 0x9e3779b1U) & (kPlanHash - 1);
}
V9_PLAN bool PlanBit(V9_LOCAL uint32_t* bits, uint32_t slot) {
    return (bits[slot / 32] >> (slot % 32)) & 1U;
}
V9_PLAN void PlanSet(V9_LOCAL uint32_t* bits, uint32_t slot) {
    bits[slot / 32] |= 1U << (slot % 32);
}

V9_PLAN bool PlanHot(V9_GLOBAL const uint32_t* bits, uint32_t token) {
    return (bits[token / 32] >> (token % 32)) & 1U;
}
// Production SIMT planner, also compiled by the host differential test.
// Only scalar/pointer parameters and a void return cross __simt_callee__.
// Workers have independent UB; persistent maps are read-only.
V9_PLAN void BuildResidentPlan(
    uint32_t queries, uint32_t budget, uint32_t length, int32_t originalCursor,
    V9_GLOBAL const uint32_t* hot, V9_GLOBAL const int32_t* forward,
    V9_GLOBAL const int32_t* reservations, V9_GLOBAL const int32_t* victims,
    V9_GLOBAL const uint32_t* victimBits,
    V9_LOCAL uint32_t* selected, V9_LOCAL uint32_t* locations,
    V9_LOCAL uint32_t* tokens, V9_LOCAL uint32_t* slots,
    V9_LOCAL uint32_t* hash, V9_LOCAL uint32_t* protectedSlots,
    V9_LOCAL uint32_t* result, V9_LOCAL uint32_t* prefetchedSlots = nullptr) {
    result[kPlanError] = result[kPlanUnique] = result[kPlanMisses] = result[kPlanSupplement] = 0;
    result[kPlanCursor] = uint32_t(originalCursor);
    for (uint32_t i = 0; i < kPlanHash; ++i) hash[i] = UINT32_MAX;
    for (uint32_t i = 0; i < kPlanBitmap; ++i) protectedSlots[i] = 0;
    for (uint32_t i = 0; i < kPlanEntries; ++i) locations[i] = UINT32_MAX;
    for (uint32_t i = 0; i < queries * 2048; ++i) {
        uint32_t token = selected[i];
        if (token == UINT32_MAX) continue;
        if (int64_t(token) >= int64_t(length) - queries + i / 2048 + 1) { result[kPlanError] = kInvalid; return; }
        uint32_t h = PlanHash(token);
        while (hash[h] != UINT32_MAX && tokens[hash[h] & 8191U] != token)
            h = (h + 1) & (kPlanHash - 1);
        if (hash[h] != UINT32_MAX) {
            uint32_t queryBit = 1U << (16 + i / 2048);
            if (hash[h] & queryBit) { result[kPlanError] = kInvalid; return; }
            hash[h] |= queryBit;
            continue;
        }
        uint32_t rank = result[kPlanUnique]++;
        hash[h] = rank | (1U << (16 + i / 2048)); tokens[rank] = token;
        int32_t slot = PlanHot(hot, token) ? forward[token] : reservations[token];
        if (slot >= 0) {
            if (uint32_t(slot) >= budget || PlanBit(protectedSlots, uint32_t(slot))) { result[kPlanError] = kInvalid; return; }
            PlanSet(protectedSlots, uint32_t(slot));
        }
        slots[rank] = uint32_t(slot);
    }
    if (result[kPlanUnique] > budget) { result[kPlanError] = kCapacity; return; }
    uint32_t victim = 0, scanned = 0;
    uint32_t cursor = originalCursor < 0 ? 0 : uint32_t(originalCursor) % budget;
    for (uint32_t i = 0; i < result[kPlanUnique]; ++i) {
        if (PlanHot(hot, tokens[i])) continue;
        ++result[kPlanMisses];
        if (slots[i] != UINT32_MAX) continue;
        ++result[kPlanSupplement];
        int32_t slot = -1;
        while (victim < 2048) {
            uint32_t candidate = uint32_t(victims[victim++]);
            if (!PlanBit(protectedSlots, candidate) && (!prefetchedSlots || !PlanBit(prefetchedSlots,candidate))) { slot = int32_t(candidate); break; }
        }
        if (slot < 0) {
            if (queries == 1) { result[kPlanError] = kCapacity; return; }
            while (scanned < budget) {
                uint32_t candidate = cursor;
                cursor = (cursor + 1) % budget; ++scanned;
                if (!((victimBits[candidate / 32] >> (candidate % 32)) & 1U) && !PlanBit(protectedSlots, candidate)) {
                    slot = int32_t(candidate); result[kPlanCursor] = int32_t(cursor); break;
                }
            }
            if (slot < 0 && prefetchedSlots) {
                // All ordinary space is occupied by U: sacrifice an unused
                // speculative item. The caller waits for its CQ before overwrite.
                for(uint32_t j=0;j<2048;++j) {
                    uint32_t candidate=uint32_t(victims[j]);
                    if(!PlanBit(protectedSlots,candidate)) { slot=int32_t(candidate); break; }
                }
            }
            if (slot < 0) { result[kPlanError] = kCapacity; return; }
        }
        slots[i] = uint32_t(slot); PlanSet(protectedSlots, uint32_t(slot));
    }
    // Resolve before compacting the token array referenced by the hash table.
    for (uint32_t i = 0; i < queries * 2048; ++i) {
        uint32_t token = selected[i];
        if (token == UINT32_MAX) continue;
        uint32_t h = PlanHash(token);
        while (tokens[hash[h] & 8191U] != token) h = (h + 1) & (kPlanHash - 1);
        locations[i] = slots[hash[h] & 8191U];
    }
    uint32_t miss = 0;
    for (uint32_t i = 0; i < result[kPlanUnique]; ++i) {
        if (!PlanHot(hot, tokens[i])) {
            tokens[miss] = tokens[i]; slots[miss] = slots[i]; ++miss;
        }
    }
    return;
}
}

#undef V9_PLAN
#undef V9_LOCAL
#undef V9_GLOBAL
