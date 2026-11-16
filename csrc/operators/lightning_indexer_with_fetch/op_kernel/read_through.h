// SPDX-License-Identifier: Apache-2.0
// Exact low-miss reads leave persistent resident KV and Index rows unchanged.
// Scratch slots live after the original hot/ABC output row and expire at the
// next invocation. A false-low prediction still fits all 4 * 2048 selections.
#pragma once

namespace read_through {
constexpr uint32_t kMagic = 0x52544831U;
constexpr uint32_t kBuckets = 16384;
constexpr uint32_t kKeys = 2 * kSlots;
constexpr uint32_t kRanks = kKeys + kBuckets;
constexpr uint32_t kWork = kRanks + kBuckets;
constexpr uint32_t kControl = kWork + kTop + 64;
constexpr uint32_t kScratchBytes = (kControl + 1024) * sizeof(uint32_t);
static_assert(kScratchBytes <= 223232, "read-through resolver exceeds user UB");

__aicore__ inline bool Active(__gm__ Context* ctx) {
    if (!ctx->readThrough) return false;
    if (ctx->admissionControls && ctx->shape.batch <= 16)
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->admissionControls),
                                    ctx->shape.batch * 16);
    return ctx->readThrough && NoPrefetchRound(ctx);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void Initialize(__ubuf__ uint32_t* keys, __ubuf__ uint32_t* ranks,
                       __ubuf__ uint32_t* ctl) {
    for (uint32_t i = threadIdx.x; i < kBuckets; i += 512) {
        keys[i] = UINT32_MAX;
        ranks[i] = UINT32_MAX;
    }
    if (threadIdx.x < 32) ctl[threadIdx.x] = 0;
    asc_syncthreads();
}

__simt_callee__ inline uint32_t Find(__ubuf__ uint32_t* keys, uint32_t token) {
    uint32_t bucket = SparseHash(token) & (kBuckets - 1);
    for (uint32_t probe = 0; probe < kBuckets; ++probe) {
        if (keys[bucket] == token) return bucket;
        bucket = (bucket + 1) & (kBuckets - 1);
    }
    return UINT32_MAX;
}

// One query at a time preserves Q0 Get overlap. The hash contains only cold
// selections and deduplicates them across all four queries; no LRU, victim
// bitmap, physical Index exchange, or hot-slot ownership update is needed.
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void Query(__gm__ Context* ctx, uint32_t history, uint32_t resident,
                  uint32_t tokenBytes, uint32_t batchCount, uint32_t b, uint32_t q,
                  uint32_t scratchBase, uint32_t tailTokens,
                  __gm__ int64_t* logicalRow, __ubuf__ uint32_t* selected,
                  __ubuf__ uint32_t* loc, __ubuf__ uint32_t* keys,
                  __ubuf__ uint32_t* ranks, __ubuf__ uint32_t* work,
                  __ubuf__ uint32_t* ctl) {
    const uint32_t tid = threadIdx.x, begin = q * kTop;
    const uint32_t owner = OwnerRow(ctx, b);
    const uint32_t ownerBase = owner * OutputStride(resident, ctx);
    const uint32_t queryEnd = QueryEnd(history, ctx, b * 4 + q);
    auto hot = reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid) + uint64_t(owner) * resident;
    uint32_t tailBegin = 0, tailEnd = 0;
    if (ctx->tailBounds) {
        auto bounds = reinterpret_cast<__gm__ uint32_t*>(ctx->tailBounds) + b * 2;
        tailBegin = asc_ldcg(bounds);
        tailEnd = asc_ldcg(bounds + 1);
    }
    for (uint32_t i = begin + tid; i < begin + kTop; i += 512) {
        uint32_t token = selected[i];
        loc[i] = UINT32_MAX;
        if (token == UINT32_MAX) continue;
        if (token >= history || token >= queryEnd) {
            asc_atomic_or(ctl + 1, 1U);
            continue;
        }
        // ABC has precedence even when its physical Index row is in the
        // resident prefix. Use the original ABC capacity, never scratch size.
        const bool inTail = token >= tailBegin && token < tailEnd;
        if (inTail && token - tailBegin < tailTokens) {
            loc[i] = resident + token % tailTokens;
            continue;
        }
        if (token < resident && !inTail && asc_ldcg(hot + token)) {
            loc[i] = token;
            continue;
        }
        uint32_t bucket = SparseHash(token) & (kBuckets - 1);
        bool inserted = false;
        for (uint32_t probe = 0; probe < kBuckets; ++probe) {
            uint32_t previous = asc_atomic_cas(keys + bucket, UINT32_MAX, token);
            if (previous == UINT32_MAX || previous == token) {
                if (!(ranks[bucket] & kStagedRankTag) || ranks[bucket] == UINT32_MAX)
                    asc_atomic_min(ranks + bucket, i);
                inserted = true;
                break;
            }
            bucket = (bucket + 1) & (kBuckets - 1);
        }
        if (!inserted) asc_atomic_or(ctl + 1, 4U);
    }
    asc_syncthreads();
    uint32_t localCount = 0;
    for (uint32_t i = begin + tid * 4; i < begin + (tid + 1) * 4; ++i) {
        uint32_t token = selected[i];
        if (token == UINT32_MAX || loc[i] != UINT32_MAX || token >= history || token >= queryEnd) continue;
        uint32_t bucket = Find(keys, token);
        if (bucket == UINT32_MAX) asc_atomic_or(ctl + 1, 4U);
        else localCount += ranks[bucket] == i;
    }
    uint32_t offset = StagedPrefix(localCount, ctl);
    uint32_t added = ctl[544 + 512 / warpSize - 1], oldCount = ctl[0];
    if (oldCount + added > kSlots) asc_atomic_or(ctl + 1, 4U);
    asc_syncthreads();
    if (ctl[1]) return;
    auto row = reinterpret_cast<__gm__ Row*>(ctx->rows) + b;
    auto addresses = reinterpret_cast<__gm__ uint64_t*>(row->addresses);
    auto blocks = reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks);
    auto offsets = reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets);
    auto destinations = reinterpret_cast<__gm__ int32_t*>(ctx->destinations);
    for (uint32_t i = begin + tid * 4; i < begin + (tid + 1) * 4; ++i) {
        uint32_t token = selected[i];
        if (token == UINT32_MAX || loc[i] != UINT32_MAX) continue;
        uint32_t bucket = Find(keys, token);
        if (ranks[bucket] != i) continue;
        uint32_t rank = oldCount + offset++;
        ranks[bucket] = kStagedRankTag | rank;
        uint32_t descriptor = b * kSlots + begin + rank - oldCount;
        StoreFallbackSource(history, tokenBytes, ctx, b, descriptor, token,
                            asc_ldcg(addresses + token), blocks, offsets);
        asc_stcg(destinations + descriptor, int32_t(ownerBase + scratchBase + rank));
    }
    asc_syncthreads();
    for (uint32_t i = begin + tid; i < begin + kTop; i += 512) {
        uint32_t token = selected[i];
        if (token != UINT32_MAX && loc[i] == UINT32_MAX)
            loc[i] = scratchBase + (ranks[Find(keys, token)] & ~kStagedRankTag);
    }
    if (!tid) {
        ctl[0] = oldCount + added;
        ctl[2 + q] = added;
        asc_atomic_add(reinterpret_cast<__gm__ uint32_t*>(ctx->count), added);
    }
    PrepareStagedQueryOutputs(logicalRow, ctl + kOutputMasksOffset + q * kOutputMasksPerQuery,
                             batchCount, b, q, selected, work, work + 64);
    asc_threadfence();
    asc_syncthreads();
}

__aicore__ inline void PublishFlag(__gm__ BatchFlag* flag, uint32_t count, uint32_t error) {
    flag->count = count;
    flag->error = error;
    flag->ready = 1;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flag), sizeof(*flag));
}

__aicore__ inline uint32_t Producer(__gm__ Args* a, uint32_t b) {
    auto ctx = reinterpret_cast<__gm__ Context*>(a->context);
    auto shape = LoadRuntimeShape(ctx);
    auto config = reinterpret_cast<__gm__ uint64_t*>(ctx->readThrough);
    auto row = reinterpret_cast<__gm__ Row*>(ctx->rows) + b;
    auto flags = reinterpret_cast<__gm__ BatchFlag*>(a->flags) + b * kFlagCount;
    auto status = reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus) + b * 32;
    auto update = reinterpret_cast<__gm__ uint64_t*>(a->updateBindings);
    auto ticks = reinterpret_cast<__gm__ uint64_t*>(a->times) + b * 64;
    const bool profile = (a->prefetchCount & 0x80000000U) != 0;
    if (profile) {
        // Clear the absent Consumer's old work, but preserve this invocation's
        // init record at Consumer[25:32]. Refresh its shared cache line before
        // scalar writes so stale data cannot overwrite the preceding kernel.
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ticks + 48), 128);
        for (uint32_t i = 0; i < 57; ++i) ticks[i] = 0;
        ticks[1] = GetSystemCycle(); ticks[9] = 2;
    }
    uint32_t error = config[0] != kMagic || config[1] < shape.resident ||
        config[2] != config[1] || config[3] != kSlots ||
        ctx->outputStride != config[1] + kSlots || row->slots || row->error ||
        !ctx->hotValid || !ctx->fallbackAddresses || !update[20] || !update[21] ||
        !update[22] || !update[23] || shape.finalWorkers != 4;
    uint32_t total = 0;
    // No consumer or output worker owns these notifications in this route.
    // Publish no mutable resident state; the finalizer still joins every Get.
    auto updateStatus = reinterpret_cast<__gm__ uint32_t*>(update[6]) + b;
    *updateStatus = error;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(updateStatus), sizeof(uint32_t));
    PublishFlag(flags + 6, 0, error);
    PublishFlag(flags + 7, 0, error);
    if (!error) {
        TPipe pipe;
        TBuf<TPosition::VECCALC> scratch;
        pipe.InitBuffer(scratch, kScratchBytes);
        auto local = scratch.Get<uint32_t>();
        auto selected = local, loc = local[kSlots], keys = local[kKeys], ranks = local[kRanks];
        auto work = local[kWork], ctl = local[kControl];
        GlobalTensor<uint32_t> input, locations;
        input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->selected) + b * kSlots);
        locations.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations) + b * kSlots);
        DataCopy(selected, input, kSlots);
        Simt::VF_CALL<Initialize>(Simt::Dim3{512, 1, 1}, FinalPtr(keys), FinalPtr(ranks), FinalPtr(ctl));
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Simt::VF_CALL<MergeQueryBitmaps>(Simt::Dim3{512, 1, 1}, FinalPtr(ctl), FinalPtr(keys), 0,
            reinterpret_cast<__gm__ int64_t*>(update[20]), reinterpret_cast<__gm__ int64_t*>(update[21]),
            shape.batch, b, FinalPtr(ctl[kOutputMasksOffset]), true);
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
        auto logical = reinterpret_cast<__gm__ int64_t*>(update[3]) +
            uint64_t(reinterpret_cast<__gm__ uint32_t*>(update[9])[b]) * uint32_t(update[24]);
        if (profile) ticks[3] = GetSystemCycle();
        for (uint32_t q = 0; q < 4; ++q) {
            if (profile) ticks[10 + q * 5] = GetSystemCycle();
            Simt::VF_CALL<Query>(Simt::Dim3{512, 1, 1}, ctx, shape.history, shape.resident,
                shape.tokenBytes, shape.batch, b, q, uint32_t(config[2]),
                uint32_t(config[1]) - shape.resident, logical, FinalPtr(selected), FinalPtr(loc),
                FinalPtr(keys), FinalPtr(ranks), FinalPtr(work), FinalPtr(ctl));
            SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
            error = ctl.GetValue(1);
            if (error) break;
            uint32_t added = ctl.GetValue(2 + q);
            total = ctl.GetValue(0);
            if (profile) ticks[11 + q * 5] = GetSystemCycle();
            SyncFetchDescriptors(ctx, b * kSlots + q * kTop, added);
            PrepareFinalFetch(shape.finalWorkers, shape.finalPostOffset, ctx, b, q,
                              added, b * kSlots + q * kTop, 0, added);
            PublishFlag(flags + 1 + q, added, 0);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(locations[q * kTop], loc[q * kTop], kTop);
            PublishQueryOutputs(update, shape.batch, b, q, selected, loc, work);
            if (profile) ticks[14 + q * 5] = GetSystemCycle();
        }
        pipe.Destroy();
    }
    for (uint32_t q = 0; q < 4; ++q) {
        if (!flags[q + 1].ready) {
            PrepareFinalFetch(shape.finalWorkers, shape.finalPostOffset, ctx, b, q,
                              0, b * kSlots + q * kTop, error ? error : 1, 0);
            PublishFlag(flags + q + 1, 0, error ? error : 1);
        }
    }
    *updateStatus = error;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(updateStatus), sizeof(uint32_t));
    status[0] = error; status[1] = total; status[2] = b * kSlots; status[3] = 0; status[4] = 0;
    row->finalBegin = 0; row->finalRequests = total; row->finalHead = 0;
    row->finalPending = total ? 1 : 0; row->finalEnd = GetSystemCycle();
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status), 128);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row) + 960, 64);
    PublishFlag(flags + kOutputsDoneFlag, total, error);
    if (profile) { ticks[8] = GetSystemCycle(); kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ticks), 64 * 8); }
    return 0;
}
}  // namespace read_through
