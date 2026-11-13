// SPDX-License-Identifier: Apache-2.0
// Native adapter for the bounded-ready scheduler. Included after PackFetchRanges.
#pragma once

#ifndef KVIP_READY_DIRECT_RANGES
#define KVIP_READY_DIRECT_RANGES 0
#endif
#if KVIP_READY_DIRECT_RANGES != 0 && KVIP_READY_DIRECT_RANGES != 1
#error "KVIP_READY_DIRECT_RANGES must be 0 or 1"
#endif

struct ReadyBatchIo {
    __gm__ Args* a;
    __gm__ Context* ctx;
    RuntimeShape shape;
    KvafExecuteBatchGetKernelArgs args;
    __gm__ int32_t* packed;
    __gm__ uint32_t* ranges;
    uint32_t query, lane, workers, stride;
    bool profile;
    uint32_t queryStarted = 0, queryReady = 0, prefetchStarted = 0;
    uint64_t begin = 0, firstNative = 0, lastNative = 0;
    uint64_t packCycles = 0, nativeCycles = 0, postCycles = 0, drainCycles = 0, groups = 0;

    __aicore__ inline uint64_t Invalid() const { return kvaf_device::kInvalidArgs; }
    __aicore__ inline uint64_t Timeout() const { return kvaf_device::kCqTimeout; }
    __aicore__ inline uint64_t Now() const { return GetSystemCycle(); }
    __aicore__ inline __gm__ uint64_t* Ticks(uint32_t b) const {
        return reinterpret_cast<__gm__ uint64_t*>(a->times) +
               uint64_t(shape.batch) * 128 + uint64_t(b) * 128 + query * 24;
    }
    __aicore__ inline ready_fetch::Publication Query(uint32_t b) {
        auto flag = reinterpret_cast<__gm__ BatchFlag*>(a->flags) + b * kFlagCount + query + 1;
        if (profile && !(queryStarted & (1U << b))) {
            Ticks(b)[0] = Now();
            queryStarted |= 1U << b;
        }
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flag), sizeof(*flag));
        if (!flag->ready) return {false, 0, 0, 0, 0};
        auto post = FinalPostAt(shape.finalPostOffset, ctx, b * shape.finalWorkers + query);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post), sizeof(*post));
        if (profile && !(queryReady & (1U << b))) {
            Ticks(b)[1] = Now();
            Ticks(b)[2] = Ticks(b)[1];
            queryReady |= 1U << b;
        }
        return {true, uint64_t(flag->error) | post->error,
                post->descriptor, post->count, post->nonconflicting};
    }
    __aicore__ inline ready_fetch::Completion Prefetch(uint32_t b) {
        auto flag = reinterpret_cast<__gm__ BatchFlag*>(a->flags) + b * kFlagCount + 7;
        if (profile && !(prefetchStarted & (1U << b))) {
            Ticks(b)[5] = Now();
            prefetchStarted |= 1U << b;
        }
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flag), sizeof(*flag));
        if (!flag->ready) return {false, 0};
        if (profile) Ticks(b)[6] = Now();
        return {true, flag->error};
    }
    __aicore__ inline ready_fetch::Submitted Submit(
        const ready_fetch::Range* batch, uint32_t count, uint32_t size) {
        uint64_t packStart = profile ? Now() : 0;
#if KVIP_READY_DIRECT_RANGES
        if (!batch || !count || count > shape.batch || !size || size > stride)
            return {Invalid(), 0};
        auto callArgs = args;
        if (count == 1) {
            const auto range = batch[0];
            const uint32_t capacity = shape.batch * ready_fetch::kSlots;
            if (range.output || range.count != size || range.begin >= capacity ||
                size > ready_fetch::kTop - range.begin % ready_fetch::kTop ||
                (range.begin % ready_fetch::kSlots) / ready_fetch::kTop != query ||
                !ctx->sourceBlocks || !ctx->sourceOffsets || !ctx->destinations)
                return {Invalid(), 0};
            // Query publication freezes this canonical range. Acquire its
            // descriptors before the synchronous native call; no producer may
            // reuse them until this kernel completes. Queue/CQ ownership and
            // the native validation/completion path are unchanged.
            SyncFetchDescriptors(ctx, range.begin, size);
            const uint64_t offset = uint64_t(range.begin) * sizeof(int32_t);
            callArgs.indices = ctx->sourceBlocks + offset;
            callArgs.offsets = ctx->sourceOffsets + offset;
            callArgs.output_indices = ctx->destinations + offset;
        } else
#endif
        {
            for (uint32_t i = 0; i < count; ++i) {
                SyncFetchDescriptors(ctx, batch[i].begin, batch[i].count);
                ranges[3 * i] = batch[i].begin;
                ranges[3 * i + 1] = batch[i].count;
                ranges[3 * i + 2] = batch[i].output;
            }
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ranges), uint64_t(count) * 12);
            Simt::VF_CALL<PackFetchRanges>(Simt::Dim3{256, 1, 1}, ctx, packed, stride, ranges, count);
            SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(packed), uint64_t(size) * 4);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(packed + stride), uint64_t(size) * 4);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(packed + 2 * stride), uint64_t(size) * 4);
        }
        if (profile) packCycles += Now() - packStart;
#if V9_SIMULATION
        return {Invalid(), 0};
#else
        auto queue = reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args.queue_state);
        auto status = reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(args.core_status);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue), 2 * sizeof(*queue));
        // The native status reports the queue's lifetime count, not this call's
        // count. Take a delta; summing lifetime counts over mini-batches is wrong.
        uint64_t completedBefore = queue[0].initialized ? queue[0].completed_batch_count : 0;
        uint64_t oldProfile = 0;
        if (profile) {
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status), sizeof(*status));
            oldProfile = status->profile.magic;
            status->profile.magic = KVAF_AIV_BATCH_GET_PROFILE_MAGIC;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(&status->profile), sizeof(status->profile));
        }
        uint64_t start = profile ? Now() : 0;
#if KVIP_READY_DIRECT_RANGES
        kvaf_device::KvafBatchGetDevice(callArgs, 0, 0, size);
#else
        kvaf_device::KvafBatchGetDevice(args, 0, 0, size);
#endif
        uint64_t end = profile ? Now() : 0;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue), 2 * sizeof(*queue));
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status), sizeof(*status));
        if (profile) {
            if (!firstNative) firstNative = start;
            lastNative = end;
            nativeCycles += end - start;
            postCycles += status->profile.post_active_cycles;
            drainCycles += status->profile.drain_wait_cycles;
            groups += status->profile.submitted_group_count;
            status->profile.magic = oldProfile;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(&status->profile), sizeof(status->profile));
        }
        uint64_t error = status->result;
        uint64_t completedAfter = queue[0].completed_batch_count;
        uint32_t completed = 0;
        if (status->magic != KVAF_AIV_BATCH_GET_STATUS_MAGIC || status->processed_request_count != size ||
            completedAfter < completedBefore || completedAfter - completedBefore > UINT32_MAX)
            error |= Invalid();
        else completed = uint32_t(completedAfter - completedBefore);
        // On native failure its poison and CQ state remain intact. The caller
        // stops submissions and publishes error; no successful hot residency.
        return {error, completed};
#endif
    }
    __aicore__ inline void Finish(const ready_fetch::Result& result) {
        if (!profile) return;
        uint64_t finish = Now();
        uint64_t prepare = finish - begin - packCycles - nativeCycles;
        for (uint32_t b = lane; b < shape.batch; b += workers) {
            auto ft = Ticks(b);
            ft[3] = firstNative; ft[4] = lastNative; ft[7] = 0; ft[8] = 0;
            ft[12] = postCycles; ft[13] = drainCycles; ft[14] = 0; ft[15] = 0;
            ft[16] = groups; ft[17] = 0; ft[18] = prepare; ft[19] = packCycles;
            ft[20] = 4;  // Bounded-ready worker: fields are shared, not per row.
            ft[21] = nativeCycles; ft[22] = result.requests; ft[23] = result.calls;
        }
        auto trace = reinterpret_cast<__gm__ uint64_t*>(a->times) + uint64_t(shape.batch) * 256 +
                     uint64_t(kResolveCores * shape.batch + query * workers + lane) * 32;
        trace[2] = begin; trace[3] = firstNative; trace[4] = lastNative; trace[5] = finish;
        trace[6] = nativeCycles; trace[7] = packCycles; trace[8] = prepare;
        trace[9] = result.calls; trace[10] = result.requests; trace[11] = groups;
    }
};
