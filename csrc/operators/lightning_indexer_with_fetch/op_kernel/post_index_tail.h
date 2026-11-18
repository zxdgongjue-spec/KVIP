// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "post_index_transport.h"
#include "post_index_update.h"
#include "post_index_prepare.h"
#include "low_q4_outputs_mte.h"
#include "post_index_finalize.h"

namespace pipeline_ab {
namespace post_index_tail {

static_assert(post_index_prepare::kPostFailureWord == post_index::kFailureMaskWord,
              "Prepare and Index must share the failure mask");

// One 128B record per AIV. Only that AIV writes its record, except init's
// batch-wide decision in record zero, read after the preceding kernel edge.
struct alignas(128) Record {
    uint64_t error, count, fallback, globalError;
    uint64_t planDone, packDone, postBegin, postDone, indexDone, cqDone;
    uint64_t start, packedRow, lane, lanes, active, completed;
};
static_assert(sizeof(Record) == 128, "Post/Index/CQ diagnostic ABI");

__aicore__ inline void Flush(__gm__ Record* row) {
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row), sizeof(Record));
}

// Rare failure publication replaces repeated per-event sibling-record loads.
// The caller retains the existing cross-core join after this VF completes.
__aicore__ inline void PublishPostFailure(__gm__ uint32_t* prefix, uint32_t row,
                                        uint64_t error) {
    if (!error) return;
    Simt::VF_CALL<post_index::MarkPostFailure>(Simt::Dim3{32,1,1}, prefix, row, error);
    SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
}

// Validate and pack one balanced slice of the request's Q4 union. The same
// event slice supplies Index exchanges and Get descriptors, so both checks
// share one VF and finish before the collective pre-Post error barrier.
template<bool BuildPrefix = false>
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void Pack(__gm__ Context* ctx, __gm__ Args* args,
                 uint32_t block, uint32_t cores, uint32_t b, uint32_t begin, uint32_t count,
                 uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
                 __gm__ uint32_t* eventSources, __gm__ uint32_t* eventDestinations,
                 uint32_t eventOffset, uint32_t resident, uint32_t history,
                 __gm__ int32_t* packed, uint32_t tableCapacity,
                 uint32_t outputRows, uint32_t tokenBytes, uint64_t remoteBase,
                 uint64_t remoteBytes, uint64_t localBytes, __ubuf__ uint32_t* error) {
    if constexpr (BuildPrefix) {
        // Only block0 calls this specialization, so packed is the base table.
        // Publish prefix even if the following ordinary Pack validation fails.
        const uint32_t batch = ctx->shape.batch;
        auto prefix = reinterpret_cast<__gm__ uint32_t*>(packed) + uint64_t(batch) * 3 * kSlots;
        post_index_prepare::PrefixScan(
            reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus), kRowStatusWords, batch, prefix);
    }
    const uint64_t total = uint64_t(c0) + c1 + c2 + c3;
    if (!threadIdx.x)
        error[0] = c0 > kTop || c1 > kTop || c2 > kTop || c3 > kTop ||
                   total > kSlots || resident > history || begin > total ||
                   count > total - begin ? 1U : 0U;
    asc_syncthreads();
    if (error[0]) return;
    auto blocks = reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks);
    auto offsets = reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets);
    auto destinations = reinterpret_cast<__gm__ int32_t*>(ctx->destinations);
    auto addresses = reinterpret_cast<__gm__ uint64_t*>(ctx->fallbackAddresses);
    auto valid = reinterpret_cast<__gm__ uint8_t*>(ctx->fallbackValid);
    auto row = reinterpret_cast<__gm__ Row*>(ctx->rows) + b;
    auto tokenAddresses = reinterpret_cast<__gm__ uint64_t*>(row->addresses);
    const uint32_t owner = ctx->rowOwners ?
        asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners) + b) : b;
    const uint64_t ownerBase = uint64_t(owner) * (ctx->outputStride ? ctx->outputStride : resident);
    for (uint32_t i = threadIdx.x; i < count; i += 256) {
        const uint32_t event = eventOffset + begin + i;
        uint32_t src = asc_ldcg(eventSources + event);
        uint32_t dst = asc_ldcg(eventDestinations + event);
        if (src >= history || dst >= resident || (src != dst && src < resident)) {
            asc_atomic_or(error, 1U); continue;
        }
        uint32_t index = begin + i, q = 0;
        if (index >= c0) { index -= c0; q = 1;
            if (index >= c1) { index -= c1; q = 2;
                if (index >= c2) { index -= c2; q = 3; }
            }
        }
        const uint32_t source = b * kSlots + q * kTop + index;
        // Freeze original addresses directly from the exact event plan. No
        // intermediate canonical descriptor table is loaded and repacked.
        if (source >= tableCapacity || !tokenAddresses || !addresses || !valid ||
            !asc_ldcg(valid + source)) {
            asc_atomic_or(error, uint32_t(kvaf_device::kInvalidLookup)); continue;
        }
        const uint64_t destination = ownerBase + dst;
        if (destination >= outputRows || destination > INT32_MAX || !tokenBytes ||
            localBytes < tokenBytes || uint64_t(destination) > (localBytes - tokenBytes) / tokenBytes) {
            asc_atomic_or(error, uint32_t(kvaf_device::kInvalidOutput)); continue;
        }
        uint64_t address = asc_ldcg(tokenAddresses + src);
        if (remoteBytes < tokenBytes || address < remoteBase || address - remoteBase > remoteBytes - tokenBytes) {
            asc_atomic_or(error, uint32_t(kvaf_device::kInvalidLookup)); continue;
        }
        asc_stcg(addresses + source, address);
        // Retain descriptor visibility for diagnostics and residency. Both
        // representations are generated from registers in this same pass.
        asc_stcg(blocks + source, int32_t(source));
        asc_stcg(offsets + source, int32_t(0));
        asc_stcg(destinations + source, int32_t(destination));
        asc_stcg(packed + begin + i, int32_t(source));
        asc_stcg(packed + kSlots + begin + i, int32_t(0));
        asc_stcg(packed + 2 * kSlots + begin + i, int32_t(destination));
    }
    asc_syncthreads();
    if (!threadIdx.x && count && !error[0])
        asc_atomic_add(reinterpret_cast<__gm__ uint32_t*>(ctx->count), count);
    asc_threadfence();
}

// No wait/Index worker remains resident alongside Post. The same AIV and
// exclusive queue are reused across all three phases. Keep this function out
// of the high-route body so its large scalar/UB lifetimes do not join there.
__aicore__ __attribute__((noinline)) inline void Run(__gm__ Args* a, uint32_t block) {
    auto ctx = reinterpret_cast<__gm__ Context*>(a->context);
    const auto shape = LoadRuntimeShape(ctx);
    const uint32_t cores = 2 * shape.aicCount, batch = shape.batch;
    const uint32_t b = block % batch, lane = block / batch;
    const uint32_t lanes = (cores - 1 - b) / batch + 1;
    auto records = reinterpret_cast<__gm__ Record*>(a->postIndex);
    auto record = records + block;
    const bool profile = (a->prefetchCount & 0x80000000U) != 0;
    // Offline overlap control, fixed before graph capture. Normal service
    // keeps this false; both orders maintain exactly the same Index state.
    const bool waitFirst = (a->prefetchCount & 0x02000000U) != 0;
    auto trace = reinterpret_cast<__gm__ uint64_t*>(a->times) +
                 uint64_t(batch) * 256 + uint64_t(block) * 32;
    // Existing packed workspace has at least 9*batch*8192 words. The first
    // 3*batch*8192 hold Get descriptors. The next 128 bytes hold the
    // batch+1 prefix words and shared error/fallback control at words 24..31.
    auto prefix = reinterpret_cast<__gm__ uint32_t*>(a->packedFetch) +
                  uint64_t(batch) * 3 * kSlots;
    if (!block) {
        // The exact producer does not write packedFetch, so these two 64B
        // lines are free before the first join. Later prefix writers are SIMT-only: a scalar
        // dirty-line writeback could overwrite another AIV's error atomic.
        prefix[post_index_prepare::kErrorWord] = 0;
        prefix[post_index_prepare::kErrorWord + 1] = 0;
        prefix[post_index_prepare::kBlockingWord] = 0;
        prefix[post_index::kFailureMaskWord] = 0;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(prefix), 128);
    }
    // Preserve init's decision; timestamps are zero in headline samples.
    auto words = reinterpret_cast<__gm__ uint64_t*>(record);
    // Peers can still be reading the entry decision. Never transiently clear
    // record zero's active word or some AIVs could miss the collective path.
    for (uint32_t i = 0; i < 16; ++i) if (i != 14) words[i] = 0;
    record->packedRow = b; record->lane = lane; record->lanes = lanes;
    if (profile) record->start = GetSystemCycle();

    if (block < batch) {
        // Old consumer diagnostic rows are inapplicable; preserve init fields.
        if (profile) {
            auto t = reinterpret_cast<__gm__ uint64_t*>(a->times) + block * 64 + 32;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(t), 256);
            for (uint32_t i = 0; i < 25; ++i) t[i] = 0;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(t), 25 * 8);
        }
        Producer<true>(a, block);
    }
    // Publish all Q4 plans. Parallel output/descriptor construction below
    // still sees the original logical/address tables until the next join.
    SyncAll();
    if (profile) {
        record->planDone = GetSystemCycle();
        // Initialize Index markers after all request plan writers have joined.
        for (uint32_t i = 2; i <= 10; ++i) trace[i] = 0;
        trace[2] = 0x50495832U;  // PIX2: production VF, event counts read offline
    }

    auto info = reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus) + b * kRowStatusWords;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(info), 128);
    uint32_t counts[4] = {}, total = 0, publishedTotal = 0;
    uint64_t error = info[0];
    auto row = reinterpret_cast<__gm__ Row*>(ctx->rows) + b;
    // Init/LI can reject queue or geometry inputs independently of the
    // resolver. Join that error before any Post, and never run a disjoint
    // zero-prefetch plan over an already-prefetched row.
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(&row->slots), 4 * sizeof(uint32_t));
    error |= row->error;
    if (row->slots) error |= kvaf_device::kInvalidArgs;
    for (uint32_t q = 0; q < 4; ++q) {
        // Producer publishes these counts and its error in this same 128B
        // row before the collective join. FinalPost is an output ABI, not a
        // second plan: avoid four dependent cache-line acquisitions per AIV.
        const uint32_t count = info[5 + q];
        if (count > kTop) error |= 1;
        publishedTotal += count;
        counts[q] = count <= kTop ? count : 0;
        total += counts[q];
    }
    if (total != info[1] || total > kSlots || shape.resident > shape.history ||
        !lanes || lane >= lanes) error |= 1;
    const uint32_t begin = lanes ? uint64_t(total) * lane / lanes : 0;
    const uint32_t end = lanes ? uint64_t(total) * (lane + 1) / lanes : 0;
    record->count = end - begin;
    auto packed = reinterpret_cast<__gm__ int32_t*>(a->packedFetch) + uint64_t(b) * 3 * kSlots;
    auto sources = reinterpret_cast<__gm__ uint32_t*>(a->eventsSrc) + b * a->maxEvents;
    auto destinations = reinterpret_cast<__gm__ uint32_t*>(a->eventsDst) + b * a->maxEvents;
    auto args = GetArgs(ctx, block);
    args.block_count = 1; args.jettys_per_core = 1;
    args.indices = uint64_t(packed + begin);
    args.offsets = uint64_t(packed + kSlots + begin);
    args.output_indices = uint64_t(packed + 2 * kSlots + begin);
    args.request_capacity = end - begin;
    args.address_table = ctx->fallbackAddresses;
    args.valid_table = ctx->fallbackValid;
    args.table_capacity = batch * kSlots;
    args.tokens_per_block = 1;
    bool blocking = false;
    if (!error) error = post_index_transport::Preflight(args, 0, end - begin, blocking);
    if (!error) {
        TPipe pipe; TBuf<TPosition::VECCALC> scratch;
        pipe.InitBuffer(scratch, 64); auto local = scratch.Get<uint32_t>();
        // Empty slices still own TND output elements. They never dereference
        // an entity that zero-count native preflight did not validate.
        auto entity = end > begin ?
            reinterpret_cast<__gm__ const KvafBatchGetDirectUrmaEntity*>(args.channel_entity) : nullptr;
        const uint64_t remoteBase = entity ? entity->remote_host_ddr_address : 0;
        const uint64_t remoteBytes = entity ? entity->remote_host_ddr_size : 0;
        const uint64_t localBytes = entity ? entity->local_hbm_size : 0;
        if (!block) {
            Simt::VF_CALL<Pack<true>>(Simt::Dim3{256,1,1}, ctx, a, block, cores, b, begin, end - begin,
                counts[0], counts[1], counts[2], counts[3], sources, destinations,
                kPrefetch, shape.resident, shape.history, packed,
                args.table_capacity, args.output_rows, args.token_bytes,
                remoteBase, remoteBytes, localBytes,
                reinterpret_cast<__ubuf__ uint32_t*>(local.GetPhyAddr()));
        } else {
            Simt::VF_CALL<Pack<false>>(Simt::Dim3{256,1,1}, ctx, a, block, cores, b, begin, end - begin,
                counts[0], counts[1], counts[2], counts[3], sources, destinations,
                kPrefetch, shape.resident, shape.history, packed,
                args.table_capacity, args.output_rows, args.token_bytes,
                remoteBase, remoteBytes, localBytes,
                reinterpret_cast<__ubuf__ uint32_t*>(local.GetPhyAddr()));
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
        error |= local.GetValue(0);
        pipe.Destroy();
    } else if (!block) {
        // Empty or rejected block0 still publishes a safe prefix. This path
        // must not dereference a channel entity that preflight did not accept.
        Simt::VF_CALL<post_index_prepare::PrepareOnly>(Simt::Dim3{32,1,1}, ctx, prefix);
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
    if (!error) {
        for (uint32_t outputRow = block; outputRow < batch * 4; outputRow += cores)
            error |= low_q4_outputs_mte::Publish(ctx, a, outputRow);
    }
    if (error || blocking) {
        Simt::VF_CALL<post_index_prepare::PublishPreflight>(
            Simt::Dim3{32,1,1}, prefix, error, uint32_t(blocking));
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
    if (profile) record->packDone = GetSystemCycle();
    // Prefix and every exceptional error/capacity writer finish at this
    // existing join. There is no scalar 64-record reduction or second join.
    SyncAll();
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(prefix), 128);
    const uint64_t globalError =
        *reinterpret_cast<__gm__ uint64_t*>(prefix + post_index_prepare::kErrorWord);
    const uint32_t indexFirst = uint64_t(prefix[batch]) * block / cores;
    const uint32_t indexEnd = uint64_t(prefix[batch]) * (block + 1) / cores;
    error |= globalError;
    blocking = (prefix[post_index_prepare::kBlockingWord] & 1U) != 0;
    record->fallback = uint64_t(blocking) | (waitFirst ? 2U : 0U);
    record->globalError = globalError;
    post_index_transport::State transport;
    if (profile) record->postBegin = GetSystemCycle();
    if (!error) error = blocking ?
        post_index_transport::GetBlocking(args, 0, end - begin, transport) :
        post_index_transport::Post(args, 0, end - begin, transport);
    if (profile) {
        record->postDone = GetSystemCycle();
        if (blocking) record->cqDone = record->postDone;
    }
    PublishPostFailure(prefix, b, error);
    // No AIV enters Index maintenance before all Post calls have returned.
    SyncAll();
    if (profile) trace[3] = GetSystemCycle();
    if (waitFirst) {
        error |= post_index_transport::Wait(transport);
        if (profile) trace[7] = record->cqDone = GetSystemCycle();
        PublishPostFailure(prefix, b, error);
        // This is the same join used after CQ in the overlapping order,
        // relocated before Index so communication is fully drained first.
        SyncAll();
    }
    if (profile) trace[4] = GetSystemCycle();
    if (!globalError && indexEnd > indexFirst) {
        auto update = reinterpret_cast<__gm__ uint64_t*>(a->updateBindings);
        // Keep global event balancing, but preload the small lookup tables.
        // Copy/wait/destruction stay within the Index timing interval.
        TPipe indexPipe;
        TBuf<TPosition::VECCALC> prefixBuffer, ownerBuffer;
        indexPipe.InitBuffer(prefixBuffer, 128);
        auto prefixLocal = prefixBuffer.Get<uint32_t>();
        GlobalTensor<uint32_t> prefixGlobal;
        prefixGlobal.SetGlobalBuffer(prefix);
        DataCopyPad(prefixLocal, prefixGlobal, DataCopyExtParams{1,128,0,0,0},
                    DataCopyPadExtParams<uint32_t>{false,0,0,0});
        __ubuf__ uint32_t* indexOwners = nullptr;
        if (ctx->rowOwners) {
            indexPipe.InitBuffer(ownerBuffer, (batch * 4 + 31) / 32 * 32);
            auto ownerLocal = ownerBuffer.Get<uint32_t>();
            GlobalTensor<uint32_t> ownerGlobal;
            ownerGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners));
            DataCopyPad(ownerLocal, ownerGlobal, DataCopyExtParams{1,batch*4,0,0,0},
                        DataCopyPadExtParams<uint32_t>{false,0,0,0});
            indexOwners = reinterpret_cast<__ubuf__ uint32_t*>(ownerLocal.GetPhyAddr());
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Simt::VF_CALL<post_index::ApplyPlan<false>>(Simt::Dim3{post_index::kThreads,1,1},
            reinterpret_cast<__gm__ uint32_t*>(a->eventsSrc),
            reinterpret_cast<__gm__ uint32_t*>(a->eventsDst),
            kPrefetch, a->maxEvents, shape.history,
            reinterpret_cast<__ubuf__ uint32_t*>(prefixLocal.GetPhyAddr()),
            batch, indexFirst, indexEnd, indexOwners,
            reinterpret_cast<__gm__ uint64_t*>(a->keys),
            reinterpret_cast<__gm__ uint32_t*>(a->scales),
            reinterpret_cast<__gm__ uint64_t*>(a->addresses),
            reinterpret_cast<__gm__ uint64_t*>(a->logical), uint32_t(update[7]),
            static_cast<__ubuf__ uint32_t*>(nullptr));
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
        indexPipe.Destroy();
    }
    if (profile) {
        trace[9] = indexEnd - indexFirst;
        trace[5] = record->indexDone = GetSystemCycle();
    }
    // Peers use the failure mask up to the final join. Publish this worker's
    // record once, at the join immediately preceding the per-row reduction.
    if (waitFirst) {
        record->completed = transport.completed; record->error = error;
        Flush(record);
    }
    SyncAll();
    if (profile) trace[6] = GetSystemCycle();
    // A peer/Index failure must never strand a successfully posted queue.
    if (!waitFirst) {
        error |= post_index_transport::Wait(transport);
        if (profile) {
            trace[7] = GetSystemCycle();
            if (!blocking) record->cqDone = trace[7];
        }
        record->completed = transport.completed; record->error = error;
        Flush(record);
        SyncAll();
    }
    if (block < batch) {
        const uint32_t row = block;
        error = 0;
        uint32_t completed = 0;
        for (uint32_t peer = row; peer < cores; peer += batch) {
            Flush(records + peer);
            error |= records[peer].error;
            completed += uint32_t(records[peer].completed);
        }
        auto rowInfo = reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus) + row * kRowStatusWords;
        if (!error && completed != rowInfo[1]) error = kvaf_device::kInvalidArgs;
        post_index_finalize::Finish(a, row, publishedTotal, error);
    }
}
}  // namespace post_index_tail
}  // namespace pipeline_ab
