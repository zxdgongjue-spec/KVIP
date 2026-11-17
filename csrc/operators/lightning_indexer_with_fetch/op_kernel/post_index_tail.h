// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "post_index_transport.h"
#include "post_index_update.h"

namespace pipeline_ab {
namespace post_index_tail {

// One cache line per AIV. Only that AIV writes its record, except init's
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

__aicore__ inline uint64_t RowError(__gm__ Record* records, uint32_t b,
                                   uint32_t batch, uint32_t cores) {
    uint64_t error = 0;
    for (uint32_t i = b; i < cores; i += batch) {
        Flush(records + i);
        error |= records[i].error;
    }
    return error;
}

// Validate and pack one balanced slice of the request's Q4 union. The same
// event slice supplies Index exchanges and Get descriptors, so both checks
// share one VF and finish before the collective pre-Post error barrier.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void Pack(__gm__ Context* ctx, uint32_t b, uint32_t begin, uint32_t count,
                 uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
                 __gm__ uint32_t* eventSources, __gm__ uint32_t* eventDestinations,
                 uint32_t eventOffset, uint32_t resident, uint32_t history,
                 __gm__ int32_t* packed, uint32_t tableCapacity,
                 uint32_t outputRows, uint32_t tokenBytes, uint64_t remoteBase,
                 uint64_t remoteBytes, uint64_t localBytes, __ubuf__ uint32_t* error) {
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
        int32_t block = asc_ldcg(blocks + source), offset = asc_ldcg(offsets + source);
        int32_t destination = asc_ldcg(destinations + source);
        // Fallback rows are frozen TOKEN addresses, not 128-token block bases.
        // Validate while these descriptors are already in registers. The
        // scalar caller reads one error word, not a 1024-entry reduction.
        if (block < 0 || uint32_t(block) >= tableCapacity || offset != 0 ||
            !asc_ldcg(valid + block)) {
            asc_atomic_or(error, uint32_t(kvaf_device::kInvalidLookup)); continue;
        }
        if (destination < 0 || uint32_t(destination) >= outputRows || !tokenBytes ||
            localBytes < tokenBytes || uint64_t(destination) > (localBytes - tokenBytes) / tokenBytes) {
            asc_atomic_or(error, uint32_t(kvaf_device::kInvalidOutput)); continue;
        }
        uint64_t address = asc_ldcg(addresses + block);
        if (remoteBytes < tokenBytes || address < remoteBase || address - remoteBase > remoteBytes - tokenBytes) {
            asc_atomic_or(error, uint32_t(kvaf_device::kInvalidLookup)); continue;
        }
        asc_stcg(packed + begin + i, block);
        asc_stcg(packed + kSlots + begin + i, offset);
        asc_stcg(packed + 2 * kSlots + begin + i, destination);
    }
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
    // 3*batch*8192 hold Get descriptors; only batch+1 more words are needed.
    auto prefix = reinterpret_cast<__gm__ uint32_t*>(a->packedFetch) +
                  uint64_t(batch) * 3 * kSlots;
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
    } else if (block < 2 * batch) {
        OutputWorker(a, block - batch);
    }
    // TND outputs use the old logical table. Every reader finishes here,
    // before any Index tuple or logical/address mapping can be exchanged.
    SyncAll();
    if (profile) {
        record->planDone = GetSystemCycle();
        // Old OutputWorker diagnostic rows share this scratch allocation.
        // Initialize Index markers after every old producer/worker writer joined.
        for (uint32_t i = 2; i <= 10; ++i) trace[i] = 0;
        trace[2] = 0x50495832U;  // PIX2: production VF, event counts read offline
    }

    auto info = reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus) + b * kRowStatusWords;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(info), 128);
    uint32_t counts[4] = {}, total = 0;
    uint64_t error = info[0];
    auto row = reinterpret_cast<__gm__ Row*>(ctx->rows) + b;
    // Init/LI can reject queue or geometry inputs independently of the
    // resolver. Join that error before any Post, and never run a disjoint
    // zero-prefetch plan over an already-prefetched row.
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(&row->slots), 4 * sizeof(uint32_t));
    error |= row->error;
    if (row->slots) error |= kvaf_device::kInvalidArgs;
    for (uint32_t q = 0; q < 4; ++q) {
        auto post = FinalPostAt(shape.finalPostOffset, ctx, b * 4 + q);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post), sizeof(*post));
        error |= post->error;
        if (post->count > kTop || post->descriptor != b * kSlots + q * kTop) error |= 1;
        counts[q] = post->count <= kTop ? post->count : 0;
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
    if (!error && end > begin) {
        TPipe pipe; TBuf<TPosition::VECCALC> scratch;
        pipe.InitBuffer(scratch, 64); auto local = scratch.Get<uint32_t>();
        auto entity = reinterpret_cast<__gm__ const KvafBatchGetDirectUrmaEntity*>(args.channel_entity);
        Simt::VF_CALL<Pack>(Simt::Dim3{256,1,1}, ctx, b, begin, end - begin,
            counts[0], counts[1], counts[2], counts[3], sources, destinations,
            kPrefetch, shape.resident, shape.history, packed,
            args.table_capacity, args.output_rows, args.token_bytes,
            entity->remote_host_ddr_address, entity->remote_host_ddr_size,
            entity->local_hbm_size,
            reinterpret_cast<__ubuf__ uint32_t*>(local.GetPhyAddr()));
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
        error |= local.GetValue(0);
        pipe.Destroy();
    }
    record->error = error; record->fallback = blocking;
    if (profile) record->packDone = GetSystemCycle();
    Flush(record);
    SyncAll();
    if (!block) {
        uint64_t anyError = 0, anyBlocking = 0;
        for (uint32_t i = 0; i <= batch; ++i) prefix[i] = 0;
        for (uint32_t i = 0; i < cores; ++i) {
            Flush(records + i);
            anyError |= records[i].error; anyBlocking |= records[i].fallback;
            prefix[i % batch + 1] += uint32_t(records[i].count);
        }
        for (uint32_t i = 1; i <= batch; ++i) prefix[i] += prefix[i - 1];
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(prefix), (batch + 1) * sizeof(uint32_t));
        record->globalError = anyError; record->fallback = anyBlocking;
        Flush(record);
    }
    SyncAll();
    Flush(records);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(prefix), (batch + 1) * sizeof(uint32_t));
    const uint32_t indexFirst = uint64_t(prefix[batch]) * block / cores;
    const uint32_t indexEnd = uint64_t(prefix[batch]) * (block + 1) / cores;
    error |= records[0].globalError;
    blocking = (records[0].fallback & 1U) != 0;
    record->fallback = uint64_t(blocking) | (waitFirst ? 2U : 0U);
    record->globalError = records[0].globalError;
    post_index_transport::State transport;
    if (profile) record->postBegin = GetSystemCycle();
    if (!error) error = blocking ?
        post_index_transport::GetBlocking(args, 0, end - begin, transport) :
        post_index_transport::Post(args, 0, end - begin, transport);
    if (profile) {
        record->postDone = GetSystemCycle();
        if (blocking) record->cqDone = record->postDone;
    }
    record->error = error; Flush(record);
    // No AIV enters Index maintenance before all Post calls have returned.
    SyncAll();
    if (profile) trace[3] = GetSystemCycle();
    if (waitFirst) {
        error |= post_index_transport::Wait(transport);
        if (profile) trace[7] = record->cqDone = GetSystemCycle();
        record->error = error; Flush(record);
        // This is the same join used after CQ in the overlapping order,
        // relocated before Index so communication is fully drained first.
        SyncAll();
    }
    if (profile) trace[4] = GetSystemCycle();
    if (!records[0].globalError && indexEnd > indexFirst) {
        auto update = reinterpret_cast<__gm__ uint64_t*>(a->updateBindings);
        // Owner loads are SIMT-only. Pass base arrays and let each VF thread
        // find the request owning its global event, without scalar OwnerRow.
        // Timing diagnostics use the production VF too: no diagnostic TPipe,
        // UB atomics or tuple counter inside the measured Index interval.
        Simt::VF_CALL<post_index::ApplyPlan<false>>(Simt::Dim3{post_index::kThreads,1,1},
            reinterpret_cast<__gm__ uint32_t*>(a->eventsSrc),
            reinterpret_cast<__gm__ uint32_t*>(a->eventsDst),
            kPrefetch, a->maxEvents, shape.history, prefix, batch, indexFirst, indexEnd,
            reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners),
            reinterpret_cast<__gm__ uint64_t*>(records), cores,
            reinterpret_cast<__gm__ uint64_t*>(a->keys),
            reinterpret_cast<__gm__ uint32_t*>(a->scales),
            reinterpret_cast<__gm__ uint64_t*>(a->addresses),
            reinterpret_cast<__gm__ uint64_t*>(a->logical), uint32_t(update[7]),
            static_cast<__ubuf__ uint32_t*>(nullptr));
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
    if (profile) {
        trace[9] = indexEnd - indexFirst;
        trace[5] = record->indexDone = GetSystemCycle();
    }
    record->completed = transport.completed; record->error = error; Flush(record);
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
        error = RowError(records, row, batch, cores);
        uint32_t completed = 0;
        for (uint32_t i = row; i < cores; i += batch) completed += records[i].completed;
        auto rowInfo = reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus) + row * kRowStatusWords;
        if (!error && completed != rowInfo[1]) error = kvaf_device::kInvalidArgs;
        auto update = reinterpret_cast<__gm__ uint64_t*>(a->updateBindings);
        Simt::VF_CALL<PublishStreamingUpdateStatus>(Simt::Dim3{32,1,1},
            reinterpret_cast<__gm__ uint32_t*>(update[6]), row, uint32_t(error));
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (uint32_t q = 0; q < 4; ++q) {
            auto post = FinalPostAt(shape.finalPostOffset, ctx, row * 4 + q);
            post->pending = error ? 0 : post->count;
            post->deferred = 0; post->error = uint32_t(error);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post), sizeof(*post));
        }
        if (!error && (a->prefetchCount & 0x10000000U)) {
            Simt::VF_CALL<PublishStreamingResidency>(Simt::Dim3{256,1,1},
                shape.resident, shape.finalPostOffset, ctx, row);
            SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
        }
        // The original finalizer publishes residency only on success,
        // invalidates the whole owner on failure and commits success feedback.
        FinalizeStreaming<false, true>(a, row, 0);
    }
}
}  // namespace post_index_tail
}  // namespace pipeline_ab
