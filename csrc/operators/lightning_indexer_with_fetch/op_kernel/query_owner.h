// SPDX-License-Identifier: Apache-2.0
// Four resident AIVs per request. A query owns its descriptor slice and its SQ/CQ.
// Prefix offsets preserve first-occurrence slot order. Overlapping Index swaps retain publication order.
#pragma once
#include "query_owner_layout.h"
namespace query_owner {
static_assert(sizeof(KvafBatchGetCoreStatus) <= kOwnerStatusWords * 4, "Get status storage");
__aicore__ inline void FromGm(LocalTensor<uint32_t> to, __gm__ uint32_t *from, uint32_t words) {
    if (words) {
        GlobalTensor<uint32_t> src;
        src.SetGlobalBuffer(from);
        DataCopy(to, src, (words + 7) & ~7U);
    }
}
__aicore__ inline void ToGm(__gm__ uint32_t *to, LocalTensor<uint32_t> from, uint32_t words) {
    if (words) {
        GlobalTensor<uint32_t> dst;
        dst.SetGlobalBuffer(to);
        DataCopy(dst, from, (words + 7) & ~7U);
    }
}
__aicore__ inline void InputReady() {
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
}
__aicore__ inline void VectorDone() {
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
}
__aicore__ inline void BeforeStore() {
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
}
__aicore__ inline void StoreDone() {
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}
__aicore__ inline void Signal(__gm__ BatchFlag *f, uint32_t count, uint32_t error) {
    index_team::Publish(f, count, error);
}
struct Slice {
    uint32_t begin = 0, count = 0, safe = 0, error = 0;
    bool disjoint = false, escape = false;
};
__aicore__ inline __gm__ uint32_t *Scratch(__gm__ Args *a, uint32_t b) {
    return reinterpret_cast<__gm__ uint32_t *>(a->queryOwnerScratch) + b * kQueryOwnerRowWords;
}

#include "offset_resolver.h"

// Slot assignment becomes final in Resolve. Logical IDs were already emitted by LI.
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void AdaptSlots(__gm__ Args *a, uint32_t b, uint32_t q) {
    auto ctx = reinterpret_cast<__gm__ Context *>(a->context);
    auto u = reinterpret_cast<__gm__ uint64_t *>(a->updateBindings);
    if (!u[20] || !u[21] || !u[22] || !u[23]) {
        return;
    }
    auto packed = reinterpret_cast<__gm__ int64_t *>(u[20]), valid = reinterpret_cast<__gm__ int64_t *>(u[21]);
    auto loc = reinterpret_cast<__gm__ uint32_t *>(ctx->locations) + (4 * b + q) * kTop;
    // One coalesced ballot per 32 output rows. Support repeated unpack
    // indices and inactive rows with the same semantics as the ordinary output worker.
    for (uint32_t base = 0; base < ctx->shape.batch * 4; base += 32) {
        const uint32_t target = base + threadIdx.x % 32;
        const bool active = target < ctx->shape.batch * 4 && asc_ldcg(valid + target) != 0;
        uint32_t targets = asc_ballot(active ? asc_ldcg(packed + target) == 4 * b + q : target == 4 * b + q);
        uint32_t activeMask = asc_ballot(active);
        while (targets) {
            const uint32_t bit = __builtin_ctz(targets);
            const uint32_t offset = (base + bit) * kTop;
            auto slot = reinterpret_cast<__gm__ uint32_t *>(u[23]) + offset;
            const bool copy = (activeMask >> bit) & 1U;
            for (uint32_t i = threadIdx.x; i < kTop; i += 256) {
                asc_stcg(slot + i, copy ? asc_ldcg(loc + i) : UINT32_MAX);
            }
            targets &= targets - 1;
        }
    }
    asc_threadfence();
}

// Unique cold sources and exclusive hot destinations form disjoint swaps.
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void SwapDisjoint(__gm__ Args *a, uint32_t b, uint32_t begin,
                                                                  uint32_t count) {
    auto ctx = reinterpret_cast<__gm__ Context *>(a->context);
    auto binding = reinterpret_cast<__gm__ uint64_t *>(a->updateBindings);
    uint32_t owner = OwnerRow(ctx, b);
    auto src = reinterpret_cast<__gm__ uint32_t *>(a->eventsSrc) + b * a->maxEvents + begin;
    auto dst = reinterpret_cast<__gm__ uint32_t *>(a->eventsDst) + b * a->maxEvents + begin;
    auto keys = reinterpret_cast<__gm__ uint64_t *>(a->keys) + uint64_t(owner) * a->history * 16;
    auto scales = reinterpret_cast<__gm__ uint32_t *>(a->scales) + uint64_t(owner) * a->history;
    auto addr = reinterpret_cast<__gm__ uint64_t *>(a->addresses) + uint64_t(owner) * binding[7];
    auto logical = reinterpret_cast<__gm__ uint64_t *>(a->logical) + uint64_t(owner) * a->history;
    for (uint32_t x = threadIdx.x; x < count * 16; x += 512) {
        uint32_t s = asc_ldcg(src + x / 16), d = asc_ldcg(dst + x / 16), j = x % 16;
        auto u = asc_ldcg(keys + uint64_t(s) * 16 + j), v = asc_ldcg(keys + uint64_t(d) * 16 + j);
        asc_stcg(keys + uint64_t(s) * 16 + j, v);
        asc_stcg(keys + uint64_t(d) * 16 + j, u);
        if (!j) {
            auto sv = asc_ldcg(scales + s), dv = asc_ldcg(scales + d);
            auto sa = asc_ldcg(addr + s), da = asc_ldcg(addr + d), sl = asc_ldcg(logical + s),
                 dl = asc_ldcg(logical + d);
            asc_stcg(scales + s, dv);
            asc_stcg(scales + d, sv);
            asc_stcg(addr + s, da);
            asc_stcg(addr + d, sa);
            asc_stcg(logical + s, dl);
            asc_stcg(logical + d, sl);
        }
    }
    asc_threadfence();
}

#include "query_owner_rows.h"

__aicore__ inline void PublishIndexCompletion(__gm__ BatchFlag *chain, uint32_t q, Slice &slice) {
    Signal(chain + q, slice.count, slice.error);
    // The final owner joins Index completion before retiring the request.
    if (q == 3) {
        for (uint32_t j = 0; j < q; ++j) {
            slice.error |= AwaitPublication(chain + j);
        }
    }
}

struct IndexTail {
    __gm__ Args *a;
    uint32_t b, q;
    Slice s;
    bool ran = false;
    // Invoked by this query's own KVAF submission after its last SQ doorbell.
    __aicore__ __attribute__((noinline)) inline void Run() {
        if (ran) {
            return;
        }
        ran = true;
        auto ctx = reinterpret_cast<__gm__ Context *>(a->context);
        // LI already saved the old logical IDs and DDR addresses. Each owner
        // can publish its final slots without reading another owner's Index.
        if (!s.error) {
            Simt::VF_CALL<AdaptSlots>(Simt::Dim3{256, 1, 1}, a, b, q);
            VectorDone();
        }
        auto chain = index_team::Control(a, b);
        auto outputs = reinterpret_cast<__gm__ BatchFlag *>(a->flags) + b * kFlagCount + 1;
        auto row = reinterpret_cast<__gm__ Row *>(ctx->rows) + b;
        if (s.disjoint) {
            outputs[q].reserved[kOwnerPlanFirst] = 0;
            outputs[q].reserved[kOwnerPlanEnd] = 0;
            outputs[q].reserved[kOwnerPlanEdges] = 0;
            Signal(outputs + q, s.count, s.error);
            if (!s.error && !q && row->slots) {
                Simt::VF_CALL<SwapDisjoint>(Simt::Dim3{512, 1, 1}, a, b, 0, row->slots);
            }
            if (!s.error && s.count) {
                Simt::VF_CALL<SwapDisjoint>(Simt::Dim3{512, 1, 1}, a, b, kPrefetch + s.begin, s.count);
            }
            VectorDone();
            PublishIndexCompletion(chain, q, s);
            return;
        }
        // Only the permutation planner consumes its predecessor's map.
        // The next query may plan while this query copies the rows.
        uint32_t first = 0, edges = 0;
        if (q && !s.error) {
            auto predecessor = outputs + q - 1;
            s.error |= AwaitPublication(predecessor);
            if (!s.error) {
                first = predecessor->reserved[kOwnerPlanEnd];
                edges = predecessor->reserved[kOwnerPlanEdges];
            }
        }
        if (s.error) {
            outputs[q].reserved[kOwnerPlanFirst] = 0;
            outputs[q].reserved[kOwnerPlanEnd] = 0;
            outputs[q].reserved[kOwnerPlanEdges] = 0;
            Signal(outputs + q, s.count, s.error);
            PublishIndexCompletion(chain, q, s);
            return;
        }
        auto saved = Scratch(a, b) + kOwnerResolverWords;
        auto dense = reinterpret_cast<__gm__ uint32_t *>(a->denseShadow) + uint64_t(b) * a->history * 6;
        auto src = reinterpret_cast<__gm__ uint32_t *>(a->eventsSrc) + b * a->maxEvents;
        auto dst = reinterpret_cast<__gm__ uint32_t *>(a->eventsDst) + b * a->maxEvents;
        TPipe pipe;
        TBuf<TPosition::VECCALC> bytes;
        pipe.InitBuffer(bytes, SCRATCH_WORDS * 4);
        auto mem = bytes.Get<uint32_t>();
        auto m = FinalPtr(mem);
        const uint32_t prior = q ? row->slots + s.begin : 0;
        bool isDense = q && prior * 2 >= MAP * 3 / 4;
        if (!isDense) {
            if (!prior) {
                Simt::VF_CALL<InitShadow>(Simt::Dim3{THREADS, 1, 1}, m);
                VectorDone();
            } else {
                FromGm(mem, saved, 4 * MAP);
                FromGm(mem[CONTROL], saved + 4 * MAP, 32);
                InputReady();
            }
        } else {
            Duplicate(mem[CONTROL], 0U, 32);
            VectorDone();
        }
        auto plan = index_team::Moves(a, b);
        auto moves = reinterpret_cast<__gm__ uint64_t *>(plan);
        auto deps = Scratch(a, b) + kEdgeDeps;
        auto completed = reinterpret_cast<__gm__ BatchFlag *>(Scratch(a, b) + kChunkFlags);
        auto last = LastWriters(a, ctx->shape.batch, b);
        mem.SetValue(CONTROL + 4, first);
        mem.SetValue(CONTROL + 5, edges);
        PipeBarrier<PIPE_ALL>();
        uint32_t seen = prior;
        for (uint32_t part = q ? 1 : 0; part < 2 && !s.error; ++part) {
            uint32_t n = part ? s.count : row->slots, begin = part ? kPrefetch + s.begin : 0;
            if (!n) {
                continue;
            }
            if (!isDense && (seen + n) * 2 >= MAP * 3 / 4) {
                Simt::VF_CALL<InitDenseShadow>(Simt::Dim3{THREADS, 1, 1}, dense, a->history, m);
                VectorDone();
                isDense = true;
            }
            if (isDense) {
                Simt::VF_CALL<ApplyDenseBatch<false, true>>(Simt::Dim3{THREADS, 1, 1}, src, dst, begin, n, nullptr,
                    nullptr, nullptr, a->resident, dense, a->history, m, nullptr, nullptr, moves);
            } else {
                Simt::VF_CALL<ApplyBatch<false, true>>(Simt::Dim3{THREADS, 1, 1}, src, dst, begin, n, nullptr,
                    nullptr, nullptr, a->resident, m, nullptr, nullptr, moves, a->history);
            }
            VectorDone();
            s.error |= mem.GetValue(CONTROL);
            seen += n;
        }
        if (!s.error && q < 3 && !isDense && seen) {
            BeforeStore();
            ToGm(saved, mem, 4 * MAP);
            ToGm(saved + 4 * MAP, mem[CONTROL], 32);
            StoreDone();
        }
        uint32_t end = s.error ? first : mem.GetValue(CONTROL + 4);
        if (!s.error && end > first) {
            Simt::VF_CALL<StampDependencies>(Simt::Dim3{512, 1, 1}, plan, deps, last, first, end);
            VectorDone();
        }
        if (!s.error) {
            edges = mem.GetValue(CONTROL + 5);
        }
        pipe.Destroy();
        outputs[q].reserved[kOwnerPlanFirst] = first;
        outputs[q].reserved[kOwnerPlanEnd] = end;
        outputs[q].reserved[kOwnerPlanEdges] = edges;
        Signal(outputs + q, s.count, s.error);
        if (end > first) {
            s.error |= CopyChunks(a,b,first,end,plan,deps,completed);
        }
        PublishIndexCompletion(chain, q, s);
    }
};

__aicore__ __attribute__((noinline)) inline void Run(__gm__ Args *a, uint32_t b, uint32_t q) {
    auto ctx = reinterpret_cast<__gm__ Context *>(a->context);
    const uint32_t batch = ctx->shape.batch;
    auto flags = reinterpret_cast<__gm__ BatchFlag *>(a->flags) + b * kFlagCount;
    auto row = reinterpret_cast<__gm__ Row *>(ctx->rows) + b;
    Slice s = Resolve(a, b, q);
    // Preserve the final plan receipt for replay. Each query owns its SQ and CQ.
    auto raw = flags + kRawQueryFlag + q;
    raw->reserved[0] = s.begin + s.count;
    Signal(raw, s.count, s.error);
    uint32_t first = (b * 4 + q) * kTop;
    auto post = FinalPostAt(ctx->shape.finalPostOffset, ctx, b * 4 + q);
    PrepareFinalFetch(4, ctx->shape.finalPostOffset, ctx, b, q, s.count, first, s.error, s.safe);

    const uint32_t view = FinalQueue(batch, 4, b, q);
    auto native = GetArgs(ctx, view);
    if (native.block_count != 1 || native.jettys_per_core != 1) {
        s.error |= kvaf_device::kInvalidArgs;
    }
    auto status = reinterpret_cast<__gm__ KvafBatchGetCoreStatus *>(Scratch(a, b) + kOwnerResolverWords +
                                                                    kOwnerIndexWords + q * kOwnerStatusWords);
    native.core_status = uint64_t(status);
    native.indices = ctx->sourceBlocks;
    native.offsets = ctx->sourceOffsets;
    native.output_indices = ctx->destinations;
    native.request_capacity = batch * kSlots;
    if (ctx->fallbackAddresses) {
        native.address_table = ctx->fallbackAddresses;
        native.valid_table = ctx->fallbackValid;
        native.table_capacity = batch * kSlots;
    }
    IndexTail tail{a, b, q, s};
    uint32_t completed = 0, processed = 0;
    if (!s.error && s.count) {
        SyncFetchDescriptors(ctx, first, s.count);
        auto queue = reinterpret_cast<__gm__ uint8_t *>(native.queue_state);
        kvaf_device::CleanCacheLines(queue, sizeof(KvafBatchGetQueueState));
        kvaf_device::WriteStatus(status, kvaf_device::kInvalidArgs, 0, 0, 0, 0, UINT64_MAX, 0);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(status), sizeof(*status));
        kvaf_device::KvafBatchGetDeviceWithPostTail(native, 0, first, s.count, tail);
        kvaf_device::CleanCacheLines(queue, sizeof(KvafBatchGetQueueState));
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(status), sizeof(*status));
        s.error |= status->result;
        if (status->magic != KVAF_AIV_BATCH_GET_STATUS_MAGIC || status->processed_request_count != s.count) {
            s.error |= kvaf_device::kInvalidArgs;
        }
        completed = status->completed_batch_count;
        processed = status->processed_request_count;
    }
    // Empty queries still publish output slots and advance the Index chain.
    if (!tail.ran) {
        tail.s.error |= s.error;
        tail.Run();
    }
    s.error |= tail.s.error;
    post->pending = completed;
    post->error = s.error;
    post->deferred = 0;
    post->ownerBlock = GetBlockIdx() + 1;
    post->queueView = view;
    post->processed = processed;
    post->offsetEscape = uint32_t(s.escape);
    post->independentIndex = uint32_t(tail.s.disjoint);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(post), sizeof(*post));
    flags[8 + q].reserved[0] = 1;
    Signal(flags + 8 + q, 0, s.error);
    if (q == 3) {
        CompletePrefetch(ctx, b, true);
        Signal(flags + 7, 0, s.error | row->error);
        uint32_t total = s.begin + s.count;
        auto info = reinterpret_cast<__gm__ uint32_t *>(ctx->rowStatus) + b * kRowStatusWords;
        info[0] = s.error;
        info[1] = total;
        info[2] = b * kSlots;
        row->finalRequests = total;
        row->finalPending = total ? 1 : 0;
        row->finalEnd = GetSystemCycle();
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(info), 128);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(row) + 960, 64);
        Signal(flags + 5, 0, s.error);
        auto update = reinterpret_cast<__gm__ uint64_t *>(a->updateBindings);
        Simt::VF_CALL<PublishStreamingUpdateStatus>(Simt::Dim3{32, 1, 1},
                                                    reinterpret_cast<__gm__ uint32_t *>(update[6]), b, s.error);
        Simt::VF_CALL<RetireSelection>(Simt::Dim3{128, 1, 1}, a, batch, b);
        VectorDone();
        Signal(flags + 6, 0, s.error);
        FinalizeStreaming<false, true>(a, b);
        ClearRankProgress(ctx, b);
    }
}
} // namespace query_owner
