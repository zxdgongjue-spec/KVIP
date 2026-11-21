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

// Emit only rows consuming this query, including unpack fanout and masked rows.
// The physical-to-logical table stays immutable until all four outputs are saved.
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void Adapt(__gm__ Args *a, uint32_t b, uint32_t q) {
    auto ctx = reinterpret_cast<__gm__ Context *>(a->context);
    auto u = reinterpret_cast<__gm__ uint64_t *>(a->updateBindings);
    if (!u[20] || !u[21] || !u[22] || !u[23]) {
        return;
    }
    auto packed = reinterpret_cast<__gm__ int64_t *>(u[20]), valid = reinterpret_cast<__gm__ int64_t *>(u[21]);
    auto ids = reinterpret_cast<__gm__ uint32_t *>(ctx->selected) + (4 * b + q) * kTop;
    auto loc = reinterpret_cast<__gm__ uint32_t *>(ctx->locations) + (4 * b + q) * kTop;
    auto logical = reinterpret_cast<__gm__ int64_t *>(u[3]) + uint64_t(OwnerRow(ctx, b)) * u[24];
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
            auto out = reinterpret_cast<__gm__ uint32_t *>(u[22]) + offset;
            auto slot = reinterpret_cast<__gm__ uint32_t *>(u[23]) + offset;
            const bool copy = (activeMask >> bit) & 1U;
            for (uint32_t i = threadIdx.x; i < kTop; i += 256) {
                uint32_t index = copy ? asc_ldcg(ids + i) : UINT32_MAX;
                asc_stcg(out + i, index < a->history ? uint32_t(asc_ldcg(logical + index)) : UINT32_MAX);
                asc_stcg(slot + i, copy ? asc_ldcg(loc + i) : UINT32_MAX);
            }
            targets &= targets - 1;
        }
    }
    asc_threadfence();
}

// Unique cold sources and exclusive hot destinations form disjoint swaps.
// All owners must finish reading the old logical layout before these writes.
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
        // Read the pre-update logical mapping in parallel across the four
        // owners before publishing that it is safe to change metadata.
        if (!s.error) {
            Simt::VF_CALL<Adapt>(Simt::Dim3{256, 1, 1}, a, b, q);
            VectorDone();
        }
        auto chain = index_team::Control(a, b);
        auto outputs = reinterpret_cast<__gm__ BatchFlag *>(a->flags) + b * kFlagCount + 1;
        outputs[q].reserved[0] = uint32_t(s.disjoint);
        Signal(outputs + q, s.count, s.error);
        bool independent = s.disjoint;
        for (uint32_t j = 0; j < 4; ++j) {
            if (j == q)
                continue;
            s.error |= AwaitPublication(outputs + j);
            independent = independent && outputs[j].reserved[0] != 0;
        }
        auto row = reinterpret_cast<__gm__ Row *>(ctx->rows) + b;
        s.disjoint = independent;
        if (independent && !s.error) {
            if (!q && row->slots)
                Simt::VF_CALL<SwapDisjoint>(Simt::Dim3{512, 1, 1}, a, b, 0, row->slots);
            if (s.count)
                Simt::VF_CALL<SwapDisjoint>(Simt::Dim3{512, 1, 1}, a, b, kPrefetch + s.begin, s.count);
            VectorDone();
            Signal(chain + q, s.count, 0);
            if (q == 3)
                for (uint32_t j = 0; j < 3; ++j)
                    s.error |= AwaitPublication(chain + j);
            return;
        }
        if (q)
            s.error |= AwaitPublication(chain + q - 1);
        // No swaps to apply and no final metadata commit on a middle owner.
        // Its successor can consume the very same saved permutation.
        if ((q == 1 || q == 2) && !s.count) {
            Signal(chain + q, 0, s.error);
            return;
        }
        auto saved = Scratch(a, b) + kOwnerResolverWords;
        auto dense = reinterpret_cast<__gm__ uint32_t *>(a->denseShadow) + uint64_t(b) * a->history * 6;
        auto u = reinterpret_cast<__gm__ uint64_t *>(a->updateBindings);
        uint32_t owner = ctx->rowOwners ? reinterpret_cast<__gm__ uint32_t *>(ctx->rowOwners)[b] : b;
        auto keys = reinterpret_cast<__gm__ uint64_t *>(a->keys) + uint64_t(owner) * a->history * 16;
        auto scales = reinterpret_cast<__gm__ uint32_t *>(a->scales) + uint64_t(owner) * a->history;
        auto valid = reinterpret_cast<__gm__ uint32_t *>(a->valid) + uint64_t(owner) * a->resident;
        auto src = reinterpret_cast<__gm__ uint32_t *>(a->eventsSrc) + b * a->maxEvents;
        auto dst = reinterpret_cast<__gm__ uint32_t *>(a->eventsDst) + b * a->maxEvents;
        TPipe pipe;
        TBuf<TPosition::VECCALC> bytes;
        pipe.InitBuffer(bytes, SCRATCH_WORDS * 4);
        auto mem = bytes.Get<uint32_t>();
        auto m = FinalPtr(mem);
        const uint32_t prior = q ? row->slots + s.begin : 0;
        bool isDense = q && prior * 2 >= MAP * 3 / 4;
        if (!s.error && !isDense) {
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
                Simt::VF_CALL<ApplyDenseBatch<false>>(Simt::Dim3{THREADS, 1, 1}, src, dst, begin, n, keys, scales,
                                                      valid, a->resident, dense, a->history, m, nullptr, nullptr,
                                                      nullptr);
            } else {
                Simt::VF_CALL<ApplyBatch<false>>(Simt::Dim3{THREADS, 1, 1}, src, dst, begin, n, keys, scales,
                                                 valid, a->resident, m, nullptr, nullptr, nullptr, a->history);
            }
            VectorDone();
            s.error |= mem.GetValue(CONTROL);
            seen += n;
        }
        if (!s.error) {
            if (q < 3 && !isDense && seen) {
                BeforeStore();
                ToGm(saved, mem, 4 * MAP);
                ToGm(saved + 4 * MAP, mem[CONTROL], 32);
                StoreDone();
            }
            if (q == 3 && seen) {
                auto addr = reinterpret_cast<__gm__ uint64_t *>(a->addresses) + uint64_t(owner) * u[7];
                auto logical = reinterpret_cast<__gm__ uint64_t *>(a->logical) + uint64_t(owner) * a->history;
                if (isDense) {
                    Simt::VF_CALL<CommitDenseMetadata>(Simt::Dim3{THREADS, 1, 1}, addr, logical, dense,
                                                       reinterpret_cast<__gm__ uint64_t *>(dense + 2 * a->history),
                                                       a->history);
                } else {
                    Simt::VF_CALL<CommitMetadata>(Simt::Dim3{THREADS, 1, 1}, addr, logical, m);
                }
                VectorDone();
            }
        }
        pipe.Destroy();
        Signal(chain + q, s.count, s.error);
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
