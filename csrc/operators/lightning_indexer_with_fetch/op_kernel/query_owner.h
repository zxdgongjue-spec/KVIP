// SPDX-License-Identifier: Apache-2.0
// Four resident AIVs per request. A query owns its descriptor slice and its SQ/CQ.
// The request-local allocator and the ordered Index commits both follow Q1 through Q4.
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
};
__aicore__ inline __gm__ uint32_t *Scratch(__gm__ Args *a, uint32_t b) {
    return reinterpret_cast<__gm__ uint32_t *>(a->queryOwnerScratch) + b * kQueryOwnerRowWords;
}

// Q4 owns the allocator for one request. Publish each descriptor slice as
// soon as it is ready so the first three query owners can post independently.
__aicore__ __attribute__((noinline)) inline void PlanRequest(__gm__ Args *a, uint32_t b) {
    auto ctx = reinterpret_cast<__gm__ Context *>(a->context);
    auto row = reinterpret_cast<__gm__ Row *>(ctx->rows) + b;
    auto flags = reinterpret_cast<__gm__ BatchFlag *>(a->flags) + b * kFlagCount;
    auto src = reinterpret_cast<__gm__ uint32_t *>(a->eventsSrc) + b * a->maxEvents;
    auto dst = reinterpret_cast<__gm__ uint32_t *>(a->eventsDst) + b * a->maxEvents;
    auto positions = reinterpret_cast<__gm__ uint32_t *>(ctx->selected) + b * kSlots;
    auto locations = reinterpret_cast<__gm__ uint32_t *>(ctx->locations) + b * kSlots;
    auto lru = reinterpret_cast<__gm__ uint32_t *>(ctx->lruSlots) + b * kLruSlots;
    auto headers = reinterpret_cast<__gm__ uint32_t *>(a->queryErrors) + 128 * b;
    uint32_t upper = 0;
    uint32_t error = 0;
    for (uint32_t j = 0; j < 4; ++j) {
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(headers + 32 * j), 64);
        error |= headers[32 * j];
        uint32_t n = headers[32 * j + 1];
        upper += n && n <= kCompactMissLimit + 1 ? n - 1 : kTop;
    }
    if (row->slots > kPrefetch) {
        error |= 4;
    }
    uint32_t buckets = upper <= 256 ? 512U : (upper <= 1024 ? 2048U : (upper <= 2048 ? 4096U : kSlots));
    TPipe pipe;
    TBuf<TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, kFinalScratchBytes);
    auto ub = storage.Get<uint32_t>(), loc = ub[kSlots], keys = ub[2 * kSlots], ranks = ub[3 * kSlots];
    auto ids = ub[4 * kSlots].ReinterpretCast<uint16_t>();
    auto slots = ub[kStagedSlotsOffset], work = ub[kStagedPositionsOffset], ctl = ub[kStagedControlOffset];
    auto staged = reinterpret_cast<__ubuf__ Context *>(ctl[128].GetPhyAddr());
    FromGm(ctl[128], reinterpret_cast<__gm__ uint32_t *>(ctx), sizeof(Context) / 4);
    FromGm(work[4096], lru, kLruSlots);
    if (ctx->rowAddressBase) {
        uint32_t owner = ctx->rowOwners ? reinterpret_cast<__gm__ uint32_t *>(ctx->rowOwners)[b] : b;
        row->addresses = RequestAddressTable(ctx, b, owner, a->history);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(row), 64);
    }
    if (!error) {
        Simt::VF_CALL<PublishPrefetch>(
            Simt::Dim3{512, 1, 1}, reinterpret_cast<__gm__ uint32_t *>(ctx->changes) + b * kChangeCapacity,
            lru, src, dst, row->slots);
        VectorDone();
    }
    Signal(flags, row->slots, error);
    Duplicate(keys, UINT32_MAX, buckets);
    Duplicate(ranks, UINT32_MAX, buckets);
    Duplicate(ctl, 0U, 32);
    VectorDone();
    ctl.SetValue(7, buckets);
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    InputReady();
    auto saved = Scratch(a, b);
    uint32_t protection = 0;
    uint32_t begin = 0;
    for (uint32_t q = 0; q < 4; ++q) {
        Slice s{begin, 0, 0, error};
        const uint32_t encoded = headers[32 * q + 1];
        if (begin > kSlots - kTop) s.error |= 4;
        if (!s.error && (q == 0 || encoded != 1)) {
            if (q) {
                FromGm(keys, saved, buckets);
                FromGm(ranks, saved + kSlots, buckets);
                FromGm(slots, saved + 2 * kSlots, begin);
                FromGm(ctl, saved + 3 * kSlots, 32);
                if (protection) FromGm(ctl[1024], saved + 3 * kSlots + 32, (a->resident + 31) / 32);
                InputReady();
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            }
            bool compact = encoded && encoded <= kCompactMissLimit + 1 &&
                           row->slots + s.begin + encoded - 1 <= kLruSlots && !ctl.GetValue(kProtectionState);
            if (compact) {
                ctl.SetValue(16, 0);
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
                auto entries = reinterpret_cast<__gm__ uint32_t *>(ctx->sourceBlocks) + (4 * b + q) * kTop;
                Simt::VF_CALL<ResolveCompact>(Simt::Dim3{64, 1, 1}, staged, b, q, a->resident, row->slots,
                    encoded - 1, entries, FinalPtr(keys), FinalPtr(ranks), FinalPtr(slots), FinalPtr(work[4096]),
                    FinalPtr(ctl), src, dst, nullptr, nullptr, 0, nullptr);
                VectorDone();
                compact = !ctl.GetValue(16);
            }
            if (!compact) {
                FromGm(ub[q * kTop], positions + q * kTop, kTop);
                FromGm(loc[q * kTop], locations + q * kTop, kTop);
                if (q >= 2) {
                    FromGm(ub[(q - 2) * kTop], positions + (q - 2) * kTop, kTop);
                    FromGm(loc[(q - 2) * kTop], locations + (q - 2) * kTop, kTop);
                }
                InputReady();
                auto protect = reinterpret_cast<__gm__ uint32_t *>(a->queryBitmaps) + b * 4 * ((a->resident + 31) / 32);
                Simt::VF_CALL<ResolveQueryWork>(
                    Simt::Dim3{512, 1, 1}, staged, a->history, a->resident, ctx->shape.tokenBytes, row->slots,
                    ctx->shape.batch, b, q, nullptr, nullptr, nullptr, src, dst, FinalPtr(ub), FinalPtr(loc),
                    FinalPtr(keys), FinalPtr(ranks), reinterpret_cast<__ubuf__ uint16_t *>(ids.GetPhyAddr()),
                    FinalPtr(slots), FinalPtr(work), FinalPtr(ctl), nullptr, 0, protect);
                VectorDone();
            }
            s.error = ctl.GetValue(1);
            if (!s.error) {
                s.count = ctl.GetValue(2 + q);
                if (!compact) {
                    BeforeStore();
                    ToGm(locations + q * kTop, loc[q * kTop], kTop);
                    StoreDone();
                    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                }
                if (q < 3) {
                    protection = ctl.GetValue(kProtectionState);
                    BeforeStore();
                    ToGm(saved, keys, buckets);
                    ToGm(saved + kSlots, ranks, buckets);
                    ToGm(saved + 2 * kSlots, slots, begin + s.count);
                    ToGm(saved + 3 * kSlots, ctl, 32);
                    if (protection) ToGm(saved + 3 * kSlots + 32, ctl[1024], (a->resident + 31) / 32);
                    StoreDone();
                    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                }
                if (s.count) {
                    // This query's input positions are dead until the explicit
                    // q-2 reload. Keep keys, ranks, slots and protection intact.
                    staged->count = a->liveCount;
                    SetFlag<HardEvent::S_V>(EVENT_ID0);
                    WaitFlag<HardEvent::S_V>(EVENT_ID0);
                    Simt::VF_CALL<PackPublicationEvents<512>>(Simt::Dim3{512, 1, 1}, staged, a->history, a->resident,
                                                              ctx->shape.tokenBytes, b, q, row->slots,
                                                              kPrefetch + s.begin, s.count, src, dst, FinalPtr(ub[q * kTop]),
                                                              reinterpret_cast<__ubuf__ uint16_t *>(ids.GetPhyAddr()),
                                                              FinalPtr(work), FinalPtr(ctl), nullptr, 0, nullptr);
                    VectorDone();
                    s.error = ctl.GetValue(1);
                    s.safe = ctl.GetValue(17);
                }
            }
        }
        auto raw = flags + kRawQueryFlag + q;
        raw->reserved[0] = begin;
        raw->reserved[1] = s.safe;
        Signal(raw, s.count, s.error);
        begin += s.count;
        error = s.error;
    }
    pipe.Destroy();
}

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
        // owners. The last commit is ordered after all four owner handoffs.
        if (!s.error) {
            Simt::VF_CALL<Adapt>(Simt::Dim3{256, 1, 1}, a, b, q);
            VectorDone();
        }
        auto chain = index_team::Control(a, b);
        if (q) {
            s.error |= AwaitPublication(chain + q - 1);
        }
        auto row = reinterpret_cast<__gm__ Row *>(ctx->rows) + b;
        // No swaps to apply and no final metadata commit on a middle owner.
        // Its successor can consume the very same saved permutation.
        if ((q == 1 || q == 2) && !s.count) {
            Signal(reinterpret_cast<__gm__ BatchFlag *>(a->flags) + b * kFlagCount + q + 1, 0, s.error);
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
                Simt::VF_CALL<ApplyDenseBatch<false>>(Simt::Dim3{THREADS, 1, 1}, src, dst, begin, n, keys,
                                                      scales, valid, a->resident, dense, a->history, m, nullptr,
                                                      nullptr, nullptr);
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
                    Simt::VF_CALL<CommitDenseMetadata>(
                        Simt::Dim3{THREADS, 1, 1}, addr, logical, dense,
                        reinterpret_cast<__gm__ uint64_t *>(dense + 2 * a->history), a->history);
                } else {
                    Simt::VF_CALL<CommitMetadata>(Simt::Dim3{THREADS, 1, 1}, addr, logical, m);
                }
                VectorDone();
            }
        }
        pipe.Destroy();
        Signal(reinterpret_cast<__gm__ BatchFlag *>(a->flags) + b * kFlagCount + q + 1, s.count, s.error);
        Signal(chain + q, s.count, s.error);
    }
};

__aicore__ __attribute__((noinline)) inline void Run(__gm__ Args *a, uint32_t b, uint32_t q) {
    auto ctx = reinterpret_cast<__gm__ Context *>(a->context);
    const uint32_t batch = ctx->shape.batch;
    auto flags = reinterpret_cast<__gm__ BatchFlag *>(a->flags) + b * kFlagCount;
    auto row = reinterpret_cast<__gm__ Row *>(ctx->rows) + b;
    // Q3 can drain the independent prefetch CQ while Q4 plans. The
    // prefetch queue is addressed by request; final CQs remain Q-owned.
    if (q == 3) {
        PlanRequest(a, b);
    } else if (q == 2) {
        auto error = AwaitPublication(flags);
        CompletePrefetch(ctx, b, true);
        Signal(flags + 7, 0, error | row->error);
    }
    auto raw = flags + kRawQueryFlag + q;
    Slice s;
    s.error = AwaitPublication(raw);
    s.begin = raw->reserved[0];
    s.count = raw->count;
    s.safe = raw->reserved[1];
    uint32_t first = (b * 4 + q) * kTop;
    auto post = FinalPostAt(ctx->shape.finalPostOffset, ctx, b * 4 + q);
    PrepareFinalFetch(4, ctx->shape.finalPostOffset, ctx, b, q, s.count, first, s.error, s.safe);
    if (s.safe < s.count) {
        s.error |= AwaitPublication(flags + 7);
    }
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
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(post), sizeof(*post));
    flags[8 + q].reserved[0] = 1;
    Signal(flags + 8 + q, 0, s.error);
    if (q == 3) {
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
