// SPDX-License-Identifier: Apache-2.0
// Four resident AIVs per request. A query owns its descriptor slice and its SQ/CQ.
// Two independent handoffs preserve first-occurrence slot order and Index permutation order.
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

// Keep the Index event order independent of the safe/conflict order in the private SQ input.
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void ResolveOwnerQuery(StagedContextPointer ctx,
    uint32_t history, uint32_t resident, uint32_t b, uint32_t q, uint32_t used,
    __gm__ uint32_t *src, __gm__ uint32_t *dst, __ubuf__ uint32_t *selected, __ubuf__ uint32_t *loc,
    __ubuf__ uint32_t *keys, __ubuf__ uint32_t *ranks, __ubuf__ uint16_t *ids, __ubuf__ uint32_t *slots,
    __ubuf__ uint32_t *work, __ubuf__ uint32_t *ctl, __gm__ uint32_t *protect, __gm__ uint32_t *liveCount) {
    StagedQuery(q, history, resident, ctx->shape.tokenBytes, b, used, ctx, selected, loc,
                keys, ranks, ids, slots, work, ctl, protect);
    if (ctl[1]) return;
    const uint32_t count = ctl[2 + q], begin = ctl[0] - count;
    if (count && (!protect || ctl[kProtectionState])) {
        // The protected allocator may have reused prefetch slots. Preserve the
        // original stable two-phase packing, consuming the resolver's UB values.
        if (!threadIdx.x) ctx->count = reinterpret_cast<uint64_t>(liveCount);
        PublishQ(selected, ids, slots, src, dst, begin, begin + count);
        PackStagedUrmaQuery<512>(history, resident, ctx->shape.tokenBytes, b, q, used, begin, count,
                                ctx, selected, ids, slots, ctl);
        return;
    }
    // StagedQuery verified the inverse LRU ordinal of every unprotected
    // destination against the unused suffix. All of these Gets are safe.
    if (count) {
        const uint32_t ownerBase = OwnerRow(ctx, b) * OutputStride(resident, ctx);
        const uint32_t output = (4 * b + q) * kTop;
        for (uint32_t i = threadIdx.x; i < count; i += 512) {
            const uint32_t rank = begin + i, token = selected[ids[rank % kTop]], slot = slots[rank];
            asc_stcg(src + kPrefetch + rank, token);
            asc_stcg(dst + kPrefetch + rank, slot);
            StoreOwnerFinalDescriptor(ctx, b, output + i, token, slot, ownerBase, ctl);
        }
        asc_threadfence();
        asc_syncthreads();
    }
    if (!threadIdx.x) {
        ctl[17] = count;
        if (count && !ctl[1]) asc_atomic_add(liveCount, count);
    }
    asc_threadfence();
    asc_syncthreads();
}

// The first resolver's prefix and later query suffixes use the same protected
// allocator as the fixed graph, including ordinary hits outside the LRU list.
__aicore__ __attribute__((noinline)) inline Slice Resolve(__gm__ Args *a, uint32_t b, uint32_t q) {
    auto ctx = reinterpret_cast<__gm__ Context *>(a->context);
    auto row = reinterpret_cast<__gm__ Row *>(ctx->rows) + b;
    auto flags = reinterpret_cast<__gm__ BatchFlag *>(a->flags) + b * kFlagCount;
    auto saved = Scratch(a, b);
    auto src = reinterpret_cast<__gm__ uint32_t *>(a->eventsSrc) + b * a->maxEvents;
    auto dst = reinterpret_cast<__gm__ uint32_t *>(a->eventsDst) + b * a->maxEvents;
    auto positions = reinterpret_cast<__gm__ uint32_t *>(ctx->selected) + b * kSlots;
    auto locations = reinterpret_cast<__gm__ uint32_t *>(ctx->locations) + b * kSlots;
    auto lru = reinterpret_cast<__gm__ uint32_t *>(ctx->lruSlots) + b * kLruSlots;
    auto headers = reinterpret_cast<__gm__ uint32_t *>(a->queryErrors) + 128 * b;
    uint32_t upper = 0, prefixUpper = 0;
    Slice s;
    for (uint32_t j = 0; j < 4; ++j) {
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(headers + 32 * j), 64);
        s.error |= headers[32 * j];
        uint32_t n = headers[32 * j + 1];
        upper += n && n <= kCompactMissLimit + 1 ? n - 1 : kTop;
        if (j == q) prefixUpper = upper;
    }
    if (row->slots > kPrefetch) {
        s.error |= 4;
    }
    if (q && headers[32 * q + 1] == 1) {
        // The classifier already filled every location. Leave the allocator
        // handoff in place; later queries must still see its protected slots.
        auto prior = flags + kRawQueryFlag + q - 1;
        s.error |= AwaitPublication(prior);
        s.begin = prior->reserved[0];
        flags[kRawQueryFlag + q].reserved[1] = prior->reserved[1];
        return s;
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
    // The final classifier bounds new slots through this Q before its predecessor
    // publishes the exact deduplicated prefix. Keep the DMA ahead of that wait.
    const uint32_t used = row->slots;
    const bool shortLru = prefixUpper < kTop && used <= kLruSlots - prefixUpper;
    if (!shortLru) {
        FromGm(work[4096], lru, kLruSlots);
    } else if (prefixUpper) {
        const uint32_t lo = used & ~7U;
        FromGm(work[4096 + lo], lru + lo, used + prefixUpper - lo);
    }
    if (!q) {
        if (ctx->rowAddressBase) {
            uint32_t owner = ctx->rowOwners ? reinterpret_cast<__gm__ uint32_t *>(ctx->rowOwners)[b] : b;
            row->addresses = RequestAddressTable(ctx, b, owner, a->history);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(row), 64);
        }
        if (!s.error) {
            Simt::VF_CALL<PublishPrefetch>(
                Simt::Dim3{512, 1, 1}, reinterpret_cast<__gm__ uint32_t *>(ctx->changes) + b * kChangeCapacity,
                lru, src, dst, row->slots);
            VectorDone();
        }
        Signal(flags, row->slots, s.error);
        Duplicate(keys, UINT32_MAX, buckets);
        Duplicate(ranks, UINT32_MAX, buckets);
        Duplicate(ctl, 0U, 32);
        VectorDone();
        ctl.SetValue(7, buckets);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
    } else {
        s.error |= AwaitPublication(flags + kRawQueryFlag + q - 1);
        auto f = flags + kRawQueryFlag + q - 1;
        s.begin = f->reserved[0];
        if (s.begin > kSlots - kTop) {
            s.error |= 4;
        }
        if (!s.error) {
            FromGm(keys, saved, buckets);
            FromGm(ranks, saved + kSlots, buckets);
            FromGm(slots, saved + 2 * kSlots, s.begin);
            FromGm(ctl, saved + 3 * kSlots, 32);
            if (f->reserved[1]) {
                FromGm(ctl[1024], saved + 3 * kSlots + 32, (a->resident + 31) / 32);
            }
        }
    }
    InputReady();
    if (!s.error) {
        // LI already identified the unresolved positions. Reuse the production
        // resolver, which checks the unused LRU suffix before changing any slot.
        const uint32_t encoded = headers[32 * q + 1];
        bool compact = encoded && encoded <= kCompactMissLimit + 1 &&
                       row->slots + s.begin + encoded - 1 <= kLruSlots && !ctl.GetValue(kProtectionState);
        if (compact) {
            ctl.SetValue(16, 0);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            auto entries = reinterpret_cast<__gm__ uint32_t *>(ctx->sourceBlocks) + (4 * b + q) * kTop;
            Simt::VF_CALL<ResolveOwnerCompact>(Simt::Dim3{64, 1, 1}, staged, b, q, a->resident, row->slots,
                encoded - 1, entries, FinalPtr(keys), FinalPtr(ranks), FinalPtr(slots), FinalPtr(work[4096]),
                FinalPtr(ctl), src, dst, reinterpret_cast<__gm__ uint32_t*>(a->liveCount));
            VectorDone();
            compact = !ctl.GetValue(16);
        }
        if (!compact) {
            // The protected allocator can inspect any ordinal in the original table.
            if (shortLru) FromGm(work[4096], lru, kLruSlots);
            FromGm(ub[q * kTop], positions + q * kTop, kTop);
            FromGm(loc[q * kTop], locations + q * kTop, kTop);
            if (q >= 2) {
                FromGm(ub[(q - 2) * kTop], positions + (q - 2) * kTop, kTop);
                FromGm(loc[(q - 2) * kTop], locations + (q - 2) * kTop, kTop);
            }
            InputReady();
            auto protect = reinterpret_cast<__gm__ uint32_t *>(a->queryBitmaps) + b * 4 * ((a->resident + 31) / 32);
            Simt::VF_CALL<ResolveOwnerQuery>(
                Simt::Dim3{512, 1, 1}, staged, a->history, a->resident, b, q, row->slots,
                src, dst, FinalPtr(ub), FinalPtr(loc),
                FinalPtr(keys), FinalPtr(ranks), reinterpret_cast<__ubuf__ uint16_t *>(ids.GetPhyAddr()),
                FinalPtr(slots), FinalPtr(work), FinalPtr(ctl), protect, reinterpret_cast<__gm__ uint32_t*>(a->liveCount));
            VectorDone();
        }
        s.error = ctl.GetValue(1);
        if (!s.error) {
            s.count = ctl.GetValue(2 + q);
            s.safe = ctl.GetValue(17);
            const uint32_t protection = ctl.GetValue(kProtectionState);
            flags[kRawQueryFlag + q].reserved[1] = protection;
            BeforeStore();
            if (!compact) ToGm(locations + q * kTop, loc[q * kTop], kTop);
            if (q < 3) {
                ToGm(saved, keys, buckets);
                ToGm(saved + kSlots, ranks, buckets);
                ToGm(saved + 2 * kSlots, slots, s.begin + s.count);
                ToGm(saved + 3 * kSlots, ctl, 32);
                if (protection) {
                    ToGm(saved + 3 * kSlots + 32, ctl[1024], (a->resident + 31) / 32);
                }
            }
            StoreDone();
        }
    }
    pipe.Destroy();
    return s;
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
    __aicore__ inline void WriteOutput() {
        Simt::VF_CALL<Adapt>(Simt::Dim3{256, 1, 1}, a, b, q);
        VectorDone();
    }
    // Invoked by this query's own KVAF submission after its last SQ doorbell.
    __aicore__ __attribute__((noinline)) inline void Run() {
        if (ran) {
            return;
        }
        ran = true;
        auto ctx = reinterpret_cast<__gm__ Context *>(a->context);
        auto outputs = reinterpret_cast<__gm__ BatchFlag *>(a->flags) + b * kFlagCount + 1;
        const bool validOutput = !s.error;
        if (q && validOutput) {
            WriteOutput();
        }
        auto chain = index_team::Control(a, b);
        if (q) {
            s.error |= AwaitPublication(chain + q - 1);
        }
        auto row = reinterpret_cast<__gm__ Row *>(ctx->rows) + b;
        // No swaps to apply and no final metadata commit on a middle owner.
        // Its successor can consume the very same saved permutation.
        if ((q == 1 || q == 2) && !s.count) {
            Signal(outputs + q, 0, s.error);
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
        if (q == 3 && seen && !s.error) {
            // Q2/Q3 read the old mapping before their Index handoff; Q1 reads it afterwards.
            s.error |= AwaitPublication(outputs);
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
        if (!q) {
            // The Index shadow is available to Q2 while Q1 saves its output.
            Signal(chain, s.count, s.error);
            if (validOutput) {
                WriteOutput();
            }
        }
        Signal(outputs + q, s.count, s.error);
        if (q) {
            Signal(chain + q, s.count, s.error);
        }
    }
};

__aicore__ __attribute__((noinline)) inline void Run(__gm__ Args *a, uint32_t b, uint32_t q) {
    auto ctx = reinterpret_cast<__gm__ Context *>(a->context);
    const uint32_t batch = ctx->shape.batch;
    auto flags = reinterpret_cast<__gm__ BatchFlag *>(a->flags) + b * kFlagCount;
    auto row = reinterpret_cast<__gm__ Row *>(ctx->rows) + b;
    // The spare odd prefetch status is separate from every private final Get status.
    if (q == 3) {
        // Q1 publishes row->addresses in the same cache line that the CQ owner cleans.
        auto error = AwaitPublication(flags);
        CompletePrefetch(ctx, b, true);
        Signal(flags + 7, 0, error | row->error);
    }
    Slice s = Resolve(a, b, q);
    // The next allocator needs the slots and captured descriptor addresses;
    // this query alone owns its subsequent SQ post and CQ.
    auto raw = flags + kRawQueryFlag + q;
    raw->reserved[0] = s.begin + s.count;
    Signal(raw, s.count, s.error);
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
