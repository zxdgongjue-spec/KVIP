// SPDX-License-Identifier: Apache-2.0
// Each Q publishes only its unresolved IDs. Its UB hash is private and expires
// with the call. Reconstructing the prefix from these immutable lists lets a Q
// allocate and post without acquiring another worker's mutable allocator.
constexpr uint32_t kListFlags = kSlots;
static_assert(kListFlags + 4 * kFlagWords <= kOwnerResolverWords);

__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void GatherMisses(__gm__ Context *ctx, uint32_t group,
                                                                  uint32_t encoded, __ubuf__ uint32_t *ids,
                                                                  __ubuf__ uint32_t *positions,
                                                                  __ubuf__ uint32_t *ctl) {
    uint32_t t = threadIdx.x;
    auto selected = reinterpret_cast<__gm__ uint32_t *>(ctx->selected) + group * kTop;
    auto loc = reinterpret_cast<__gm__ uint32_t *>(ctx->locations) + group * kTop;
    if (encoded && encoded <= kCompactMissLimit + 1 && ctx->shape.history <= (1U << (32 - kPositionBits))) {
        if (t + 1 < encoded) {
            auto entries = reinterpret_cast<__gm__ uint32_t *>(ctx->sourceBlocks) + group * kTop;
            uint32_t entry = asc_ldcg(entries + t);
            ids[t] = entry >> kPositionBits;
            positions[t] = entry & kPositionMask;
        }
        if (!t)
            ctl[0] = encoded - 1;
    } else {
        uint32_t bits = 0, token[4];
        for (uint32_t j = 0; j < 4; ++j) {
            uint32_t i = 4 * t + j;
            token[j] = asc_ldcg(selected + i);
            if (token[j] != UINT32_MAX && asc_ldcg(loc + i) == UINT32_MAX)
                bits |= 1U << j;
        }
        uint32_t offset = StagedPrefix(__popc(bits), ctl);
        for (uint32_t j = 0; j < 4; ++j) {
            if (bits & (1U << j)) {
                ids[offset] = token[j];
                positions[offset++] = 4 * t + j;
            }
        }
        if (!t)
            ctl[0] = ctl[559];
    }
}

#include "offset_allocation.h"

__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void EmitOffsetPlan(
    __gm__ Args *a, __gm__ Context *ctx, uint32_t b, uint32_t q, uint32_t begin, uint32_t end,
    __ubuf__ uint32_t *ids, __ubuf__ uint32_t *positions, __ubuf__ uint32_t *rank, __ubuf__ uint32_t *slots,
    __ubuf__ uint32_t *ctl) {
    uint32_t t = threadIdx.x, group = (4 * b + q) * kTop, base = OwnerRow(ctx, b) * OutputStride(a->resident, ctx);
    auto scratch = reinterpret_cast<__gm__ uint32_t *>(a->queryOwnerScratch) + b * kQueryOwnerRowWords;
    auto oldAddresses = reinterpret_cast<__gm__ uint64_t *>(scratch + kOldAddresses) + q * kTop;
    auto inv = reinterpret_cast<__gm__ int32_t *>(ctx->reservedHot) + b * a->resident;
    auto changes = reinterpret_cast<__gm__ uint32_t *>(ctx->changes) + b * kChangeCapacity;
    auto changeSlots = reinterpret_cast<__gm__ uint32_t *>(ctx->changeSlots) + b * kChangeCapacity;
    auto events = reinterpret_cast<__gm__ uint32_t *>(a->eventsSrc) + b * a->maxEvents;
    auto dest = reinterpret_cast<__gm__ uint32_t *>(a->eventsDst) + b * a->maxEvents;
    for (uint32_t i = begin + t; i < end; i += 512) {
        uint32_t r = rank[i], unique = (r & kStagedRankTag), order = (unique ? r : rank[r]) & ~kStagedRankTag;
        uint32_t slot = slots[order], token = ids[i];
        asc_stcg(reinterpret_cast<__gm__ uint32_t *>(ctx->locations) + group + positions[i - begin], slot);
        if (!unique)
            continue;
        uint32_t out = group + order - ctl[2];
        int32_t ord = asc_ldcg(inv + slot);
        // A non-LRU escape slot always uses its extended change record.
        bool regular = ord >= 0 && uint32_t(ord) < kLruSlots &&
                       asc_ldcg(reinterpret_cast<__gm__ uint32_t *>(ctx->lruSlots) + b * kLruSlots + ord) == slot;
        uint32_t record = regular ? uint32_t(ord) : kLruSlots + order;
        if (regular)
            asc_stcg(changes + kLruSlots + order, UINT32_MAX);
        asc_stcg(changes + record, token);
        asc_stcg(changeSlots + record, slot);
        asc_stcg(events + kPrefetch + order, token);
        asc_stcg(dest + kPrefetch + order, slot);
        uint64_t remote = ctx->fallbackAddresses ? asc_ldcg(oldAddresses + positions[i - begin]) : 0;
        StoreFallbackSource(a->history, ctx->shape.tokenBytes, ctx, b, out, token, remote,
                            reinterpret_cast<__gm__ int32_t *>(ctx->sourceBlocks),
                            reinterpret_cast<__gm__ int32_t *>(ctx->sourceOffsets));
        asc_stcg(reinterpret_cast<__gm__ int32_t *>(ctx->destinations) + out, int32_t(base + slot));
    }
    if (!t) {
        if (q == 3)
            asc_stcg(reinterpret_cast<__gm__ uint32_t *>(ctx->changeCount) + b, kLruSlots + ctl[3]);
        if (ctl[3] > ctl[2])
            asc_atomic_add(reinterpret_cast<__gm__ uint32_t *>(a->liveCount), ctl[3] - ctl[2]);
    }
    asc_threadfence();
    asc_syncthreads();
}

__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void CheckPrefetch(__gm__ Args *a, __gm__ Context *ctx, uint32_t b,
                                                                   uint32_t used, __ubuf__ uint32_t *result) {
    auto src = reinterpret_cast<__gm__ uint32_t *>(a->eventsSrc) + b * a->maxEvents;
    auto dst = reinterpret_cast<__gm__ uint32_t *>(a->eventsDst) + b * a->maxEvents;
    auto inv = reinterpret_cast<__gm__ int32_t *>(ctx->reservedHot) + b * a->resident;
    auto mapping = reinterpret_cast<__gm__ uint32_t *>(ctx->mapping) + uint64_t(b) * a->history;
    if (!threadIdx.x)
        result[0] = 0;
    asc_syncthreads();
    for (uint32_t i = threadIdx.x; i < used; i += 512) {
        uint32_t token = asc_ldcg(src + i), slot = asc_ldcg(dst + i);
        if (slot >= a->resident || asc_ldcg(inv + slot) != int32_t(i))
            asc_atomic_or(result, 3U);
        if (token < a->resident || token >= a->history || asc_ldcg(mapping + token) != a->resident + i)
            asc_atomic_or(result, 2U);
    }
    asc_syncthreads();
}

__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void JoinLists(__ubuf__ uint32_t *ids, __ubuf__ uint32_t *banks,
                                                               uint32_t last, uint32_t n0, uint32_t n1,
                                                               uint32_t n2, uint32_t n3) {
    uint32_t n[4] = {n0, n1, n2, n3}, cursor = 0;
    for (uint32_t j = 0; j <= last; ++j) {
        for (uint32_t i = threadIdx.x; i < n[j]; i += 512)
            ids[cursor + i] = banks[j * kTop + i];
        cursor += n[j];
    }
    asc_syncthreads();
}

__aicore__ __attribute__((noinline)) inline Slice Resolve(__gm__ Args *a, uint32_t b, uint32_t q) {
    auto ctx = reinterpret_cast<__gm__ Context *>(a->context);
    auto row = reinterpret_cast<__gm__ Row *>(ctx->rows) + b;
    auto flags = reinterpret_cast<__gm__ BatchFlag *>(a->flags) + b * kFlagCount;
    auto lists = Scratch(a, b);
    auto ready = reinterpret_cast<__gm__ BatchFlag *>(lists + kListFlags);
    auto headers = reinterpret_cast<__gm__ uint32_t *>(a->queryErrors) + (4 * b + q) * 32;
    uint32_t used = row->slots;
    Slice s;
    uint32_t possibleAlias = 0, sizes[4] = {};
    bool packetsReady = true;
    s.error = used > kLruSlots ? 4U : 0U;
    for (uint32_t j = 0; j < 4; ++j) {
        auto hdr = reinterpret_cast<__gm__ uint32_t *>(a->queryErrors) + (4 * b + j) * 32;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(hdr), 64);
        s.error |= hdr[0];
        possibleAlias |= hdr[kQueryAliasWord];
        uint32_t encoded = hdr[kOwnerPacketCountWord];
        if (encoded > kTop + 1)
            s.error |= 4U;
        packetsReady = packetsReady && encoded != 0;
        sizes[j] = encoded ? encoded - 1 : 0;
    }
    auto lru = reinterpret_cast<__gm__ uint32_t *>(ctx->lruSlots) + b * kLruSlots;
    TPipe pipe;
    TBuf<TPosition::VECCALC> storage;
    constexpr uint32_t idWords = kSlots;
    constexpr uint32_t total = idWords + 2 * kSlots + kSlots + kSlots + kTop + kLruSlots + 4096 + 576;
    static_assert(total * 4 <= kFinalScratchBytes);
    pipe.InitBuffer(storage, total * 4);
    auto ids = storage.Get<uint32_t>(), table = ids[idWords], ranks = table[2 * kSlots], slots = ranks[kSlots];
    auto positions = slots[kSlots], victims = positions[kTop], bitmap = victims[kLruSlots], ctl = bitmap[4096];
    auto ptr = FinalPtr(ctl);
    if (!q) {
        if (ctx->rowAddressBase) {
            uint32_t owner = ctx->rowOwners ? reinterpret_cast<__gm__ uint32_t *>(ctx->rowOwners)[b] : b;
            row->addresses = RequestAddressTable(ctx, b, owner, a->history);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(row), 64);
        }
        if (!s.error) {
            Simt::VF_CALL<PublishPrefetch>(
                Simt::Dim3{512, 1, 1}, reinterpret_cast<__gm__ uint32_t *>(ctx->changes) + b * kChangeCapacity,
                lru, reinterpret_cast<__gm__ uint32_t *>(a->eventsSrc) + b * a->maxEvents,
                reinterpret_cast<__gm__ uint32_t *>(a->eventsDst) + b * a->maxEvents, used);
            VectorDone();
        }
        uint32_t check = 0;
        if (used && !s.error) {
            Simt::VF_CALL<CheckPrefetch>(Simt::Dim3{512, 1, 1}, a, ctx, b, used, ptr);
            VectorDone();
            check = ctl.GetValue(0);
        }
        flags->reserved[0] = check;
        Signal(flags, used, s.error);
    }
    const uint32_t last = a->resident < kSlots + kLruSlots ? 3 : q;
    if (!packetsReady) {
        // All owners see the same four LI headers. If a packet is missing,
        // every owner uses the original resolver and may reuse its own bank.
        uint32_t own = 0;
        if (!s.error) {
            Simt::VF_CALL<GatherMisses>(Simt::Dim3{512, 1, 1}, ctx, 4 * b + q, headers[1],
                                        FinalPtr(table[q * kTop]), FinalPtr(positions), ptr);
            VectorDone();
            own = ctl.GetValue(0);
        }
        if (q < 3 || a->resident < kSlots + kLruSlots) {
            BeforeStore();
            ToGm(lists + q * kTop, table[q * kTop], own);
            StoreDone();
        }
        Signal(ready + q, own, s.error);
        for (uint32_t j = 0; j <= last; ++j) {
            if (j != q)
                s.error |= AwaitPublication(ready + j);
            sizes[j] = j == q ? own : ready[j].count;
            if (sizes[j] > kTop)
                s.error |= 4U;
        }
    }
    const uint32_t own = sizes[q];
    uint32_t begin = 0, n = 0;
    for (uint32_t j = 0; j <= last; ++j) {
        if (j < q)
            begin += sizes[j];
        n += sizes[j];
    }
    if (q)
        s.error |= AwaitPublication(flags);
    if (!s.error) {
        if (!packetsReady) {
            for (uint32_t j = 0; j <= last; ++j)
                if (j != q)
                    FromGm(table[j * kTop], lists + j * kTop, sizes[j]);
        }
        FromGm(victims, lru, kLruSlots);
        InputReady();
        if (packetsReady) {
            Simt::VF_CALL<LoadOwnerMissPackets>(Simt::Dim3{512, 1, 1}, lists, FinalPtr(ids), FinalPtr(positions),
                                                q, last, sizes[0], sizes[1], sizes[2], sizes[3]);
        } else {
            Simt::VF_CALL<JoinLists>(Simt::Dim3{512, 1, 1}, FinalPtr(ids), FinalPtr(table), last,
                                     sizes[0], sizes[1], sizes[2], sizes[3]);
        }
        VectorDone();
        uint32_t buckets = 8;
        while (buckets < n * 2)
            buckets *= 2;
        Simt::VF_CALL<PrefixRanks>(Simt::Dim3{512, 1, 1}, FinalPtr(ids), n, begin, begin + own, buckets,
                                   FinalPtr(table), FinalPtr(ranks), ptr);
        VectorDone();
        Simt::VF_CALL<ChooseSlots>(Simt::Dim3{512, 1, 1}, a->resident, used, (flags->reserved[0] & 1U) != 0,
                                   reinterpret_cast<__gm__ int32_t *>(ctx->reservedHot) + b * a->resident,
                                   reinterpret_cast<__gm__ uint32_t *>(a->queryBitmaps) +
                                       b * 4 * ((a->resident + 31) / 32),
                                   FinalPtr(victims), FinalPtr(slots), FinalPtr(bitmap), ptr);
        VectorDone();
        s.error |= ctl.GetValue(1);
        if (!s.error) {
            Simt::VF_CALL<EmitOffsetPlan>(Simt::Dim3{512, 1, 1}, a, ctx, b, q, begin, begin + own, FinalPtr(ids),
                                          FinalPtr(positions), FinalPtr(ranks), FinalPtr(slots), ptr);
            VectorDone();
            s.begin = ctl.GetValue(2);
            s.count = ctl.GetValue(3) - s.begin;
            s.safe = s.count;
            s.escape = ctl.GetValue(5) != 0;
        }
    }
    s.disjoint = possibleAlias == 0 && flags->reserved[0] == 0;
    pipe.Destroy();
    return s;
}
