// SPDX-License-Identifier: Apache-2.0
// Included in rotating, after CompactMisses and ReserveIncremental.
#if KVIP_FETCH_BITMAP || KVIP_FETCH_DENSE
constexpr uint32_t kRankWords = 3072; // 2048 u32 IDs, 2048 u16 keys.
constexpr uint32_t kRankHeader = 32;
constexpr uint32_t kFetchEpochs = 9;
constexpr uint32_t kRankNeedsFetch = 0x80000000U;
// V_MTE2 event 0/1 belongs to the two live LI key-scale banks during
// RotateFetch. Event 3 is balanced locally by LI before entering Service.
constexpr auto kRankScratchReleased = EVENT_ID3;

// Diagnostic-only stores occur after the ordinary rank DMA has completed.
__aicore__ inline void RankTimer(__gm__ Context* ctx, uint32_t query, uint32_t rounds,
                                 uint32_t epoch, uint32_t field, uint64_t ticks) {
    auto header = reinterpret_cast<__gm__ uint64_t*>(ctx->fetchRanks
        + uint64_t(query*rounds+epoch)*(kRankWords+kRankHeader)*sizeof(uint32_t));
    header[field] = ticks;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(header), 64);
}

__simd_vf__ inline void RemapFetchSeed(__ubuf__ uint32_t* ids,uint32_t resident,uint32_t begin) {
    using namespace MicroAPI;
    auto all=CreateMask<uint32_t,MaskPattern::ALL>();
    RegTensor<uint32_t> token,recent; MaskReg mask;
    for(uint16_t i=0;i<2048;i+=64) {
        LoadAlign(token,ids+i);Adds(recent,token,begin-resident,all);
        CompareScalar<uint32_t,CMPMODE::GE>(mask,token,resident,all);
        Select(token,recent,token,mask);StoreAlign(ids+i,token,all);
    }
}

// Each bit represents a 128-score tile. Summarize up to 16K scores from
// the producer bitmap in one DMA, before reusing its input space for scores.
#if KVIP_FETCH_BITMAP
__simd_vf__ inline void FetchTileMask(__ubuf__ uint32_t* words, __ubuf__ uint32_t* result,
                                      uint32_t tiles) {
    using namespace MicroAPI;
    RegTensor<int32_t> lane;
    Arange(lane, int32_t(0));
    RegTensor<uint32_t> weight, one, zero, low, high, sum;
    auto all = CreateMask<uint32_t, MaskPattern::ALL>();
    auto eight = CreateMask<uint32_t, MaskPattern::VL8>();
    auto first = CreateMask<uint32_t, MaskPattern::VL1>();
    Duplicate(one, uint32_t(1));
    Duplicate(zero, uint32_t(0));
    ShiftLeft(weight, one, lane, eight);
    // Eight packed bitmap words describe exactly one 128-score tile. The
    // hardware block maximum reduces eight such tiles per contiguous load.
    for (uint16_t t = 0; t < tiles; t += 16) {
        LoadAlign(low, words + t * 8);
        LoadAlign(high, words + t * 8 + 64);
        ReduceMaxWithDataBlock(low, low, all);
        ReduceMaxWithDataBlock(high, high, all);
        MaskReg lowHit, highHit;
        CompareScalar<uint32_t, CMPMODE::NE>(lowHit, low, uint32_t(0), eight);
        CompareScalar<uint32_t, CMPMODE::NE>(highHit, high, uint32_t(0), eight);
        Select(low, weight, zero, lowHit);
        Select(high, weight, zero, highHit);
        ShiftLefts(high, high, int16_t(8), eight);
        Or(low, low, high, eight);
        ReduceSum(sum, low, eight);
        StoreAlign(result + (t / 16) * 8, sum, first);
    }
}
#endif

template<bool clipped>
__simd_vf__ inline void PackRankOffsets(__ubuf__ uint16_t* raw, __ubuf__ uint16_t* offsets,
    uint16_t cutoff, uint16_t valid, uint16_t skipLo, uint16_t skipHi,
    uint16_t tiles, uint32_t tileMask) {
    using namespace MicroAPI;
    auto all = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<uint16_t> value, packed;
    RegTensor<int16_t> lanes;
    MaskReg keep;
    UnalignRegForStore writer;
    ClearSpr<SpecialPurposeReg::AR>();
    for (uint16_t tile = 0; tile < tiles; ++tile) {
#if KVIP_FETCH_BITMAP
        if (!(tileMask & (1u << tile))) continue;
#endif
        const uint16_t i = tile * 128;
        LoadAlign(value, raw + i);
        Arange(lanes, int16_t(i));
        auto& index = reinterpret_cast<RegTensor<uint16_t>&>(lanes);
        CompareScalar<uint16_t, CMPMODE::GT>(keep, value, cutoff, all);
        if constexpr (clipped) {
            MaskReg causal, left, right;
            CompareScalar<uint16_t, CMPMODE::LT>(causal, index, valid, all);
            CompareScalar<uint16_t, CMPMODE::LT>(left, index, skipLo, all);
            CompareScalar<uint16_t, CMPMODE::GE>(right, index, skipHi, all);
            MaskOr(left, left, right, all);
            MaskAnd(keep, keep, causal, all);
            MaskAnd(keep, keep, left, all);
        }
        Squeeze<uint16_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(packed, index, keep);
        StoreUnAlign<uint16_t,PostLiteral::POST_MODE_UPDATE>(offsets, packed, writer);
    }
    StoreUnAlignPost(offsets, writer);
}

// Widen packed u16 keys/offsets using aligned word gathers. Keep all rank
// maintenance on the vector pipe, including the common short tie update.
__simd_callee__ inline void RankGather16(MicroAPI::RegTensor<uint32_t>& result,
    __ubuf__ uint16_t* input, MicroAPI::RegTensor<uint32_t>& index) {
    using namespace MicroAPI;
    auto all = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<uint32_t> wordIndex, word, high, odd, one, mask;
    Duplicate(one, uint32_t(1)); Duplicate(mask, uint32_t(65535));
    ShiftRights(wordIndex, index, int16_t(1), all);
    Gather(word, reinterpret_cast<__ubuf__ uint32_t*>(input), wordIndex, all);
    ShiftRights(high, word, int16_t(16), all);
    And(odd, index, one, all);
    MaskReg upper;
    CompareScalar<uint32_t, CMPMODE::NE>(upper, odd, uint32_t(0), all);
    Select(result, high, word, upper); And(result, result, mask, all);
}

__simd_callee__ inline void RankPhysical(MicroAPI::RegTensor<uint32_t>& id,
    uint32_t reverseBegin, uint32_t reverseEnd) {
    using namespace MicroAPI;
    auto all = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<uint32_t> block, lane, lowMask, highMask, sum, reversed;
    Duplicate(lowMask, uint32_t(2047)); Duplicate(highMask, uint32_t(~2047U));
    Duplicate(sum, reverseBegin + reverseEnd - 2048);
    And(block, id, highMask, all); And(lane, id, lowMask, all);
    Sub(reversed, sum, block, all); Add(reversed, reversed, lane, all);
    MaskReg lo, hi, inside;
    CompareScalar<uint32_t, CMPMODE::GE>(lo, id, reverseBegin, all);
    CompareScalar<uint32_t, CMPMODE::LT>(hi, id, reverseEnd, all);
    MaskAnd(inside, lo, hi, all); Select(id, reversed, id, inside);
}

// Accumulate a block's sparse scores and logical offsets without updating K.
// Destination masks handle arbitrary append offsets without scalar stores.
__simd_vf__ inline void CollectRankBlock(__ubuf__ uint16_t* raw, __ubuf__ uint16_t* offsets,
    __ubuf__ uint16_t* values, __ubuf__ uint16_t* positions,
    uint32_t pending, uint32_t count, uint32_t scanOffset) {
    using namespace MicroAPI;
    auto all = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<uint32_t> zero; Duplicate(zero, uint32_t(0));
    for (uint16_t base = pending / 64 * 64; base < pending + count; base += 64) {
        RegTensor<int32_t> sequence; Arange(sequence, int32_t(base));
        auto& lane = reinterpret_cast<RegTensor<uint32_t>&>(sequence);
        RegTensor<uint32_t> source, offset, value, logical;
        MaskReg lo, hi, write;
        CompareScalar<uint32_t, CMPMODE::GE>(lo, lane, pending, all);
        CompareScalar<uint32_t, CMPMODE::LT>(hi, lane, pending + count, all);
        MaskAnd(write, lo, hi, all);
        Adds(source, lane, uint32_t(0) - pending, all);
        Select(source, source, zero, write);
        RankGather16(offset, offsets, source); RankGather16(value, raw, offset);
        Adds(logical, offset, scanOffset, all);
        StoreAlign<uint16_t, StoreDist::DIST_PACK_B32>(values + base,
            reinterpret_cast<RegTensor<uint16_t>&>(value), write);
        StoreAlign<uint16_t, StoreDist::DIST_PACK_B32>(positions + base,
            reinterpret_cast<RegTensor<uint16_t>&>(logical), write);
    }
}

template<bool select>
__simd_vf__ inline void WriteWindowRank(__ubuf__ uint32_t* output, __ubuf__ uint16_t* outputKeys,
    __ubuf__ uint32_t* previous, __ubuf__ uint16_t* values, __ubuf__ uint16_t* ordinals,
    __ubuf__ uint16_t* positions, uint32_t count, uint32_t strict,
    uint32_t scanBegin, uint32_t reverseBegin, uint32_t reverseEnd,
    uint32_t tailBegin, uint32_t tailEnd, bool frozenMap) {
    using namespace MicroAPI;
    auto all = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<uint32_t> zero, last;
    Duplicate(zero, uint32_t(0)); Duplicate(last, kTop - 1 + strict);
    uint32_t begin = select ? 0 : strict / 64 * 64;
    uint32_t end = select ? kTop : strict + count;
    for (uint16_t base = begin; base < end; base += 64) {
        RegTensor<int32_t> sequence; Arange(sequence, int32_t(base));
        auto& lane = reinterpret_cast<RegTensor<uint32_t>&>(sequence);
        RegTensor<uint32_t> ordinal, key, source, token, oldIndex, oldId;
        MaskReg old, write;
        if constexpr (select) {
            RegTensor<uint32_t> normal;
            Sub(normal, last, lane, all);
            CompareScalar<uint32_t, CMPMODE::LT>(old, lane, strict, all);
            Select(normal, lane, normal, old);
            RankGather16(ordinal, ordinals, normal);
            RankGather16(key, values, ordinal);
            CompareScalar<uint32_t, CMPMODE::LT>(old, ordinal, kTop, all);
            Select(oldIndex, ordinal, zero, old);
            Gather(oldId, previous, oldIndex, all);
            Adds(source, ordinal, uint32_t(0) - kTop, all);
            Select(source, zero, source, old);
            write = all;
        } else {
            MaskReg lo, hi;
            CompareScalar<uint32_t, CMPMODE::GE>(lo, lane, strict, all);
            CompareScalar<uint32_t, CMPMODE::LT>(hi, lane, end, all);
            MaskAnd(write, lo, hi, all);
            Adds(source, lane, uint32_t(0) - strict, all);
            Select(source, source, zero, write);
            RankGather16(key, values, source);
        }
        RankGather16(token, positions, source);
        Adds(token, token, scanBegin, all);
        RankPhysical(token, reverseBegin, reverseEnd);
        RegTensor<uint32_t> tagged, flag;
        Duplicate(flag, kRankNeedsFetch); Or(tagged, token, flag, all);
        MaskReg left, right, outside;
        CompareScalar<uint32_t, CMPMODE::LT>(left, token, tailBegin, all);
        CompareScalar<uint32_t, CMPMODE::GE>(right, token, tailEnd, all);
        MaskOr(outside, left, right, all);
        if (frozenMap) token = tagged;
        else Select(token, tagged, token, outside);
        if constexpr (select) Select(token, oldId, token, old);
        StoreAlign(output + base, token, write);
        StoreAlign<uint16_t, StoreDist::DIST_PACK_B32>(outputKeys + base,
            reinterpret_cast<RegTensor<uint16_t>&>(key), write);
    }
}

template<typename T> __aicore__ inline __ubuf__ T* RankPtr(LocalTensor<T> tensor) {
    return reinterpret_cast<__ubuf__ T*>(tensor.GetPhyAddr());
}

__simd_vf__ inline void PadRankKeys(__ubuf__ uint16_t* values, uint32_t begin, uint32_t end) {
    using namespace MicroAPI;
    auto all = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<uint16_t> value, zero; Duplicate(zero, uint16_t(0));
    for (uint16_t base = begin / 128 * 128; base < end; base += 128) {
        RegTensor<int16_t> sequence; Arange(sequence, int16_t(base));
        MaskReg keep;
        CompareScalar<uint16_t, CMPMODE::LT>(keep,
            reinterpret_cast<RegTensor<uint16_t>&>(sequence), uint16_t(begin), all);
        LoadAlign(value, values + base); Select(value, value, zero, keep);
        StoreAlign(values + base, value, all);
    }
}

// Merge once per completed window in the sparse case. A full 2048-entry
// candidate bank flushes early, then the current block is filtered again
// against the raised cutoff. Capacity never discards a candidate.
__aicore__ inline void MergeRankWindow(LocalTensor<uint32_t> arena, uint32_t scanBegin,
    kvip_cold_scan::Range order, uint32_t added, uint16_t& kth, uint32_t& strict,
    uint32_t tailBegin, uint32_t tailEnd, bool frozenMap) {
    if (!added) return;
    auto ids = arena, nextIds = arena[2048];
    auto keys = arena[4096].ReinterpretCast<uint16_t>();
    auto nextKeys = arena[5120].ReinterpretCast<uint16_t>();
    auto values = arena[7168].ReinterpretCast<uint16_t>();
    auto offsets = arena[8224].ReinterpretCast<uint16_t>();
    if (strict + added < kTop) {
        WriteWindowRank<false>(RankPtr(ids), RankPtr(keys), RankPtr(ids), RankPtr(values),
            RankPtr(offsets), RankPtr(offsets), added, strict, scanBegin, order.begin, order.end, tailBegin, tailEnd, frozenMap);
        PipeBarrier<PIPE_V>();
        strict += added;
        return;
    }
    auto input = arena[6144].ReinterpretCast<uint16_t>();
    auto cutoff = arena[9472].ReinterpretCast<uint16_t>();
    auto ord = keys;
    uint32_t size = (kTop + added + 255) / 256 * 256;
    DataCopy(input, keys, kTop);
    PadRankKeys(RankPtr(values), added, size - kTop);
    PipeBarrier<PIPE_V>();
    kvip_recent_threshold::RegisterThreshold(RankPtr(input), RankPtr(cutoff), size/256, size-kTop+1);
    PipeBarrier<PIPE_V>();
    auto cutoff32 = reinterpret_cast<__ubuf__ uint32_t*>(cutoff.GetPhyAddr());
    topkb16gather::FindIdxGTOutputVFImpl(RankPtr(ord), RankPtr(input), 0, cutoff32, size/128);
    SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
    strict = GetSpr<SpecialPurposeReg::AR>() / sizeof(uint16_t);
    topkb16gather::FindIdxEQOutputVFImpl(RankPtr(ord), RankPtr(input), 0, cutoff32, size/128);
    PipeBarrier<PIPE_V>();
    WriteWindowRank<true>(RankPtr(nextIds), RankPtr(nextKeys), RankPtr(ids), RankPtr(input),
        RankPtr(ord), RankPtr(offsets), kTop, strict, scanBegin, order.begin, order.end, tailBegin, tailEnd, frozenMap);
    PipeBarrier<PIPE_V>();
    DataCopy(keys, nextKeys, kTop); DataCopy(ids, nextIds, kTop);
    SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
    kth = cutoff.GetValue(0);
}

// Residency and victim eligibility are immutable throughout this LI call.
// Compute the resident tags once, together with the seed tie permutation.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void SeedRankIds(__gm__ Context* ctx, uint32_t batch, uint32_t resident,
    __ubuf__ uint32_t* source, __ubuf__ uint32_t* destination, uint32_t strict, bool frozenMap) {
    for (uint32_t i = threadIdx.x; i < kTop; i += 256) {
        uint32_t id = source[i < strict ? i : kTop - 1 + strict - i];
        bool needs = frozenMap || ((id >= resident || !HotValid(resident,ctx,batch,id) ||
            (ctx->reservedHot && asc_ldcg(reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)
                + batch*resident+id) >= 0)) && !IsTailToken(ctx,batch,id));
        destination[i] = id | (needs ? kRankNeedsFetch : 0);
    }
}

__aicore__ inline uint16_t ClipRankOffset(uint32_t value,uint32_t base,uint32_t n) {
    return value<=base?0:uint16_t(value-base<n?value-base:n);
}

template<bool profile>
__aicore__ inline uint32_t AdvanceFetchRank(__gm__ Context* ctx, LocalTensor<uint32_t> a,
    GM_ADDR scoreBuffer, uint32_t q, uint32_t history, uint32_t resident, uint32_t rounds,
    uint32_t epoch, uint32_t queryEnd, kvip_cold_scan::Range order, bool frozenMap) {
    uint64_t entry = 0, loaded = 0, mergeTicks = 0;
    uint32_t blockCount = 0, candidateCount = 0, mergeCount = 0;
    if constexpr (profile) entry = GetSystemCycle();
    GlobalTensor<uint32_t> store;
    store.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->fetchRanks)
        + uint64_t(q*rounds+epoch)*(kRankWords+kRankHeader));
    auto keys=a[4096].ReinterpretCast<uint16_t>();
    uint32_t strict=0;
    uint16_t kth=0;
    uint32_t tailBegin = 0, tailEnd = 0;
    if (ctx->tailBounds) {
        auto tail = reinterpret_cast<__gm__ uint32_t*>(ctx->tailBounds) + (q/4)*2;
        tailBegin = tail[0]; tailEnd = tail[1];
    }
    if(!epoch) {
        // The seed is saved in ordinary native order by its producer.
        auto bytes=reinterpret_cast<__gm__ uint8_t*>(ctx->hits) -
                   ctx->shape.batch*(4*kTop*6+1024) + (q/4)*(4*kTop*6+1024);
        GlobalTensor<uint32_t> seedIds; GlobalTensor<uint16_t> seedKeys;
        seedIds.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(bytes+4*kTop*2)+(q%4)*kTop);
        seedKeys.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(bytes)+(q%4)*kTop);
        DataCopy(a,seedIds,kTop);DataCopy(keys,seedKeys,kTop);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        kth=keys.GetValue(kTop-1);
        uint32_t end=kTop;
        while(strict<end) {
            uint32_t mid=(strict+end)/2;
            if(keys.GetValue(mid)>kth)strict=mid+1; else end=mid;
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Simt::VF_CALL<SeedRankIds>(Simt::Dim3{256,1,1}, ctx, q/4, resident,
            RankPtr(a), RankPtr(a[2048]), strict, frozenMap);
        PipeBarrier<PIPE_V>();
        DataCopy(a,a[2048],kTop);
        PipeBarrier<PIPE_V>();
        if constexpr (profile) {
            SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
            loaded = GetSystemCycle();
        }
    } else {
        GlobalTensor<uint32_t> old;
        old.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->fetchRanks)
            + uint64_t(q*rounds+epoch-1)*(kRankWords+kRankHeader));
        auto hdr=a[9792];
        DataCopy(hdr,old,kRankHeader);DataCopy(a,old[kRankHeader],kTop);
        DataCopy(keys,old[kRankHeader+kTop].ReinterpretCast<uint16_t>(),kTop);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        kth=hdr.GetValue(0);strict=hdr.GetValue(1);
        if constexpr (profile) loaded = GetSystemCycle();
        GlobalTensor<uint16_t> scores;
        scores.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(scoreBuffer)+uint64_t(q)*history);
        GlobalTensor<uint32_t> bits;
        bits.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->hits)+uint64_t(q)*history/16);
        auto ranges=reinterpret_cast<__gm__ uint32_t*>(ctx->thresholdBounds)+q/4*2;
        uint32_t lo=ranges[0],hi=ranges[1];
        uint32_t stop=Trigger(history,resident,rounds,epoch)*128;
        auto raw=a[6144].ReinterpretCast<uint16_t>();
        auto off = a[2048].ReinterpretCast<uint16_t>(); // idle next-ID bank
        auto values = a[7168].ReinterpretCast<uint16_t>();
        auto offsets = a[8224].ReinterpretCast<uint16_t>();
        const uint32_t start = Trigger(history,resident,rounds,epoch-1)*128;
        uint32_t pending = 0;
#if KVIP_FETCH_BITMAP
        auto masks = a[9632]; // eight aligned tile-mask words, disjoint from the rank.
        uint32_t batchStart = 0, batchEnd = 0;
#endif
        for(uint32_t scan=Trigger(history,resident,rounds,epoch-1)*128;scan<stop;) {
            uint32_t n=2048-(scan%2048);if(n>stop-scan)n=stop-scan;
            uint32_t p=kvip_cold_scan::Remap(scan,order);
            uint32_t valid=ClipRankOffset(queryEnd,p,n);
            if(valid && !(p>=lo && p+valid<=hi)) {
                uint32_t tileMask = (1u << (n / 128)) - 1;
#if KVIP_FETCH_BITMAP
                if (scan >= batchEnd) {
                    batchStart = scan - scan % 2048;
                    batchEnd = stop - batchStart > 16384 ? batchStart + 16384 : stop;
                    uint32_t tiles = (batchEnd - batchStart) / 128;
                    auto words = a[6144];
                    DataCopy(words, bits[batchStart / 16], tiles * 8);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    FetchTileMask(RankPtr(words), RankPtr(masks), tiles);
                    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                }
                uint32_t tile = (scan - batchStart) / 128;
                tileMask &= masks.GetValue((tile / 16) * 8) >> (tile % 16);
                if (!tileMask) { scan += n; continue; }
                for (uint32_t runs = tileMask; runs;) {
                    uint32_t first = __builtin_ctz(runs);
                    uint32_t length = __builtin_ctz(~(runs >> first));
                    DataCopy(raw[first * 128], scores[p + first * 128], length * 128);
                    runs &= ~(((1u << length) - 1) << first);
                }
#else
                DataCopy(raw,scores[p],n);
#endif
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                if (valid == n && (p >= hi || p + n <= lo)) {
                    PackRankOffsets<false>(RankPtr(raw), RankPtr(off), kth, n, 0, 0, n/128, tileMask);
                } else {
                    PackRankOffsets<true>(RankPtr(raw), RankPtr(off), kth, valid,
                        ClipRankOffset(lo,p,n),ClipRankOffset(hi,p,n),n/128,tileMask);
                }
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                uint32_t added = GetSpr<SpecialPurposeReg::AR>() / sizeof(uint16_t);
                if constexpr (profile) ++blockCount;
                if (pending + added > kTop) {
                    uint64_t begin = 0;
                    if constexpr (profile) begin = GetSystemCycle();
                    MergeRankWindow(a,start,order,pending,kth,strict,tailBegin,tailEnd,frozenMap);
                    if constexpr (profile) {
                        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
                        mergeTicks += GetSystemCycle() - begin;
                        ++mergeCount;
                    }
                    pending = 0;
                    SetFlag<HardEvent::V_MTE2>(kRankScratchReleased);
                    WaitFlag<HardEvent::V_MTE2>(kRankScratchReleased);
                    continue; // merge reused raw/offset scratch; reload this block
                }
                if (added) {
                    CollectRankBlock(RankPtr(raw),RankPtr(off),RankPtr(values),RankPtr(offsets),
                                     pending,added,scan-start);
                    pending += added;
                    if constexpr (profile) candidateCount += added;
                }
                // The next DMA reuses raw after both the sieve and merge readers.
                SetFlag<HardEvent::V_MTE2>(kRankScratchReleased);
                WaitFlag<HardEvent::V_MTE2>(kRankScratchReleased);
            }
            scan+=n;
        }
        uint64_t begin = 0;
        if constexpr (profile) begin = GetSystemCycle();
        MergeRankWindow(a,start,order,pending,kth,strict,tailBegin,tailEnd,frozenMap);
        if constexpr (profile) {
            SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
            mergeTicks += GetSystemCycle() - begin;
            mergeCount += pending != 0;
        }
    }
    uint64_t storeBegin = 0;
    auto hdr=a[9792]; hdr.SetValue(0,kth);hdr.SetValue(1,strict);
    if constexpr (profile) {
        auto times = hdr.ReinterpretCast<uint64_t>();
        uint64_t done = GetSystemCycle();
        times.SetValue(2, loaded - entry);
        times.SetValue(3, done - loaded - mergeTicks);
        times.SetValue(4, mergeTicks);
        hdr.SetValue(18, blockCount); hdr.SetValue(19, candidateCount);
        hdr.SetValue(20, mergeCount);
        storeBegin = GetSystemCycle();
    }
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(store,hdr,kRankHeader);DataCopy(store[kRankHeader],a,kTop);
    DataCopy(store[kRankHeader+kTop].ReinterpretCast<uint16_t>(),keys,kTop);
    // Rank storage is visible on return, and the next VF can reuse scratch in place.
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    if constexpr (profile) RankTimer(ctx,q,rounds,epoch,5,GetSystemCycle()-storeBegin);
    return strict;
}

// Filter immutable hits on-chip before any dynamic mapping reads. Query
// order and unsent winners are retained for the existing Q4 reservation rule.
__simd_vf__ inline void RankEligible(__ubuf__ uint32_t* rank, __ubuf__ uint32_t* output,
                                    uint32_t strict) {
    using namespace MicroAPI;
    auto all = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<uint32_t> last, lowMask;
    Duplicate(last,kTop-1+strict); Duplicate(lowMask,~kRankNeedsFetch);
    UnalignRegForStore writer; ClearSpr<SpecialPurposeReg::AR>();
    for (uint16_t base=0; base<kTop; base+=64) {
        RegTensor<int32_t> sequence; Arange(sequence,int32_t(base));
        auto& lane=reinterpret_cast<RegTensor<uint32_t>&>(sequence);
        RegTensor<uint32_t> index,id,packed;
        MaskReg prefix,eligible;
        Sub(index,last,lane,all);
        CompareScalar<uint32_t,CMPMODE::LT>(prefix,lane,strict,all);
        Select(index,lane,index,prefix); Gather(id,rank,index,all);
        CompareScalar<uint32_t,CMPMODE::GE>(eligible,id,kRankNeedsFetch,all);
        And(id,id,lowMask,all);
        Squeeze<uint32_t,MicroAPI::GatherMaskMode::STORE_REG>(packed,id,eligible);
        StoreUnAlign<uint32_t,PostLiteral::POST_MODE_UPDATE>(output,packed,writer);
    }
    StoreUnAlignPost(output,writer);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void RankMap(__ubuf__ uint32_t* ids, __ubuf__ uint32_t* states,
                   __gm__ uint32_t* map, uint32_t count) {
    for (uint32_t i=threadIdx.x;i<count;i+=256) states[i]=asc_ldcg(map+ids[i]);
}
__aicore__ inline uint32_t AdmitLocalRank(__gm__ Context* ctx, LocalTensor<uint32_t> scratch,
    LocalTensor<uint32_t> cfg, LocalTensor<uint32_t> selectedIds, __gm__ uint32_t* map,
    uint32_t b, uint32_t rt_resident, uint32_t strict, uint32_t allowance,
    uint32_t slots, uint32_t& missing, bool frozenMap) {
    RankEligible(RankPtr(scratch),RankPtr(scratch[2048]),strict);
    SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
    uint32_t eligible=GetSpr<SpecialPurposeReg::AR>()/sizeof(uint32_t);
    if (!eligible) return 0;
    Simt::VF_CALL<RankMap>(Simt::Dim3{256,1,1},RankPtr(scratch[2048]),
                          RankPtr(scratch[4096]),map,eligible);
    PipeBarrier<PIPE_V>();
    CompactMisses(reinterpret_cast<__ubuf__ uint32_t*>(scratch[2048].GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(scratch[4096].GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),eligible,allowance,
        reinterpret_cast<__ubuf__ uint32_t*>(selectedIds.GetPhyAddr()));
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    uint32_t found=cfg.GetValue(56),n=found<allowance?found:allowance;
    if(n) {
        Simt::VF_CALL<ReserveIncremental>(Simt::Dim3{256,1,1},
            reinterpret_cast<__ubuf__ uint32_t*>(selectedIds.GetPhyAddr()),
            map,n,rt_resident+slots);
        PipeBarrier<PIPE_V>();
    }
    missing += found;
    return n;
}

// A rotating service core owns all four ranks for its request. Advance the
// last visible epoch and admit directly from each query's live UB state.
// An unchanged frontier may still have unsent winners from an earlier budget.
template<bool profile = false>
__aicore__ inline uint32_t ServiceFetchRank(__gm__ Context* ctx, LocalTensor<uint32_t> scratch,
    LocalTensor<uint32_t> cfg, LocalTensor<uint32_t> selectedIds, GM_ADDR scores, __gm__ uint32_t* map,
    uint32_t b, uint32_t history, uint32_t resident, uint32_t rounds, uint32_t versions,
    uint32_t target, const uint32_t* ends, kvip_cold_scan::Range order, uint32_t allowance,
    uint32_t slots, uint32_t& missing, bool frozenMap) {
    uint32_t count = 0;
    for (uint32_t v = versions; v < target; ++v) {
        for (uint32_t q = 0; q < 4; ++q) {
            uint32_t strict = AdvanceFetchRank<profile>(ctx, scratch, scores, b*4+q, history, resident,
                                               rounds, v, ends[q], order, frozenMap);
            uint64_t admitBegin = 0;
            if constexpr (profile) admitBegin = GetSystemCycle();
            if (v+1 == target && count < allowance) {
                count += AdmitLocalRank(ctx, scratch, cfg, selectedIds[count], map, b, resident,
                                        strict, allowance-count, slots+count, missing, frozenMap);
            }
            if constexpr (profile) RankTimer(ctx,b*4+q,rounds,v,6,GetSystemCycle()-admitBegin);
        }
    }
    if (versions == target && target) {
        for (uint32_t q = 0; q < 4 && count < allowance; ++q) {
            GlobalTensor<uint32_t> bank;
            bank.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->fetchRanks)
                + uint64_t((b*4+q)*rounds+target-1)*(kRankWords+kRankHeader));
            DataCopy(scratch, bank[kRankHeader], kTop);
            DataCopy(scratch[4096], bank, kRankHeader);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            uint32_t strict = scratch.GetValue(4097);
            count += AdmitLocalRank(ctx, scratch, cfg, selectedIds[count], map, b, resident, strict,
                                    allowance-count, slots+count, missing, frozenMap);
        }
    }
    return count;
}
#endif
