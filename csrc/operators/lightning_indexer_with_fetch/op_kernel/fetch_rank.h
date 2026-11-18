// SPDX-License-Identifier: Apache-2.0
// Included in rotating, after CompactMisses and ReserveIncremental.
#if KVIP_FETCH_BITMAP || KVIP_FETCH_DENSE
constexpr uint32_t kRankWords = 3072; // 2048 u32 IDs, 2048 u16 keys.
constexpr uint32_t kRankHeader = 32;
constexpr uint32_t kFetchEpochs = 9;
// V_MTE2 event 0/1 belongs to the two live LI key-scale banks during
// RotateFetch. Event 3 is balanced locally by LI before entering Service.
constexpr auto kRankScratchReleased = EVENT_ID3;

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

// Only the overflow path needs a contiguous score array for the histogram.
// Read the winners by their compact offsets and initialize histogram padding.
__simd_vf__ inline void GatherRankKeys(__ubuf__ uint16_t* raw, __ubuf__ uint16_t* offsets,
                                      __ubuf__ uint16_t* out, uint32_t count) {
    using namespace MicroAPI;
    auto all = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<uint16_t> pos, value, zero;
    Duplicate(zero, uint16_t(0));
    uint16_t tiles = ((count + 255) / 256) * 2;
    for (uint16_t i = 0; i < tiles; ++i) {
        auto valid = UpdateMask<uint16_t>(count);
        LoadAlign(pos, offsets + i * 128);
        Gather(value, raw, pos, valid);
        Select(value, value, zero, valid);
        StoreAlign(out + i * 128, value, all);
    }
}

template<bool gather>
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void WriteRank(__ubuf__ uint32_t* out, __ubuf__ uint16_t* outKeys,
    __ubuf__ uint32_t* old, __ubuf__ uint16_t* values, __ubuf__ uint16_t* ord,
    __ubuf__ uint16_t* offsets, uint32_t physical, uint32_t count, uint32_t strict) {
    for (uint32_t i = threadIdx.x; i < count; i += 256) {
        if constexpr (gather) {
            uint32_t j = ord[i], dst = i < strict ? i : kTop - 1 - (i - strict);
            out[dst] = j < kTop ? old[j] : physical + offsets[j-kTop];
            outKeys[dst] = values[j];
        } else {
            out[strict+i] = physical + offsets[i];
            outKeys[strict+i] = values[offsets[i]];
        }
    }
}

template<typename T> __aicore__ inline __ubuf__ T* RankPtr(LocalTensor<T> tensor) {
    return reinterpret_cast<__ubuf__ T*>(tensor.GetPhyAddr());
}

// Up to 2048 strict additions per contiguous source block. Previous K precede
// new entries at a tied boundary. A block merge preserves the ordered window
// result because the cutoff never decreases.
__aicore__ inline void MergeRankBlock(LocalTensor<uint32_t> arena, uint32_t physical,
                                     uint32_t added, uint16_t& kth, uint32_t& strict) {
    if (!added) return;
    auto ids = arena, nextIds = arena[2048];
    auto keys = arena[4096].ReinterpretCast<uint16_t>();
    auto nextKeys = arena[5120].ReinterpretCast<uint16_t>();
    auto values = arena[7168].ReinterpretCast<uint16_t>();
    auto offsets = arena[8224].ReinterpretCast<uint16_t>();
    auto raw = arena[6144].ReinterpretCast<uint16_t>();
    // Store cutoff ties backwards, so appending strict winners touches only the
    // discarded end of the tie cohort. Ties are converted back at publication.
    if (strict + added < kTop) {
        Simt::VF_CALL<WriteRank<false>>(Simt::Dim3{256,1,1}, RankPtr(ids), RankPtr(keys),
            RankPtr(ids), RankPtr(raw), RankPtr(offsets), RankPtr(offsets), physical, added, strict);
        PipeBarrier<PIPE_V>();
        strict += added;
        return;
    }
    // There are now K values strictly above the old cutoff. Its reversed tie
    // suffix is entirely discarded, so its internal order is irrelevant.
    auto input = arena[6144].ReinterpretCast<uint16_t>(); // contiguous with values
    auto cutoff = arena[9472].ReinterpretCast<uint16_t>();
    auto ord = keys; // previous scores are copied before reusing this space
    uint32_t size = (kTop + added + 255) / 256 * 256;
    GatherRankKeys(RankPtr(raw), RankPtr(offsets), RankPtr(values), added);
    PipeBarrier<PIPE_V>(); // Complete the gather before replacing raw with old scores.
    DataCopy(input,keys,kTop);
    PipeBarrier<PIPE_V>();
    kvip_recent_threshold::RegisterThreshold(RankPtr(input),RankPtr(cutoff),size/256,size-kTop+1);
    PipeBarrier<PIPE_V>();
    auto cutoff32=reinterpret_cast<__ubuf__ uint32_t*>(cutoff.GetPhyAddr());
    topkb16gather::FindIdxGTOutputVFImpl(RankPtr(ord),RankPtr(input),0,cutoff32,size/128);
    strict=GetSpr<SpecialPurposeReg::AR>()/sizeof(uint16_t);
    topkb16gather::FindIdxEQOutputVFImpl(RankPtr(ord),RankPtr(input),0,cutoff32,size/128);
    PipeBarrier<PIPE_V>();
    Simt::VF_CALL<WriteRank<true>>(Simt::Dim3{256,1,1}, RankPtr(nextIds), RankPtr(nextKeys),
        RankPtr(ids), RankPtr(input), RankPtr(ord), RankPtr(offsets), physical, kTop, strict);
    PipeBarrier<PIPE_V>();
    DataCopy(keys,nextKeys,kTop); DataCopy(ids,nextIds,kTop);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    kth=cutoff.GetValue(0);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void ReverseSeedTies(__ubuf__ uint32_t* ids, uint32_t strict) {
    for (uint32_t i = threadIdx.x; i < (kTop-strict)/2; i += 256) {
        uint32_t l=strict+i, r=kTop-1-i, id=ids[l]; ids[l]=ids[r];ids[r]=id;
    }
}

__aicore__ inline uint16_t ClipRankOffset(uint32_t value,uint32_t base,uint32_t n) {
    return value<=base?0:uint16_t(value-base<n?value-base:n);
}

__aicore__ inline uint32_t AdvanceFetchRank(__gm__ Context* ctx, LocalTensor<uint32_t> a,
    GM_ADDR scoreBuffer, uint32_t q, uint32_t history, uint32_t resident, uint32_t rounds,
    uint32_t epoch, uint32_t queryEnd, kvip_cold_scan::Range order) {
    GlobalTensor<uint32_t> store;
    store.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->fetchRanks)
        + uint64_t(q*rounds+epoch)*(kRankWords+kRankHeader));
    auto keys=a[4096].ReinterpretCast<uint16_t>();
    uint32_t strict=0;
    uint16_t kth=0;
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
        Simt::VF_CALL<ReverseSeedTies>(Simt::Dim3{256,1,1}, RankPtr(a), strict);
        PipeBarrier<PIPE_V>();
    } else {
        GlobalTensor<uint32_t> old;
        old.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->fetchRanks)
            + uint64_t(q*rounds+epoch-1)*(kRankWords+kRankHeader));
        auto hdr=a[9792];
        DataCopy(hdr,old,kRankHeader);DataCopy(a,old[kRankHeader],kTop);
        DataCopy(keys,old[kRankHeader+kTop].ReinterpretCast<uint16_t>(),kTop);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        kth=hdr.GetValue(0);strict=hdr.GetValue(1);
        GlobalTensor<uint16_t> scores;
        scores.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(scoreBuffer)+uint64_t(q)*history);
        GlobalTensor<uint32_t> bits;
        bits.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->hits)+uint64_t(q)*history/16);
        auto ranges=reinterpret_cast<__gm__ uint32_t*>(ctx->thresholdBounds)+q/4*2;
        uint32_t lo=ranges[0],hi=ranges[1];
        uint32_t stop=Trigger(history,resident,rounds,epoch)*128;
        auto raw=a[6144].ReinterpretCast<uint16_t>();
        auto off = a[8224].ReinterpretCast<uint16_t>();
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
                if (added) MergeRankBlock(a,p,added,kth,strict);
                // The next DMA reuses raw after both the sieve and merge readers.
                SetFlag<HardEvent::V_MTE2>(kRankScratchReleased);
                WaitFlag<HardEvent::V_MTE2>(kRankScratchReleased);
            }
            scan+=n;
        }
    }
    auto hdr=a[9792]; hdr.SetValue(0,kth);hdr.SetValue(1,strict);
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(store,hdr,kRankHeader);DataCopy(store[kRankHeader],a,kTop);
    DataCopy(store[kRankHeader+kTop].ReinterpretCast<uint16_t>(),keys,kTop);
    // Rank storage is visible on return, and the next VF can reuse scratch in place.
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    return strict;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void RankMisses(__gm__ Context* ctx,uint32_t b,uint32_t resident,
    __ubuf__ uint32_t* internal,uint32_t strict,__ubuf__ uint32_t* ids,
    __ubuf__ uint32_t* states,__gm__ uint32_t* map, bool frozenMap) {
    for(uint32_t i=threadIdx.x;i<kTop;i+=256) {
        uint32_t id=internal[i<strict?i:kTop-1-(i-strict)];
        if(frozenMap) { ids[i]=id; states[i]=asc_ldcg(map+id);continue; }
        bool eligible=id>=resident || !HotValid(resident,ctx,b,id) ||
            (ctx->reservedHot && asc_ldcg(reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+b*resident+id)>=0);
        ids[i]=id;
        states[i]=eligible && !IsTailToken(ctx,b,id)?asc_ldcg(map+id):0;
    }
}
__aicore__ inline uint32_t AdmitLocalRank(__gm__ Context* ctx, LocalTensor<uint32_t> scratch,
    LocalTensor<uint32_t> cfg, LocalTensor<uint32_t> selectedIds, __gm__ uint32_t* map,
    uint32_t b, uint32_t rt_resident, uint32_t strict, uint32_t allowance,
    uint32_t slots, uint32_t& missing, bool frozenMap) {
    Simt::VF_CALL<RankMisses>(Simt::Dim3{256,1,1},ctx,b,rt_resident,
        reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr()),strict,
        reinterpret_cast<__ubuf__ uint32_t*>(scratch[2048].GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(scratch[4096].GetPhyAddr()),map,frozenMap);
    PipeBarrier<PIPE_V>();
    CompactMisses(reinterpret_cast<__ubuf__ uint32_t*>(scratch[2048].GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(scratch[4096].GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),kTop,allowance,
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
__aicore__ inline uint32_t ServiceFetchRank(__gm__ Context* ctx, LocalTensor<uint32_t> scratch,
    LocalTensor<uint32_t> cfg, LocalTensor<uint32_t> selectedIds, GM_ADDR scores, __gm__ uint32_t* map,
    uint32_t b, uint32_t history, uint32_t resident, uint32_t rounds, uint32_t versions,
    uint32_t target, const uint32_t* ends, kvip_cold_scan::Range order, uint32_t allowance,
    uint32_t slots, uint32_t& missing, bool frozenMap) {
    uint32_t count = 0;
    for (uint32_t v = versions; v < target; ++v) {
        for (uint32_t q = 0; q < 4; ++q) {
            uint32_t strict = AdvanceFetchRank(ctx, scratch, scores, b*4+q, history, resident,
                                               rounds, v, ends[q], order);
            if (v+1 == target && count < allowance) {
                count += AdmitLocalRank(ctx, scratch, cfg, selectedIds[count], map, b, resident,
                                        strict, allowance-count, slots+count, missing, frozenMap);
            }
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
