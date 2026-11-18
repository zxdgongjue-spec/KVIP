// SPDX-License-Identifier: Apache-2.0
// Included in rotating, after CompactMisses and ReserveIncremental.
#if KVIP_FETCH_BITMAP || KVIP_FETCH_DENSE
constexpr uint32_t kRankWords = 3072; // 2048 u32 IDs, 2048 u16 keys.
constexpr uint32_t kRankHeader = 32;
constexpr uint32_t kFetchEpochs = 9;

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

template<bool offsetsOnly>
__simd_vf__ inline void PackRankKeys(__ubuf__ uint16_t* raw, __ubuf__ uint16_t* values,
    __ubuf__ uint16_t* offsets, uint16_t cutoff, uint16_t valid, uint16_t skipLo,
    uint16_t skipHi, uint16_t tiles) {
    using namespace MicroAPI;
    auto all = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<uint16_t> value, packed;
    RegTensor<int16_t> lanes;
    MaskReg above, causal, left, right, keep;
    UnalignRegForStore writeValues, writeOffsets;
    ClearSpr<SpecialPurposeReg::AR>();
    for (uint16_t tile = 0; tile < tiles; ++tile) {
        uint16_t i = tile * 128;
        LoadAlign(value, raw + i);
        Arange(lanes, int16_t(i));
        auto& index = reinterpret_cast<RegTensor<uint16_t>&>(lanes);
        CompareScalar<uint16_t, CMPMODE::GT>(above, value, cutoff, all);
        CompareScalar<uint16_t, CMPMODE::LT>(causal, index, valid, all);
        CompareScalar<uint16_t, CMPMODE::LT>(left, index, skipLo, all);
        CompareScalar<uint16_t, CMPMODE::GE>(right, index, skipHi, all);
        MaskOr(keep, left, right, all);
        MaskAnd(keep, keep, causal, all);
        MaskAnd(keep, keep, above, all);
        if constexpr(offsetsOnly) {
            Squeeze<uint16_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(packed,index,keep);
            StoreUnAlign<uint16_t,PostLiteral::POST_MODE_UPDATE>(offsets,packed,writeOffsets);
        } else {
            Squeeze<uint16_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(packed,value,keep);
            StoreUnAlign<uint16_t,PostLiteral::POST_MODE_UPDATE>(values,packed,writeValues);
        }
    }
    if constexpr(offsetsOnly) StoreUnAlignPost(offsets,writeOffsets);
    else StoreUnAlignPost(values,writeValues);
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
            outKeys[strict+i] = values[i];
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
    // Store cutoff ties backwards, so appending strict winners touches only the
    // discarded end of the tie cohort. Ties are converted back at publication.
    if (strict + added < kTop) {
        Simt::VF_CALL<WriteRank<false>>(Simt::Dim3{256,1,1}, RankPtr(ids), RankPtr(keys),
            RankPtr(ids), RankPtr(values), RankPtr(offsets), RankPtr(offsets), physical, added, strict);
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
    DataCopy(input,keys,kTop);
    // The histogram includes padding in the ascending rank. A strictly higher
    // cutoff is positive, so padded zeroes cannot enter the selected set.
    for(uint32_t i=kTop+added;i<size;++i) input.SetValue(i,0);
    SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
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

__aicore__ inline void AdvanceFetchRank(__gm__ Context* ctx, LocalTensor<uint32_t> a,
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
        auto vals=a[7168].ReinterpretCast<uint16_t>(), off=a[8224].ReinterpretCast<uint16_t>();
        auto mask=a[9280];
        for(uint32_t scan=Trigger(history,resident,rounds,epoch-1)*128;scan<stop;) {
            uint32_t n=2048-(scan%2048);if(n>stop-scan)n=stop-scan;
            uint32_t p=kvip_cold_scan::Remap(scan,order);
            uint32_t valid=ClipRankOffset(queryEnd,p,n);
            if(valid && !(p>=lo && p+valid<=hi)) {
#if KVIP_FETCH_BITMAP
                DataCopy(mask,bits[scan/16],n/16);
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                bool any=false;
                for(uint32_t i=0;i<n;i+=128) {
                    bool hit=false;
                    for(uint32_t j=0;j<8;++j)hit|=mask.GetValue(i/16+j)!=0;
                    if(hit) {DataCopy(raw[i],scores[p+i],128);any=true;}
                    else Duplicate(raw[i],uint16_t(0),128);
                }
                if(!any) {PipeBarrier<PIPE_V>();scan+=n;continue;}
                PipeBarrier<PIPE_V>();
#else
                DataCopy(raw,scores[p],n);
#endif
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                PackRankKeys<false>(reinterpret_cast<__ubuf__ uint16_t*>(raw.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint16_t*>(vals.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint16_t*>(off.GetPhyAddr()),
                    kth,valid,ClipRankOffset(lo,p,n),ClipRankOffset(hi,p,n),n/128);
                // AR is the byte count of the final StoreUnAlign stream.
                uint32_t added=GetSpr<SpecialPurposeReg::AR>()/sizeof(uint16_t);
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                if(added) {
                PackRankKeys<true>(reinterpret_cast<__ubuf__ uint16_t*>(raw.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint16_t*>(vals.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint16_t*>(off.GetPhyAddr()),
                    kth,valid,ClipRankOffset(lo,p,n),ClipRankOffset(hi,p,n),n/128);
                    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                }
                MergeRankBlock(a,p,added,kth,strict);
                PipeBarrier<PIPE_ALL>();
            }
            scan+=n;
        }
    }
    auto hdr=a[9792]; hdr.SetValue(0,kth);hdr.SetValue(1,strict);
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(store,hdr,kRankHeader);DataCopy(store[kRankHeader],a,kTop);
    DataCopy(store[kRankHeader+kTop].ReinterpretCast<uint16_t>(),keys,kTop);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
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
__aicore__ inline uint32_t AdmitFetchRank(__gm__ Context* ctx,LocalTensor<uint32_t> scratch,
    LocalTensor<uint32_t> cfg, LocalTensor<uint32_t> selectedIds,__gm__ uint32_t* map,
    uint32_t b,uint32_t rt_resident,uint32_t rt_rounds,uint32_t target,
    uint32_t allowance,uint32_t slots,uint32_t& missing,bool frozenMap) {
    uint32_t count=0;
        for(uint32_t q=0;target && q<4 && count<allowance;++q) {
            GlobalTensor<uint32_t> bank;
            bank.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->fetchRanks)
                +uint64_t((b*4+q)*rt_rounds+target-1)*(kRankWords+kRankHeader));
            DataCopy(scratch,bank[kRankHeader],kTop);
            DataCopy(scratch[4096],bank,kRankHeader);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            uint32_t strict=scratch.GetValue(4097);
            Simt::VF_CALL<RankMisses>(Simt::Dim3{256,1,1},ctx,b,rt_resident,
                reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr()),strict,
                reinterpret_cast<__ubuf__ uint32_t*>(scratch[2048].GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(scratch[4096].GetPhyAddr()),map,frozenMap);
            PipeBarrier<PIPE_V>();
            CompactMisses(reinterpret_cast<__ubuf__ uint32_t*>(scratch[2048].GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(scratch[4096].GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),kTop,allowance-count,
                reinterpret_cast<__ubuf__ uint32_t*>(selectedIds[count].GetPhyAddr()));
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            uint32_t found=cfg.GetValue(56),n=found<allowance-count?found:allowance-count;
            if(n) {
                Simt::VF_CALL<ReserveIncremental>(Simt::Dim3{256,1,1},
                    reinterpret_cast<__ubuf__ uint32_t*>(selectedIds[count].GetPhyAddr()),
                    map,n,rt_resident+slots+count);
                PipeBarrier<PIPE_V>();
            }
            count+=n;missing+=found;
        }
    return count;
}
#endif
