// Adapted from KVIP e2315cf incremental_methods.h; only dynamic TopK.
// Publish cold offers while the recovered TopK IDs are still on chip.
bool mappedExport_=false;uint32_t mappedValid_=0;
bool exportWindowCold_=false;
uint32_t rankedEpoch_=0;
uint32_t rankedCounts_[8]={}; // BS16, two requests per lane, Q4.
__aicore__ inline void PublishWindowRank(uint32_t query, bool remapPacked=false) {
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    // Preserve the threshold in scratch word zero. The rest of TopK scratch is dead.
    auto packed=remapPacked ? incrementalNew_.template ReinterpretCast<uint32_t>() : topkSharedTmpLocal_[64];
    auto count=topkSharedTmpLocal_[remapPacked ? 8 : 32];
    if(!remapPacked) {
        lifetch_ranked::PackOffers(reinterpret_cast<__ubuf__ uint32_t*>(indicesOutLocal_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(packed.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(count.GetPhyAddr()),runtimeShape_.history);
    }
    SetFlag<HardEvent::V_S>(EVENT_ID5);WaitFlag<HardEvent::V_S>(EVENT_ID5);
    uint32_t n=count.GetValue(0);
    const uint32_t offer = ((rankedEpoch_+1)<<16) | n;
    rankedCounts_[incrementalMetadataIndex_]=offer;
    incrementalMetadata_.SetValue(incrementalMetadataIndex_*16+3,offer);
    if(n) {
        GlobalTensor<uint32_t> out;
        uint64_t row=uint64_t(rankedEpoch_)*runtimeShape_.batch*4+query;
        out.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->rankedOffers)+row*2048);
        const uint32_t aligned=(n+7U)&~7U;
        if(incrementalPipeline_ && n<=rotating::kRankedLocalOffers) {
            auto bank=rankedLocalOffers_[incrementalMetadataIndex_*rotating::kRankedLocalOffers];
            DataCopy(bank,packed,aligned);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID5);WaitFlag<HardEvent::V_MTE3>(EVENT_ID5);
            // Immutable backing for a later unchanged window. This window can
            // read bank directly. The metadata fence joins all eight writes.
            DataCopy(out,bank,aligned);
        } else {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID5);WaitFlag<HardEvent::V_MTE3>(EVENT_ID5);
            DataCopy(out,packed,aligned);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID5);WaitFlag<HardEvent::MTE3_S>(EVENT_ID5);
        }
    }
}
__aicore__ inline GM_ADDR WindowTopkPrefix() { return prefixOutput_; }
// Included in QLIVector. Use the existing merge/output/TopK scratch buffers.
// Raw scores and predicates die before TopK reuses its scratch. ID maps
// follow the LI trunk geometry; optional next-Q input uses only spare LI UB.
TBuf<TPosition::VECCALC> incrementalIdsBuf_;
LocalTensor<uint32_t> incrementalOld_;
LocalTensor<uint16_t> incrementalNew_;
LocalTensor<uint64_t> incrementalMetadata_;
bool incrementalMetadataActive_=false;
// Eight Fetch windows use eight AIVs: two BS16 requests per lane.
static constexpr uint32_t kRankedMaxRequests=2;
static constexpr uint32_t kRankedMetadataBytes=kRankedMaxRequests*512;
uint32_t incrementalMetadataIndex_=0;
// Stage only the next raw scores and bitmap. Read old TopK state on demand.
// The unchanged TopK scratch is larger than the donor's; a 40 KiB stage
// silently disables overlap under this source's UB guard. Stage 8K positions
// at a time, retaining the existing loop for the rest of the ready interval.
static constexpr uint32_t kIncrementalInputTokens=8192;
static constexpr uint32_t kRankedRawWords=kIncrementalInputTokens+kIncrementalInputTokens/8;
static constexpr uint32_t kIncrementalInputBytes=kRankedRawWords*2+8*rotating::kRankedLocalOffers*4;
LocalTensor<uint32_t> rankedLocalOffers_;
TBuf<TPosition::VECCALC> incrementalInputBuf_;
LocalTensor<uint16_t> incrementalInput_;
bool incrementalPipeline_=false,incrementalInputPending_=false;
uint32_t incrementalFinalValid_=0;
uint32_t incrementalInputQuery_=0,incrementalInputBase_=0,incrementalInputCount_=0;
uint32_t incrementalNextQuery_=0,incrementalNextIndex_=0,incrementalNextEnd0_=0,incrementalNextEnd1_=0;
bool incrementalHasNext_=false;
__aicore__ inline __gm__ uint64_t* IncrementalMeta(uint32_t query) {
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    return reinterpret_cast<__gm__ uint64_t*>(ctx->rankedState)+query*16;
}
__aicore__ inline void SeedIncremental(uint32_t query,uint32_t row,bool hasOrderedSeed) {
    auto meta=IncrementalMeta(query);
    meta[0]=reinterpret_cast<uint64_t>(scoreGm.GetPhyAddr())+
        uint64_t(row)*runtimeShape_.history*2;
    meta[1]=uint64_t(runtimeShape_.resident)|(uint64_t(runtimeShape_.resident)<<32);
    meta[2]=prefixThreshold_.GetValue((query%2)*128);
    for(uint32_t i=3;i<16;++i)meta[i]=0;
    // Zero is unknown. Encoded zero-strict is one and is a valid state.
    if(hasOrderedSeed)meta[5]=topkOp_.RankedStrictCount()+1;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(meta),128);
}
__aicore__ inline void FlushIncremental(uint32_t count,uint32_t base,bool clearCandidates=true) {
    mappedExport_=false;
    if(!count)return;
    PipeBarrier<PIPE_V>();
    DataCopy(mrgValueLocal_,scoreOutLocal_,2048);
    DataCopy(incrementalOld_,indicesOutLocal_,2048);
    PipeBarrier<PIPE_V>();
    if(count%256) {
        lifetch_ranked::ClearRankPadding(reinterpret_cast<__ubuf__ uint16_t*>(mrgValueLocal_[2048].GetPhyAddr()),count);
        PipeBarrier<PIPE_V>();
    }
    topkOp_(mrgValueLocal_,indicesOutLocal_,scoreOutLocal_,2048+((count+255)/256)*256,0,1,true);
    PipeBarrier<PIPE_V>();
    // Use 32000 bytes of TopK scratch: 2048 old IDs plus 5952 added IDs.
    if(count<=5952) {
        DataCopy(topkSharedTmpLocal_,incrementalOld_,2048);
        lifetch_incremental::BuildAddedMap(reinterpret_cast<__ubuf__ uint16_t*>(incrementalNew_.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(topkSharedTmpLocal_[2048].GetPhyAddr()),count,base);
        PipeBarrier<PIPE_V>();
        lifetch_incremental::MapAndExport(reinterpret_cast<__ubuf__ uint32_t*>(indicesOutLocal_.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(topkSharedTmpLocal_.GetPhyAddr()),reinterpret_cast<__ubuf__ uint16_t*>(scoreOutLocal_.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(incrementalNew_.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(topkSharedTmpLocal_.GetPhyAddr()),0,mappedValid_,exportWindowCold_ && !clearCandidates);
        mappedExport_=true;
    } else {
    Simt::VF_CALL<lifetch_incremental::MapBitmap>(Simt::Dim3{256,1,1},
        reinterpret_cast<__ubuf__ uint32_t*>(indicesOutLocal_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(incrementalOld_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint16_t*>(incrementalNew_.GetPhyAddr()),base);
    }
    PipeBarrier<PIPE_V>();
}
// A committed rank is immutable until its next rank owner. Posting uses a
// separate offer packet, never the prefix being consumed by the final query.
__aicore__ inline void WaitForRank(uint32_t epoch,uint32_t lane) {
    if(!epoch)return;
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    auto event=reinterpret_cast<__gm__ volatile rotating::Event*>(ctx->events)
        +(epoch-1)*runtimeShape_.serviceLanes+lane;
    do {
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ rotating::Event*>(event)),64);
    } while(!event->rankReady);
}
// Snapshot the next Q before submitting any DMA. The owner of this service
// lane is the only writer of these incremental states until the window ends.
__aicore__ inline void PreloadIncremental(uint32_t half) {
    if(!incrementalPipeline_ || !incrementalHasNext_)return;
    uint32_t mi=incrementalNextIndex_*16;
    uint64_t source=incrementalMetadata_.GetValue(mi);
    uint64_t cursor=incrementalMetadata_.GetValue(mi+1);
    uint32_t prior[2]={uint32_t(cursor),uint32_t(cursor>>32)};
    uint32_t ends[2]={incrementalNextEnd0_,incrementalNextEnd1_};
    if(!source || ends[0]<runtimeShape_.resident)return;
    uint32_t base=Min(prior[0],half?prior[1]:prior[0]),limit=ends[0];
    if(half && (prior[0]!=prior[1] || ends[0]!=ends[1])) {
    while(base<Max(ends[0],half?ends[1]:ends[0])) {
        if(!half)break;
        auto slice=rotating::SequenceSliceAt(base,runtimeShape_.resident,half);
        if(base<prior[slice.shard]){base=Min(slice.end,prior[slice.shard]);continue;}
        limit=Min(slice.end,ends[slice.shard]);
        if(base>=limit){base=slice.end;continue;}
        uint32_t next=limit;
        while(next<Max(ends[0],ends[1]) && next-base<kIncrementalInputTokens) {
            auto following=rotating::SequenceSliceAt(next,runtimeShape_.resident,half);
            if(next<prior[following.shard] || next>=ends[following.shard])break;
            uint32_t stop=Min(following.end,ends[following.shard]);
            if(stop<=next)break;
            limit=stop;next=stop;
        }
        break;
    }
    }
    if(base>=limit)return;
    uint32_t n=Min(kIncrementalInputTokens,limit-base);
    uint32_t query=incrementalNextQuery_;
    GlobalTensor<uint16_t> raw,mask;
    raw.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(source));
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    mask.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(ctx->hits)+uint64_t(query)*runtimeShape_.history/8);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID6);WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);
    DataCopy(incrementalInput_,raw[base],n);
    DataCopy(incrementalInput_[kIncrementalInputTokens],mask[base/8],n/8);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID6);
    incrementalInputQuery_=query;incrementalInputBase_=base;incrementalInputCount_=n;
    incrementalInputPending_=true;
}
__aicore__ inline bool AdvanceIncremental(uint32_t query,uint32_t end0,uint32_t end1,uint32_t half) {
    constexpr uint32_t prefixBytes=4*2048*6+1024;
    auto meta=IncrementalMeta(query);
    if(!incrementalMetadataActive_)kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(meta),128);
    uint32_t mi=incrementalMetadataIndex_*16;
    uint64_t source=incrementalMetadataActive_?incrementalMetadata_.GetValue(mi):meta[0];
    uint64_t cursor=incrementalMetadataActive_?incrementalMetadata_.GetValue(mi+1):meta[1];
    uint32_t prior[2]={uint32_t(cursor),uint32_t(cursor>>32)};
    uint32_t ends[2]={end0,end1};
    if (!source) return false;
    const uint32_t savedOffer=exportWindowCold_ ? uint32_t(incrementalMetadata_.GetValue(mi+3)) : 0;
    const uint32_t savedStrict=exportWindowCold_ ? uint32_t(incrementalMetadata_.GetValue(mi+5)) : uint32_t(meta[5]);
    mappedExport_=false;
    if (end0<=prior[0] && (!half || end1<=prior[1])) {
        // The prior TopK is already persisted. Fetch can reuse it unchanged.
        if(savedOffer)rankedCounts_[incrementalMetadataIndex_]=savedOffer;
        else if(exportWindowCold_) {
            GlobalTensor<uint32_t> priorIds;
            uint64_t addr=reinterpret_cast<uint64_t>(WindowTopkPrefix())+uint64_t(query/4)*prefixBytes+4*2048*2;
            priorIds.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(addr)+(query%4)*2048);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID5);WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
            DataCopy(indicesOutLocal_,priorIds,2048);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID5);WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
            PublishWindowRank(query);
        }
        return false;
    }
    const uint32_t batch=query/4,q=query%4;
    mappedValid_=incrementalFinalValid_ ? Min(runtimeShape_.history,incrementalFinalValid_) : runtimeShape_.history;
    auto mappedCtx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    if(mappedCtx->queryEnds)mappedValid_=Min(mappedValid_,reinterpret_cast<__gm__ uint32_t*>(mappedCtx->queryEnds)[query]);
    GlobalTensor<uint16_t> stateScores,raw,bitmap;
    GlobalTensor<uint32_t> stateIds;
    stateScores.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(WindowTopkPrefix()+batch*prefixBytes)+q*2048);
    stateIds.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(WindowTopkPrefix()+batch*prefixBytes+4*2048*2)+q*2048);
    raw.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(source));
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    bitmap.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(ctx->hits)+uint64_t(query)*runtimeShape_.history/8);
    if(!incrementalInputPending_ && incrementalPipeline_ && incrementalMetadataActive_) {
        uint32_t nq=incrementalNextQuery_,ni=incrementalNextIndex_,e0=incrementalNextEnd0_,e1=incrementalNextEnd1_;bool hn=incrementalHasNext_;
        incrementalNextQuery_=query;incrementalNextIndex_=incrementalMetadataIndex_;incrementalNextEnd0_=end0;incrementalNextEnd1_=end1;incrementalHasNext_=true;
        PreloadIncremental(half);
        incrementalNextQuery_=nq;incrementalNextIndex_=ni;incrementalNextEnd0_=e0;incrementalNextEnd1_=e1;incrementalHasNext_=hn;
    }
    bool staged=incrementalInputPending_ && incrementalInputQuery_==query;
    if(staged) {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID6);
        incrementalInputPending_=false;
    }
    bool prefixLoaded=false;
    // A cold score has to beat the cutoff before the old rank is needed.
    auto rawLocal=topkSharedTmpLocal_.template ReinterpretCast<uint16_t>();
    const uint32_t rawCapacity=((topkOp_.GetSharedTmpBufferSize()-128)*4/9/128)*128;
    auto scratchRaw=rawLocal;
    auto maskLocal=rawLocal[rawCapacity];
    auto scratchMask=maskLocal;
    auto countLocal=maskLocal[rawCapacity/8].template ReinterpretCast<uint32_t>();
    uint32_t packed=0,packBase=0;
    const uint16_t cutoff=uint16_t(incrementalMetadataActive_?incrementalMetadata_.GetValue(mi+2):meta[2]);
    for (uint32_t base=Min(prior[0],half?prior[1]:prior[0]);base<Max(end0,half?end1:end0);) {
        uint32_t shard=0,limit=end0;
        bool contiguous=!half || (prior[0]==prior[1] && end0==end1);
        if (half && !contiguous) {
            auto slice=rotating::SequenceSliceAt(base,runtimeShape_.resident,half);
            shard=slice.shard;limit=Min(slice.end,ends[shard]);
            if (base<prior[shard]) {base=Min(slice.end,prior[shard]);continue;}
            if (base>=limit) {base=slice.end;continue;}
        }
        // Adjacent ready shards are contiguous score input. Load them together
        // instead of paying a separate MTE2 round trip for every half-window.
        if(half && !contiguous) {
            uint32_t next=limit;
            while(next<Max(end0,end1) && next-base<rawCapacity) {
                auto slice=rotating::SequenceSliceAt(next,runtimeShape_.resident,half);
                if(next<prior[slice.shard] || next>=ends[slice.shard])break;
                uint32_t stop=Min(slice.end,ends[slice.shard]);
                if(stop<=next)break;
                limit=stop;next=stop;
            }
        }
        uint32_t n=Min(Min(rawCapacity,kIncrementalInputTokens),limit-base);
        if (!n) break;
        // Published boundaries are whole 128-score tiles. Final causal bounds
        // use the ordinary path unless aligned, avoiding padded score reads.
        if(staged && base==incrementalInputBase_) {
            n=incrementalInputCount_;
            rawLocal=incrementalInput_;maskLocal=incrementalInput_[kIncrementalInputTokens];
            staged=false;
        } else {
            rawLocal=scratchRaw;maskLocal=scratchMask;
        SetFlag<HardEvent::V_MTE2>(EVENT_ID5);WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
        DataCopy(rawLocal,raw[base],n);DataCopy(maskLocal,bitmap[base/8],n/8);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID5);WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
        }
        auto seeded=kvip_recent_index::RecentRange(reinterpret_cast<__gm__ rotating::Context*>(rotationContext_),query/4,mappedValid_);
        if(base<seeded.end && base+n>seeded.begin) {
            kvip_recent_index::ExcludeSeeded(reinterpret_cast<__ubuf__ uint16_t*>(maskLocal.GetPhyAddr()),base,n,seeded.begin,seeded.end);
            PipeBarrier<PIPE_V>();
        }
        // Final causal Q lengths can end inside a 128-token bitmap tile.
        // The last aligned read includes later Q tokens; remove those bits
        // before counting or packing candidates, including on scratch reloads.
        if (base+n>mappedValid_) {
            kvip_recent_index::ExcludeSeeded(reinterpret_cast<__ubuf__ uint16_t*>(maskLocal.GetPhyAddr()),base,n,mappedValid_,base+n);
            PipeBarrier<PIPE_V>();
        }
        auto tileList=indicesOutLocal_[2048].template ReinterpretCast<uint16_t>();
        lifetch_incremental::MarkNonemptyTiles(
            reinterpret_cast<__ubuf__ uint16_t*>(maskLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint16_t*>(tileList.GetPhyAddr()),n);
        PipeBarrier<PIPE_V>();
        lifetch_incremental::PackTileList(reinterpret_cast<__ubuf__ uint16_t*>(tileList.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(countLocal.GetPhyAddr()));
        SetFlag<HardEvent::V_S>(EVENT_ID5);WaitFlag<HardEvent::V_S>(EVENT_ID5);
        uint32_t bound=countLocal.GetValue(0),blocks=countLocal.GetValue(8);
        if(bound<=trunkLen_-packed && (!packed || base+n-packBase<=65536)) {
        if(!packed)packBase=base;
        lifetch_incremental::CompactSparseTiles(
            reinterpret_cast<__ubuf__ uint16_t*>(rawLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint16_t*>(maskLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint16_t*>(tileList.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint16_t*>(mrgValueLocal_[2048+packed].GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint16_t*>(incrementalNew_[packed].GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(countLocal.GetPhyAddr()),blocks,uint16_t(base-packBase),cutoff);
        SetFlag<HardEvent::V_S>(EVENT_ID5);WaitFlag<HardEvent::V_S>(EVENT_ID5);
        uint32_t count=countLocal.GetValue(0);packed+=count;
        } else {
        lifetch_incremental::Count(reinterpret_cast<__ubuf__ uint16_t*>(rawLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint16_t*>(maskLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(countLocal.GetPhyAddr()),n,cutoff);
        SetFlag<HardEvent::V_S>(EVENT_ID5);WaitFlag<HardEvent::V_S>(EVENT_ID5);
        uint32_t count=countLocal.GetValue(0);
        // Dense input must still fit the unchanged merge capacity.
        if(count>trunkLen_) {
            n=trunkLen_;
            SetFlag<HardEvent::V_MTE2>(EVENT_ID5);WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
            DataCopy(rawLocal,raw[base],n);DataCopy(maskLocal,bitmap[base/8],n/8);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID5);WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
            if(base<seeded.end && base+n>seeded.begin) {
                kvip_recent_index::ExcludeSeeded(reinterpret_cast<__ubuf__ uint16_t*>(maskLocal.GetPhyAddr()),base,n,seeded.begin,seeded.end);
                PipeBarrier<PIPE_V>();
            }
            if (base+n>mappedValid_) {
                kvip_recent_index::ExcludeSeeded(reinterpret_cast<__ubuf__ uint16_t*>(maskLocal.GetPhyAddr()),base,n,mappedValid_,base+n);
                PipeBarrier<PIPE_V>();
            }
            lifetch_incremental::Count(reinterpret_cast<__ubuf__ uint16_t*>(rawLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint16_t*>(maskLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(countLocal.GetPhyAddr()),n,cutoff);
            SetFlag<HardEvent::V_S>(EVENT_ID5);WaitFlag<HardEvent::V_S>(EVENT_ID5);
            count=countLocal.GetValue(0);
        }
        if (count) {
                // Capacity is measured in compact candidates, not raw score tiles.
            // Reuse the demo's uint16 relative IDs across both sequence halves.
            if(packed && (packed+count>trunkLen_ || base+n-packBase>65536)) {
                if(!prefixLoaded) {
                    DataCopy(scoreOutLocal_,stateScores,2048);DataCopy(indicesOutLocal_,stateIds,2048);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID7);WaitFlag<HardEvent::MTE2_V>(EVENT_ID7);
                    prefixLoaded=true;
                }
                FlushIncremental(packed,packBase);packed=0;
                // TopK reuses raw/mask scratch; reload only on a capacity flush.
                SetFlag<HardEvent::V_MTE2>(EVENT_ID5);WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
                DataCopy(rawLocal,raw[base],n);DataCopy(maskLocal,bitmap[base/8],n/8);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID5);WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
                if(base<seeded.end && base+n>seeded.begin) {
                kvip_recent_index::ExcludeSeeded(reinterpret_cast<__ubuf__ uint16_t*>(maskLocal.GetPhyAddr()),base,n,seeded.begin,seeded.end);
                PipeBarrier<PIPE_V>();
            }
            if (base+n>mappedValid_) {
                kvip_recent_index::ExcludeSeeded(reinterpret_cast<__ubuf__ uint16_t*>(maskLocal.GetPhyAddr()),base,n,mappedValid_,base+n);
                PipeBarrier<PIPE_V>();
            }
            lifetch_incremental::Count(reinterpret_cast<__ubuf__ uint16_t*>(rawLocal.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint16_t*>(maskLocal.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(countLocal.GetPhyAddr()),n,cutoff);
                PipeBarrier<PIPE_V>();
            }
            if(!packed)packBase=base;
            lifetch_incremental::Compact(reinterpret_cast<__ubuf__ uint16_t*>(rawLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint16_t*>(maskLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint16_t*>(mrgValueLocal_[2048+packed].GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint16_t*>(incrementalNew_[packed].GetPhyAddr()),n,uint16_t(base-packBase));
            packed+=count;
        }
        }
        uint32_t stop=base+n;
        if(contiguous && (!half || (stop-runtimeShape_.resident)%(2*half)==0)) {prior[0]=stop;prior[1]=stop;base=stop;}
        while(base<stop) {
            auto slice=half?rotating::SequenceSliceAt(base,runtimeShape_.resident,half):rotating::SequenceSlice{0,stop};
            base=Min(slice.end,stop);prior[slice.shard]=base;
        }
    }
    // Current input is dead after compaction. Start the next Q's MTE2 while
    // this Q performs TopK, ID mapping, threshold reduction and writeback.

    PreloadIncremental(half);
    const uint64_t nextCursor=uint64_t(prior[0])|(uint64_t(prior[1])<<32);
    if(!packed && !prefixLoaded) {
        // No winner changed. Reoffer the immutable old list: unposted IDs
        // remain eligible. The final pass already has its IDs in stateIds.
        if(exportWindowCold_) {
            if(savedOffer)rankedCounts_[incrementalMetadataIndex_]=savedOffer;
            else {
                DataCopy(indicesOutLocal_,stateIds,2048);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID7);WaitFlag<HardEvent::MTE2_V>(EVENT_ID7);
                PublishWindowRank(query);
            }
            incrementalMetadata_.SetValue(mi+1,nextCursor);
        }
        return false;
    }

    const uint32_t strict=savedStrict ? savedStrict-1 : 2048;
    const bool tieOnly=!prefixLoaded && packed && packed<2048-strict;
    // Aligned down for the prefix DMA; every element preceding strict is preserved.
    const uint32_t first=tieOnly ? (strict & ~15U) : 0;
    if(!prefixLoaded) {
        DataCopy(scoreOutLocal_[first],stateScores[first],2048-first);
        DataCopy(indicesOutLocal_,stateIds,2048);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID7);WaitFlag<HardEvent::MTE2_V>(EVENT_ID7);
    }
    uint32_t threshold=cutoff;
    if(tieOnly) {
        auto freshIds=topkSharedTmpLocal_;
        lifetch_incremental::BuildAddedMap(reinterpret_cast<__ubuf__ uint16_t*>(incrementalNew_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(freshIds.GetPhyAddr()),packed,packBase);
        PipeBarrier<PIPE_V>();
        lifetch_ranked::InsertAboveBoundary(reinterpret_cast<__ubuf__ uint32_t*>(indicesOutLocal_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint16_t*>(scoreOutLocal_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(freshIds.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint16_t*>(mrgValueLocal_[2048].GetPhyAddr()),strict,packed);
        PipeBarrier<PIPE_V>();
        if(!exportWindowCold_)return true;
        PublishWindowRank(query);
        incrementalMetadata_.SetValue(mi+5,strict+packed+1);
    } else {
        FlushIncremental(packed,packBase,false);
        PipeBarrier<PIPE_V>();
        // This query's public output is the sole consumer of the final IDs.
        // Leave the last committed window prefix and its metadata together.
        if(!exportWindowCold_)return true;
        if(!mappedExport_)lifetch_incremental::Minimum(reinterpret_cast<__ubuf__ uint16_t*>(scoreOutLocal_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(topkSharedTmpLocal_.GetPhyAddr()));
        SetFlag<HardEvent::V_S>(EVENT_ID5);WaitFlag<HardEvent::V_S>(EVENT_ID5);
        threshold=topkSharedTmpLocal_.template ReinterpretCast<uint16_t>().GetValue(0);
        PublishWindowRank(query,mappedExport_);
        incrementalMetadata_.SetValue(mi+5,topkOp_.RankedStrictCount()+1);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID5);WaitFlag<HardEvent::V_MTE3>(EVENT_ID5);
    DataCopy(stateScores[first],scoreOutLocal_[first],2048-first);
    DataCopy(stateIds[first],indicesOutLocal_[first],2048-first);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID5);WaitFlag<HardEvent::MTE3_S>(EVENT_ID5);
    if(incrementalMetadataActive_) {
        incrementalMetadata_.SetValue(mi+1,uint64_t(prior[0])|(uint64_t(prior[1])<<32));
        incrementalMetadata_.SetValue(mi+2,threshold);
    } else {
        meta[1]=uint64_t(prior[0])|(uint64_t(prior[1])<<32);meta[2]=threshold;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(meta),128);
    }
    return true;
}
__aicore__ inline bool AdvanceWindow(uint32_t epoch,uint32_t half) {
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    uint32_t lane=rotating::ServiceLane(runtimeShape_.batch,runtimeShape_.splitQ,
        runtimeShape_.serviceAics,blockId_/2,blockId_%2);
    // The predecessor wait is part of the optional window service and can
    // delay the producer. Capture it separately from incremental computation.
    bool trace=streamingWorkspace_ && (reinterpret_cast<__gm__ pipeline_ab::Args*>(streamingWorkspace_)->prefetchCount & 0x80000000U);
    if(trace) {
        auto event=reinterpret_cast<__gm__ rotating::Event*>(ctx->events)+epoch*runtimeShape_.serviceLanes+lane;
        event->waitBegin=GetSystemCycle();
    }
    // Only the prefix writer must finish here. Ordered row/SQ ownership
    // is acquired by FastService after this window has frozen its offers.
    WaitForRank(epoch,lane);
    uint64_t begin=trace?GetSystemCycle():0;
    rankedEpoch_=epoch;
    for(auto& count:rankedCounts_)count=0;
    exportWindowCold_=true;
    incrementalMetadataActive_=(runtimeShape_.batch+runtimeShape_.serviceLanes-1)/runtimeShape_.serviceLanes<=kRankedMaxRequests;
    if(incrementalMetadataActive_) {
    GlobalTensor<uint32_t> progressSource;progressSource.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->progress));
    for(uint32_t b=lane,i=0;b<runtimeShape_.batch;b+=runtimeShape_.serviceLanes,i+=128)
        DataCopy(incrementalOld_[i],progressSource[b*128],128);
    }
    GlobalTensor<uint64_t> metadata;
    metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(ctx->rankedState));
    if(incrementalMetadataActive_) {
        for(uint32_t b=lane,i=0;b<runtimeShape_.batch;b+=runtimeShape_.serviceLanes,i+=64)
            DataCopy(incrementalMetadata_[i],metadata[b*64],64);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID5);WaitFlag<HardEvent::MTE2_S>(EVENT_ID5);
    }
    uint32_t ownedQueries=lane<runtimeShape_.batch?((runtimeShape_.batch-1-lane)/runtimeShape_.serviceLanes+1)*4:0;
    uint32_t readyMasks[kRankedMaxRequests]={};
    if(incrementalMetadataActive_) {
        for(uint32_t i=0;i<ownedQueries;++i)
            incrementalMetadata_.SetValue(i*16+4,uint64_t(incrementalOld_.GetValue(i*32))|(uint64_t(incrementalOld_.GetValue(i*32+16))<<32));
        if(trace)for(uint32_t i=0;i<ownedQueries;++i)
            incrementalMetadata_.SetValue(i*16+8,incrementalPipeline_?3:2);
    }
    uint32_t cap=half?rotating::SequenceWindowEnd(runtimeShape_.history,runtimeShape_.resident,half,epoch):
        rotating::Trigger(runtimeShape_.history,runtimeShape_.resident,runtimeShape_.rounds,epoch)*128;
    for(uint32_t b=lane;b<runtimeShape_.batch;b+=runtimeShape_.serviceLanes) for(uint32_t q=0;q<4;++q) {
        uint32_t progressIndex=(((b-lane)/runtimeShape_.serviceLanes)*4+q)*16+4;
        uint64_t ready;
        if(incrementalMetadataActive_)ready=incrementalMetadata_.GetValue(progressIndex);
        else {
            auto progress=reinterpret_cast<__gm__ volatile uint32_t*>(ctx->progress)+(b*4+q)*32;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ uint32_t*>(progress)),128);
            ready=uint64_t(progress[0])|(uint64_t(progress[16])<<32);
        }
        uint32_t end0=Min(uint32_t(ready),cap),end1=half?Min(uint32_t(ready>>32),cap):0;
        if(end0<runtimeShape_.resident)continue;
        readyMasks[(b-lane)/runtimeShape_.serviceLanes]|=1U<<q;
        incrementalMetadataIndex_=((b-lane)/runtimeShape_.serviceLanes)*4+q;
        uint32_t nextB=q==3?b+runtimeShape_.serviceLanes:b,nextQ=(q+1)%4;
        incrementalHasNext_=incrementalPipeline_ && incrementalMetadataActive_ && nextB<runtimeShape_.batch;
        if(incrementalHasNext_) {
            incrementalNextQuery_=nextB*4+nextQ;incrementalNextIndex_=incrementalMetadataIndex_+1;
            uint64_t nextReady=incrementalMetadata_.GetValue(progressIndex+16);
            incrementalNextEnd0_=Min(uint32_t(nextReady),cap);
            incrementalNextEnd1_=half?Min(uint32_t(nextReady>>32),cap):0;
        }
        AdvanceIncremental(b*4+q,end0,end1,half);
    }
    if(incrementalInputPending_) {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID6);incrementalInputPending_=false;
    }
    incrementalHasNext_=false;
    if(incrementalMetadataActive_) {
        SetFlag<HardEvent::S_MTE3>(EVENT_ID5);WaitFlag<HardEvent::S_MTE3>(EVENT_ID5);
        // Use the existing progress snapshot: the common all-ready row is
        // one DMA. Only a partial seed needs query-granular writeback, so a
        // producer that is still initializing its header cannot be clobbered.
        for(uint32_t b=lane,i=0;b<runtimeShape_.batch;b+=runtimeShape_.serviceLanes,++i) {
            if(readyMasks[i]==15)DataCopy(metadata[b*64],incrementalMetadata_[i*64],64);
            else for(uint32_t q=0;q<4;++q)
                if(readyMasks[i]&(1U<<q))
                    DataCopy(metadata[b*64+q*16],incrementalMetadata_[i*64+q*16],16);
        }
        SetFlag<HardEvent::MTE3_S>(EVENT_ID5);WaitFlag<HardEvent::MTE3_S>(EVENT_ID5);
        incrementalMetadataActive_=false;
    }
    // The next rank needs only this committed prefix. It writes a different
    // epoch's offer buffer. Row, mapping and SQ ownership still pass on complete.
    auto published=reinterpret_cast<__gm__ rotating::Event*>(ctx->events)+epoch*runtimeShape_.serviceLanes+lane;
    published->rankReady=1;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(published),64);
    exportWindowCold_=false;
    // The producer's progress lookup table shares the TopK score buffer.
    // Rebuild it before LI resumes; otherwise score bytes become fake progress.
    PrepareRotationProgress();
    if(trace) {
        auto event=reinterpret_cast<__gm__ rotating::Event*>(ctx->events)+epoch*runtimeShape_.serviceLanes+lane;
        auto stamp=reinterpret_cast<__gm__ uint64_t*>(event->pad);
        stamp[2]=begin;stamp[3]=GetSystemCycle();
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(event),128);
    }
    return true;
}
__aicore__ inline void FinalIncremental(uint32_t query,uint32_t valid) {
    const uint32_t lane=(query/4)%runtimeShape_.serviceLanes;
    WaitForRank(runtimeShape_.rounds,lane);
    uint32_t half=rotating::UseSequenceSplit(runtimeShape_)?rotating::SequenceHalfWindow(runtimeShape_):0;
    uint32_t alignedValid=(valid+127U)&~127U;
    incrementalFinalValid_=valid;
    const bool localIds=AdvanceIncremental(query,alignedValid,half?alignedValid:0,half);
    incrementalFinalValid_=0;
    if(!localIds) {
        constexpr uint32_t prefixBytes=4*2048*6+1024;
        GlobalTensor<uint32_t> ids;
        ids.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(WindowTopkPrefix()+(query/4)*prefixBytes+4*2048*2)+(query%4)*2048);
        PipeBarrier<PIPE_ALL>();DataCopy(indicesOutLocal_,ids,2048);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID5);WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
    }
    lifetch_ranked::DecodePublicIds(reinterpret_cast<__ubuf__ uint32_t*>(indicesOutLocal_.GetPhyAddr()));
    PipeBarrier<PIPE_V>();
}
