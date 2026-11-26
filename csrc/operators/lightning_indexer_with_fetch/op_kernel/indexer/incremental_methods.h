bool taggedLocations_=false;
__aicore__ inline void TagResident(uint32_t query) {
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    auto work=reinterpret_cast<__gm__ pipeline_ab::Args*>(streamingWorkspace_);
    bool profile=streamingWorkspace_ && (work->prefetchCount&0x80000000U);
    auto ticks=streamingWorkspace_?reinterpret_cast<__gm__ uint64_t*>(work->times)+uint64_t(runtimeShape_.batch)*64+uint64_t(query)*16:nullptr;
    if(profile)ticks[12]=GetSystemCycle();
    uint32_t b=query/4,owner=ctx->rowOwners?reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b]:b;
    uint32_t tailBegin=0,tailEnd=0;
    if(ctx->tailBounds){auto bounds=reinterpret_cast<__gm__ uint32_t*>(ctx->tailBounds)+b*2;tailBegin=bounds[0];tailEnd=bounds[1];
        uint32_t capacity=(ctx->outputStride?ctx->outputStride:runtimeShape_.resident)-runtimeShape_.resident;
        tailEnd=Min(tailEnd,Min(runtimeShape_.history,tailBegin+capacity));}
    LocalTensor<uint32_t> hot,reserved;uint32_t chunk;
    if(streamingTopkOnly_){hot=streamingValid_;reserved=streamingReserved_;chunk=8192;}
    else {
        PipeBarrier<PIPE_ALL>();
        // ProcessVec1 has returned the MM result banks to the AIC. Fixpipe
        // may already write the next tile there while resident TopK runs;
        // an AIV pipe barrier cannot reclaim those cross-core buffers.
        // TopK is finished here, so its scratch is private until tagging ends.
        hot=topkSharedTmpLocal_;
        reserved=mrgValueLocal_.template ReinterpretCast<uint32_t>();
        chunk=4096;
    }
    GlobalTensor<uint32_t> h,r;
    if(ctx->hotValid)h.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+uint64_t(owner)*runtimeShape_.resident);
    if(ctx->reservedHot)r.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->reservedHot)+uint64_t(b)*runtimeShape_.resident);
    for(uint32_t base=0;base<runtimeShape_.resident;base+=chunk){
        // EVENT_ID4 already carries ProcessTopK's next-query signal here.
        // Use the balanced scratch-copy event so this wait cannot consume
        // that older signal and overwrite metadata still read by tagging.
        if(base){SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);}
        uint32_t n=Min(chunk,runtimeShape_.resident-base);
        if(!base && streamingTopkOnly_ && streamingPrefetched_) {}
        else {if(ctx->hotValid)DataCopy(hot,h[base],n);if(ctx->reservedHot)DataCopy(reserved,r[base],n);}
        if(!ctx->hotValid)Duplicate(hot,uint32_t(1),n);if(!ctx->reservedHot)Duplicate(reserved,UINT32_MAX,n);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID4);WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
        kvip_prefetch_rank::TagResidentHalf(reinterpret_cast<__ubuf__ uint32_t*>(indicesOutLocal_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(hot.GetPhyAddr()),reinterpret_cast<__ubuf__ int32_t*>(reserved.GetPhyAddr()),base,n,tailBegin,tailEnd);
    }
    PipeBarrier<PIPE_V>();streamingPrefetched_=false;
    if(profile){SetFlag<HardEvent::V_S>(EVENT_ID4);WaitFlag<HardEvent::V_S>(EVENT_ID4);ticks[13]=GetSystemCycle();kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ticks),128);}
}
// Optimized B window path; A retains methods.h for the fixed comparison.
bool incrementalIdsPending_=false;
// Do not consume the primed LI/TopK events (MTE3_V 3, 4, 7).
static constexpr uint32_t WINDOW_COLD_OUTPUT_EVENT = EVENT_ID1;
bool coldOutputPending_=false;
__aicore__ inline void JoinColdOutput() {
    static_assert(WINDOW_COLD_OUTPUT_EVENT != TOPK_MTE3_V_EVENT &&
                  WINDOW_COLD_OUTPUT_EVENT != VEC1_MTE3_V_EVENT &&
                  WINDOW_COLD_OUTPUT_EVENT != VEC1_MTE3_V_EVENT + 1);
    if (coldOutputPending_) {
        WaitFlag<HardEvent::MTE3_V>(WINDOW_COLD_OUTPUT_EVENT);
        coldOutputPending_ = false;
    }
}
bool mappedExport_=false;uint32_t mappedValid_=0;
// QLIVector member: called after ID recovery while the complete Q is in UB.
bool exportWindowCold_=false;bool rankTrace_=false;
uint32_t rankedEpoch_=0;bool windowCoveredEmpty_=false;
__aicore__ inline uint32_t WindowRank(__gm__ rotating::Context* ctx, uint32_t valid,
    uint32_t end0, uint32_t end1, uint32_t half, kvip_recent_index::Range recent, uint32_t& covered) const {
    const auto resident = runtimeShape_.resident;
    const auto initial = kvip_coverage::Covered(0, 0, resident, valid, 0, recent.begin, recent.end);
    covered = kvip_coverage::Covered(end0, end1, resident, valid, half, recent.begin, recent.end);
    return kvip_coverage::Rank(covered, initial, valid, ctx->prefetchRankStart ? uint32_t(ctx->prefetchRankStart) : 1024);
}
#if KVIP_STAGE_TRACE
__aicore__ inline kvip_trace::ScanSample RankSample(uint32_t query) const {
#if KVIP_PARALLEL_TERMINAL
    const bool terminal = rankedEpoch_ == runtimeShape_.rounds;
#else
    constexpr bool terminal = false;
#endif
    return {exportWindowCold_ ? KVIP_TRACE_WORKSPACE(streamingWorkspace_) : 0,
        terminal ? kvip_trace::Terminal : kvip_trace::Cold, terminal ? 0U : rankedEpoch_,
        kvip_trace::PipeDetail(query / 4, query % 4)};
}
#endif
#if KVIP_SERVICE_UB_CANDIDATES
TBuf<TPosition::VECCALC> windowCandidatesBuf_;
LocalTensor<uint32_t> windowCandidates_;
bool windowCandidatesAllocated_ = false;
bool windowCandidatesActive_ = false;
#endif
__aicore__ inline void ExportWindowCold(uint32_t query) {
    // Even a window with zero newly compacted scores exports the saved TopK.
    // Its output bank may still feed the preceding Q's asynchronous DMA.
    JoinColdOutput();
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    uint32_t valid=ctx->queryEnds?reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds)[query]:runtimeShape_.history;
    if(valid>runtimeShape_.history)valid=runtimeShape_.history;
    // The complete Top2048 remains untouched for the next incremental window.
    auto compact=incrementalNew_.template ReinterpretCast<uint32_t>();
    auto recent=kvip_recent_index::RecentRange(ctx,query/4,valid);
    uint32_t coveredCount;
    uint32_t rank=WindowRank(ctx, valid, rankEnd0_, rankEnd1_, rankHalf_, recent, coveredCount);
    if(rankTrace_ && rankedEpoch_<32) {
        uint32_t ready=kvip_coverage::Covered(uint32_t(rankReady_),uint32_t(rankReady_>>32),runtimeShape_.resident,valid,rankHalf_,recent.begin,recent.end);
        uint64_t record=uint64_t(ready)|(uint64_t(coveredCount)<<18)|(uint64_t(rank)<<36);
        if(runtimeShape_.history>=(1U<<18))
            record=(1ULL<<63)|uint64_t(ready)|(uint64_t(coveredCount)<<26)|(uint64_t(rank-1)<<52);
        auto trace=reinterpret_cast<__gm__ uint64_t*>(ctx->incremental+
            runtimeShape_.batch*(512+4*2048*6+1024+(runtimeShape_.history+255)/256*32));
        trace[query*32+rankedEpoch_]=record;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(trace+query*32+rankedEpoch_),8);
    }
    uint32_t b=query/4;
    auto covered=reinterpret_cast<__gm__ uint32_t*>(kvip_prefetch_rank::CoveredAddress(
        ctx->incremental,runtimeShape_.batch,runtimeShape_.history,b));
    uint64_t rankClocks[5];
    uint32_t bitmapLength=runtimeShape_.history, limit=ctx->windowLimit/4, minimum=limit/2, rankCut=UINT32_MAX;
#if KVIP_PARALLEL_TERMINAL
    if (rankedEpoch_==runtimeShape_.rounds) {
        rank=2048;
        limit=2048;
        minimum=0;
    }
#elif KVIP_URMA_WINDOW_LI
    bitmapLength=valid?valid:runtimeShape_.history;
    rankCut=windowRankCut_;
#endif
#if KVIP_STAGE_TRACE
    const auto pipeSample = RankSample(query);
#endif
    KVIP_RANK_SAMPLE(selectScope, pipeSample, WindowTopKVf, TerminalTopKVf);
    uint32_t count=kvip_prefetch_rank::SelectBitmap(indicesOutLocal_,scoreOutLocal_,compact,
        topkSharedTmpLocal_[128],covered,bitmapLength,valid,rank,limit,rankTrace_?rankClocks:nullptr,minimum,
        windowCoveredEmpty_,rankCut);
    KVIP_TRACE_END(selectScope);
    if(rankTrace_) {
        uint64_t delta[3]={rankClocks[1]-rankClocks[0],rankClocks[4]-rankClocks[1],1};
        for(uint32_t k=0;k<3;++k) {
            if(incrementalMetadataActive_) {
                uint32_t i=incrementalMetadataIndex_*16+5+k;
                incrementalMetadata_.SetValue(i,incrementalMetadata_.GetValue(i)+delta[k]);
            } else {
                auto meta=IncrementalMeta(query);meta[5+k]+=delta[k];
            }
        }
    }
    mappedExport_=true;
    if(incrementalMetadataActive_)incrementalMetadata_.SetValue(incrementalMetadataIndex_*16+3,count);
    else {auto meta=IncrementalMeta(query);meta[3]=count;kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(meta),128);}
#if KVIP_PARALLEL_TERMINAL
    if(count && rankedEpoch_!=runtimeShape_.rounds) {
        GlobalTensor<uint32_t> jobs;jobs.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(rankedEpoch_==runtimeShape_.rounds?ctx->sourceBlocks:ctx->locations)+query*2048);
#else
    if(count) {
        GlobalTensor<uint32_t> jobs;jobs.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+query*2048);
#endif
#if KVIP_SERVICE_UB_CANDIDATES
        if (windowCandidatesActive_) {
            // Keep the original per-Q candidates and their enumeration order.
            // Only this AIV consumes the compact IDs, before leaving the window.
            DataCopy(windowCandidates_[(query % 4) * rotating::kWindowCandidatePitch],
                compact, (count + 7) / 8 * 8);
            PipeBarrier<PIPE_V>();
            return;
        }
#endif
        KVIP_RANK_SAMPLE(outputScope, pipeSample, WindowOutputMte3, TerminalTopKMte3);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID5);WaitFlag<HardEvent::V_MTE3>(EVENT_ID5);
        DataCopy(jobs,compact,(count+7)/8*8);
        if(mappedExport_){SetFlag<HardEvent::MTE3_V>(WINDOW_COLD_OUTPUT_EVENT);coldOutputPending_=true;}
        else {SetFlag<HardEvent::MTE3_V>(EVENT_ID5);WaitFlag<HardEvent::MTE3_V>(EVENT_ID5);}
    }
}
__aicore__ inline GM_ADDR WindowTopkPrefix() {
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    return reinterpret_cast<GM_ADDR>(ctx->incremental)+runtimeShape_.batch*512;
}
// Included in QLIVector. Use the existing merge/output/TopK scratch buffers.
// Raw scores and predicates die before TopK reuses its scratch. ID maps
// follow the LI trunk geometry; optional next-Q input uses only spare LI UB.
uint32_t rankEnd0_=0,rankEnd1_=0,rankHalf_=0;uint64_t rankReady_=0;
uint32_t incrementalCalls_=0,incrementalScanned_=0,incrementalCandidates_=0;
TBuf<TPosition::VECCALC> incrementalIdsBuf_;
LocalTensor<uint32_t> incrementalOld_;
LocalTensor<uint16_t> incrementalNew_;
LocalTensor<uint64_t> incrementalMetadata_;
bool incrementalMetadataActive_=false;
uint32_t incrementalMetadataIndex_=0;
// One additional Q: 4 KiB old scores, 32 KiB raw and 4 KiB bitmap.
// Old IDs load separately into their final UB buffer while candidates compact.
static constexpr uint32_t kIncrementalInputTokens=16384;
static constexpr uint32_t kIncrementalInputBytes=40960;
TBuf<TPosition::VECCALC> incrementalInputBuf_;
LocalTensor<uint16_t> incrementalInput_;
bool incrementalPipeline_=false,incrementalInputPending_=false,incrementalOutputPending_=false;
uint32_t incrementalFinalValid_=0;
uint32_t incrementalInputQuery_=0,incrementalInputBase_=0,incrementalInputCount_=0;
uint32_t incrementalNextQuery_=0,incrementalNextIndex_=0,incrementalNextEnd0_=0,incrementalNextEnd1_=0;
bool incrementalHasNext_=false;
__aicore__ inline __gm__ uint64_t* IncrementalMeta(uint32_t query) {
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    return reinterpret_cast<__gm__ uint64_t*>(rotating::TopKHeader(ctx))+query*16;
}
__aicore__ inline void SeedIncremental(uint32_t query,uint32_t row) {
    auto meta=IncrementalMeta(query);
    meta[0]=reinterpret_cast<uint64_t>(scoreGm.GetPhyAddr())+
        uint64_t(row)*runtimeShape_.history*2;
    meta[1]=uint64_t(runtimeShape_.resident)|(uint64_t(runtimeShape_.resident)<<32);
    meta[2]=0;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(meta),128);
}
#if KVIP_URMA_WINDOW_LI
// The staged window winners and the prefetch cutoff are two ranks of the
// same input. The terminal uses the ordinary dense prefix and cursor.
uint32_t windowRank_ = 0, windowRankCut_ = UINT32_MAX;
uint16_t windowFloor_ = 0;
__aicore__ inline bool MergeWindowRanks(uint32_t added, uint32_t base) {
    if (!windowRank_ || windowRank_ > 2048 || added > 1024 || topkOp_.GetSharedTmpBufferSize() < 30000)
        return false;
    auto scratch = topkSharedTmpLocal_;
    // Scratch words 0..3071 hold IDs and 3072..6143 hold their selected positions.
    auto hist = scratch[7152].template ReinterpretCast<uint16_t>();
    auto cuts = scratch[7232].template ReinterpretCast<uint16_t>();
    auto slots = scratch[3072];
    PipeBarrier<PIPE_V>();
    DataCopy(mrgValueLocal_, scoreOutLocal_, 2048);
    uint32_t floor = windowFloor_;
    if (!floor) {
        lifetch_incremental::Minimum(reinterpret_cast<__ubuf__ uint16_t *>(scoreOutLocal_.GetPhyAddr()),
                                     reinterpret_cast<__ubuf__ uint32_t *>(scratch.GetPhyAddr()));
        SetFlag<HardEvent::V_S>(EVENT_ID5);
        WaitFlag<HardEvent::V_S>(EVENT_ID5);
        floor = scratch.template ReinterpretCast<uint16_t>().GetValue(0);
    }
    auto keys = reinterpret_cast<__ubuf__ uint16_t *>(mrgValueLocal_.GetPhyAddr());
    auto cdf = reinterpret_cast<__ubuf__ uint16_t *>(hist.GetPhyAddr());
    uint32_t count = 2048 + added;
    PipeBarrier<PIPE_V>();
    kvip_window_ranks::AddScores(keys, cdf, floor, count);
    PipeBarrier<PIPE_V>();
    kvip_window_ranks::ReadCuts(cdf, reinterpret_cast<__ubuf__ uint16_t *>(cuts.GetPhyAddr()), floor, count,
                                windowRank_);
    SetFlag<HardEvent::V_S>(EVENT_ID5);
    WaitFlag<HardEvent::V_S>(EVENT_ID5);
    uint32_t cut = cuts.GetValue(0), rankCut = cuts.GetValue(128);
    if (cut < floor || rankCut < floor || cut - floor >= 64 || rankCut - floor >= 64)
        return false;
    DataCopy(scratch, indicesOutLocal_, 2048);
    lifetch_incremental::BuildAddedMap(reinterpret_cast<__ubuf__ uint16_t *>(incrementalNew_.GetPhyAddr()),
                                       reinterpret_cast<__ubuf__ uint32_t *>(scratch[2048].GetPhyAddr()), added,
                                       base);
    auto positions = reinterpret_cast<__ubuf__ uint32_t *>(slots.GetPhyAddr());
    kvip_window_ranks::ScoreSlots(keys, count, cut, positions);
    PipeBarrier<PIPE_V>();
    kvip_window_ranks::GatherPairs(
        positions, reinterpret_cast<__ubuf__ uint32_t *>(scratch.GetPhyAddr()), keys, 2048,
        reinterpret_cast<__ubuf__ uint32_t *>(indicesOutLocal_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t *>(scoreOutLocal_.GetPhyAddr()));
    SetFlag<HardEvent::V_S>(EVENT_ID5);
    WaitFlag<HardEvent::V_S>(EVENT_ID5);
    scratch.SetValue(0, cut);
    windowRankCut_ = rankCut;
    mappedExport_ = true;
    return true;
}
#endif
__aicore__ inline void FlushIncremental(uint32_t count,uint32_t base,bool clearCandidates=true KVIP_PIPE_PARAM_DEFAULT) {
    if(incrementalIdsPending_){WaitFlag<HardEvent::MTE2_V>(EVENT_ID7);incrementalIdsPending_=false;}
    mappedExport_=false;
    if(!count)return;
    ++incrementalCalls_;
    if(incrementalOutputPending_) {
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);incrementalOutputPending_=false;
    }
#if KVIP_URMA_WINDOW_LI
    if (MergeWindowRanks(count,base)) return;
#endif
    PipeBarrier<PIPE_V>();
    DataCopy(mrgValueLocal_,scoreOutLocal_,2048);
    DataCopy(incrementalOld_,indicesOutLocal_,2048);
    PipeBarrier<PIPE_V>();
    KVIP_RANK_SAMPLE(topkScope, pipeSample, WindowTopKVf, TerminalTopKVf);
    topkOp_(mrgValueLocal_,indicesOutLocal_,scoreOutLocal_,2048+((count+255)/256)*256,0,1,true);
    PipeBarrier<PIPE_V>();
    // 32000-byte TopK scratch: 2048 old IDs plus at most 5952 added IDs.
    if(count<=5952) {
        DataCopy(topkSharedTmpLocal_,incrementalOld_,2048);
        lifetch_incremental::BuildAddedMap(reinterpret_cast<__ubuf__ uint16_t*>(incrementalNew_.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(topkSharedTmpLocal_[2048].GetPhyAddr()),count,base);
        PipeBarrier<PIPE_V>();
        lifetch_incremental::MapAndExport(reinterpret_cast<__ubuf__ uint32_t*>(indicesOutLocal_.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(topkSharedTmpLocal_.GetPhyAddr()),reinterpret_cast<__ubuf__ uint16_t*>(scoreOutLocal_.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(incrementalNew_.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(topkSharedTmpLocal_.GetPhyAddr()),runtimeShape_.resident,mappedValid_,!exportWindowCold_);
        mappedExport_=true;
    } else {
    Simt::VF_CALL<lifetch_incremental::MapBitmap>(Simt::Dim3{256,1,1},
        reinterpret_cast<__ubuf__ uint32_t*>(indicesOutLocal_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(incrementalOld_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint16_t*>(incrementalNew_.GetPhyAddr()),base);
    }
    PipeBarrier<PIPE_V>();
    KVIP_TRACE_END(topkScope);
    if(clearCandidates)Duplicate(mrgValueLocal_[2048],uint16_t(0),trunkLen_);
}
// Skipped windows do not own state. Follow them to the last real service.
__aicore__ inline bool PreviousServiceReady(uint32_t epoch,uint32_t lane) {
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    auto events=reinterpret_cast<__gm__ volatile rotating::Event*>(ctx->events);
    while(epoch) {
        auto previous=events+(--epoch)*runtimeShape_.serviceLanes+lane;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ rotating::Event*>(previous)),128);
        if(!previous->complete)return false;
        if(!previous->skipped)return true;
    }
    return true;
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
    uint32_t query=incrementalNextQuery_,b=query/4,q=query%4;
    GlobalTensor<uint16_t> scores,raw,mask;
    GlobalTensor<uint32_t> ids;
    scores.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(WindowTopkPrefix()+b*(4*2048*6+1024))+q*2048);
    ids.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(WindowTopkPrefix()+b*(4*2048*6+1024)+4*2048*2)+q*2048);
    raw.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(source));
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    mask.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(ctx->hits)+uint64_t(query)*runtimeShape_.history/8);
    static_assert(V_MTE2_EVENT3 != VEC1_V_MTE2_EVENT_QSCALE &&
                  V_MTE2_EVENT3 != VEC1_V_MTE2_EVENT_QSCALE + 1);
#if KVIP_STAGE_TRACE
    const auto pipeSample = RankSample(query);
#endif
    KVIP_RANK_SAMPLE(preloadScope, pipeSample, WindowInputMte2, TerminalInputMte2);
    // Q-scale bank 0 owns a primed V_MTE2 event 6 throughout this window.
    // Use the balanced scratch event: wait for this Q's reads, not that signal.
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
    DataCopy(incrementalInput_,scores,2048);

    DataCopy(incrementalInput_[2048],raw[base],n);
    DataCopy(incrementalInput_[2048+kIncrementalInputTokens],mask[base/8],n/8);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID6);
    incrementalInputQuery_=query;incrementalInputBase_=base;incrementalInputCount_=n;
    incrementalInputPending_=true;
}
__aicore__ inline void AdvanceIncremental(uint32_t query,uint32_t end0,uint32_t end1,uint32_t half) {
#if KVIP_URMA_WINDOW_LI
    windowRank_=0; windowRankCut_=UINT32_MAX;
#endif
    constexpr uint32_t prefixBytes=4*2048*6+1024;
    auto meta=IncrementalMeta(query);
    if(!incrementalMetadataActive_)kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(meta),128);
    uint32_t mi=incrementalMetadataIndex_*16;
    uint64_t source=incrementalMetadataActive_?incrementalMetadata_.GetValue(mi):meta[0];
    uint64_t cursor=incrementalMetadataActive_?incrementalMetadata_.GetValue(mi+1):meta[1];
    uint32_t prior[2]={uint32_t(cursor),uint32_t(cursor>>32)};
    uint32_t ends[2]={end0,end1};
    rankEnd0_=prior[0];rankEnd1_=prior[1];
    if (!source) return;
#if KVIP_STAGE_TRACE
    const auto pipeSample = RankSample(query);
#endif
    const uint32_t callsBefore=incrementalCalls_;
    mappedExport_=false;
    if (end0<=prior[0] && (!half || end1<=prior[1])) {
        JoinColdOutput();
        if(exportWindowCold_) {
            GlobalTensor<uint32_t> saved; saved.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(WindowTopkPrefix()+uint64_t(query/4)*prefixBytes+4*2048*2)+(query%4)*2048);
            GlobalTensor<uint16_t> savedScores;
            savedScores.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(WindowTopkPrefix()+uint64_t(query/4)*prefixBytes)+(query%4)*2048);
            KVIP_RANK_SAMPLE(loadScope, pipeSample, WindowInputMte2, TerminalInputMte2);
            PipeBarrier<PIPE_ALL>();DataCopy(indicesOutLocal_,saved,2048);DataCopy(scoreOutLocal_,savedScores,2048);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID5);WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
            KVIP_TRACE_END(loadScope);
            ExportWindowCold(query);
        }
        return;
    }
    const uint32_t batch=query/4,q=query%4;
    mappedValid_=incrementalFinalValid_ ? Min(runtimeShape_.history,incrementalFinalValid_) : runtimeShape_.history;
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    if(ctx->queryEnds)mappedValid_=Min(mappedValid_,reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds)[query]);
    GlobalTensor<uint16_t> stateScores,raw,bitmap;
    GlobalTensor<uint32_t> stateIds;
    stateScores.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(WindowTopkPrefix()+batch*prefixBytes)+q*2048);
    stateIds.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(WindowTopkPrefix()+batch*prefixBytes+4*2048*2)+q*2048);
    raw.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(source));
    bitmap.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(ctx->hits)+uint64_t(query)*runtimeShape_.history/8);
    if(!incrementalInputPending_ && incrementalPipeline_ && incrementalMetadataActive_) {
        uint32_t nq=incrementalNextQuery_,ni=incrementalNextIndex_,e0=incrementalNextEnd0_,e1=incrementalNextEnd1_;bool hn=incrementalHasNext_;
        incrementalNextQuery_=query;incrementalNextIndex_=incrementalMetadataIndex_;incrementalNextEnd0_=end0;incrementalNextEnd1_=end1;incrementalHasNext_=true;
        PreloadIncremental(half);
        incrementalNextQuery_=nq;incrementalNextIndex_=ni;incrementalNextEnd0_=e0;incrementalNextEnd1_=e1;incrementalHasNext_=hn;
    }
    KVIP_RANK_SAMPLE(savedScope, pipeSample, WindowInputMte2, TerminalInputMte2);
    bool staged=incrementalInputPending_ && incrementalInputQuery_==query;
    incrementalIdsPending_=staged;
    if(staged) {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID6);
        DataCopy(scoreOutLocal_,incrementalInput_,2048);
        DataCopy(indicesOutLocal_,stateIds,2048);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID7);
        PipeBarrier<PIPE_V>();
        incrementalInputPending_=false;
    } else {
        PipeBarrier<PIPE_ALL>();
        DataCopy(scoreOutLocal_,stateScores,2048);DataCopy(indicesOutLocal_,stateIds,2048);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID7);incrementalIdsPending_=true;
    }
    KVIP_TRACE_END(savedScope);
    // Join state and first score tile together below; they use disjoint UB.
    auto rawLocal=topkSharedTmpLocal_.template ReinterpretCast<uint16_t>();
    const uint32_t rawCapacity=((topkOp_.GetSharedTmpBufferSize()-128)*4/9/128)*128;
    auto scratchRaw=rawLocal;
    auto maskLocal=rawLocal[rawCapacity];
    auto scratchMask=maskLocal;
    auto countLocal=maskLocal[rawCapacity/8].template ReinterpretCast<uint32_t>();
    uint32_t packed=0,packBase=0;
#if KVIP_STAGE_TRACE
    bool rawRecorded = false;
#endif
    const uint16_t cutoff=uint16_t(incrementalMetadataActive_?incrementalMetadata_.GetValue(mi+2):meta[2]);
    Duplicate(mrgValueLocal_[2048],uint16_t(0),trunkLen_);
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
            rawLocal=incrementalInput_[2048];maskLocal=incrementalInput_[2048+kIncrementalInputTokens];
            staged=false;
        } else {
            rawLocal=scratchRaw;maskLocal=scratchMask;
#if KVIP_STAGE_TRACE
        auto rawSample = pipeSample;
        if (rawRecorded) rawSample.base = 0;
        rawRecorded = true;
#endif
        KVIP_RANK_SAMPLE(rawScope, rawSample, WindowInputMte2, TerminalInputMte2);
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
        JoinColdOutput();
        lifetch_incremental::CompactSparseTiles(
            reinterpret_cast<__ubuf__ uint16_t*>(rawLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint16_t*>(maskLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint16_t*>(tileList.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint16_t*>(mrgValueLocal_[2048+packed].GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint16_t*>(incrementalNew_[packed].GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(countLocal.GetPhyAddr()),blocks,uint16_t(base-packBase),cutoff);
        SetFlag<HardEvent::V_S>(EVENT_ID5);WaitFlag<HardEvent::V_S>(EVENT_ID5);
        uint32_t count=countLocal.GetValue(0);packed+=count;incrementalScanned_+=n;incrementalCandidates_+=count;
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
        incrementalScanned_+=n;incrementalCandidates_+=count;
        if (count) {
            JoinColdOutput();
            // Capacity is measured in compact candidates, not raw score tiles.
            // Reuse the demo's uint16 relative IDs across both sequence halves.
            if(packed && (packed+count>trunkLen_ || base+n-packBase>65536)) {
                FlushIncremental(packed,packBase,true KVIP_PIPE_ARG);packed=0;
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
#if KVIP_URMA_WINDOW_LI
    if (packed && packed<=1024 && exportWindowCold_ && rankedEpoch_<runtimeShape_.rounds) {
        const auto recent=kvip_recent_index::RecentRange(ctx,query/4,mappedValid_);
        uint32_t covered;
        windowRank_=WindowRank(ctx, mappedValid_, prior[0], prior[1], half, recent, covered);
        windowFloor_=cutoff;
    }
#endif
    FlushIncremental(packed,packBase,false KVIP_PIPE_ARG);
    PipeBarrier<PIPE_V>();
    // A nonzero cutoff already describes the unchanged TopK when this
    // window did not flush any candidates. Final TopK keeps its old path.
    uint32_t threshold=cutoff;
    if(!exportWindowCold_ || mappedExport_ || !cutoff || incrementalCalls_!=callsBefore) {
        if(!mappedExport_)lifetch_incremental::Minimum(reinterpret_cast<__ubuf__ uint16_t*>(scoreOutLocal_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(topkSharedTmpLocal_.GetPhyAddr()));
        SetFlag<HardEvent::V_S>(EVENT_ID5);WaitFlag<HardEvent::V_S>(EVENT_ID5);
        threshold=topkSharedTmpLocal_.template ReinterpretCast<uint16_t>().GetValue(0);
    }
    rankEnd0_=prior[0];rankEnd1_=prior[1];
    if(exportWindowCold_)ExportWindowCold(query);
    KVIP_RANK_SAMPLE(outputScope, pipeSample, WindowOutputMte3, TerminalTopKMte3);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID5);WaitFlag<HardEvent::V_MTE3>(EVENT_ID5);
    if(incrementalPipeline_ && incrementalMetadataActive_) {
        if(incrementalOutputPending_) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);incrementalOutputPending_=false;
        }
        // TopK prefix/old-ID maps are dead until the NEXT FlushIncremental.
        // Stage here so next Q can overwrite working outputs immediately.
        DataCopy(mrgValueLocal_,scoreOutLocal_,2048);
        DataCopy(incrementalOld_,indicesOutLocal_,2048);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID6);WaitFlag<HardEvent::V_MTE3>(EVENT_ID6);
        DataCopy(stateScores,mrgValueLocal_,2048);DataCopy(stateIds,incrementalOld_,2048);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID6);incrementalOutputPending_=true;
    } else {
        DataCopy(stateScores,scoreOutLocal_,2048);DataCopy(stateIds,indicesOutLocal_,2048);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID5);WaitFlag<HardEvent::MTE3_S>(EVENT_ID5);
    }
    KVIP_TRACE_END(outputScope);
    if(incrementalMetadataActive_) {
        incrementalMetadata_.SetValue(mi+1,uint64_t(prior[0])|(uint64_t(prior[1])<<32));
        incrementalMetadata_.SetValue(mi+2,threshold);
    } else {
        meta[1]=uint64_t(prior[0])|(uint64_t(prior[1])<<32);meta[2]=threshold;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(meta),128);
    }
}
__aicore__ inline bool AdvanceWindow(uint32_t epoch,uint32_t half,uint32_t lane) {
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    GlobalTensor<uint64_t> metadata;
    metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(rotating::TopKHeader(ctx)));
#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
    const bool p8Row = lane == 0 && coldParts_ == 8 && kvip_window_probe::BulkShape(runtimeShape_);
    const bool earlyShape = lane < runtimeShape_.batch && !coldParts_ && !dedicatedProgress_ &&
        kvip_window_probe::EarlyShape(runtimeShape_);
    const auto option = p8Row || earlyShape ? kvip_window_probe::Option(streamingWorkspace_) : 0;
    auto topkReady = earlyShape ? kvip_window_probe::Ready(option) : nullptr;
    bool headerIssued = false;
#endif
    // The predecessor wait is part of the optional window service and can
    // delay the producer. Capture it separately from incremental computation.
    bool trace=streamingWorkspace_ && (reinterpret_cast<__gm__ pipeline_ab::Args*>(streamingWorkspace_)->prefetchCount & 0x80000000U);
    if(trace) {
        auto event=reinterpret_cast<__gm__ rotating::Event*>(ctx->events)+epoch*runtimeShape_.serviceLanes+lane;
        event->waitBegin=GetSystemCycle();
    }
    // Same predecessor dependency as original FastService, moved before the
    // added calculation so private per-lane state cannot have two writers.
    if(epoch) {
        KVIP_TRACE_BEGIN(predecessorScope, KVIP_TRACE_WORKSPACE(streamingWorkspace_),
            Cold, ServicePredecessor, epoch, lane);
        auto previous=reinterpret_cast<__gm__ volatile rotating::Event*>(ctx->events)+(epoch-1)*runtimeShape_.serviceLanes+lane;
        auto previousBytes = reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ rotating::Event*>(previous));
        do {
            kvaf_device::CleanCacheLines(previousBytes,64);
#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
            if (topkReady && !headerIssued && !previous->complete) {
                const uint32_t index = (epoch - 1) * runtimeShape_.serviceLanes + lane;
                auto flag = topkReady + index * kvip_window_probe::kReadyWords;
                kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flag),
                    kvip_window_probe::kReadyBytes);
                if (*reinterpret_cast<__gm__ volatile uint32_t*>(flag)) {
                    KVIP_TRACE_BEGIN(preloadScope, KVIP_TRACE_WORKSPACE(streamingWorkspace_),
                        Cold, EarlyTopKHeaderEnqueue, epoch, lane);
                    // Live cursors, row admissions and WQE still use complete.
                    DataCopy(incrementalMetadata_,metadata[lane*64],64);
                    headerIssued = true;
                }
            }
#endif
        } while(!previous->complete);
    }
    KVIP_TRACE_BEGIN(windowScope, KVIP_TRACE_WORKSPACE(streamingWorkspace_), Cold, LiWindow, epoch, lane);
    KVIP_TRACE_BEGIN(readyScope, KVIP_TRACE_WORKSPACE(streamingWorkspace_), Cold, WindowReadyLoad, epoch, lane);
    uint64_t begin=trace?GetSystemCycle():0;
    exportWindowCold_=true;rankedEpoch_=epoch;rankTrace_=trace;
    incrementalCalls_=0;incrementalScanned_=0;incrementalCandidates_=0;
    const bool stagedMetadata=(runtimeShape_.batch+runtimeShape_.serviceLanes-1)/runtimeShape_.serviceLanes<=2;
    incrementalMetadataActive_=stagedMetadata;
    if(incrementalMetadataActive_) {
#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
    if (!coldParts_ && !dedicatedProgress_)
#endif
    {
    GlobalTensor<uint32_t> progressSource;progressSource.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->progress));
    for(uint32_t b=lane,i=0;b<runtimeShape_.batch;b+=runtimeShape_.serviceLanes,i+=128)
        DataCopy(incrementalOld_[i],progressSource[b*128],128);
    }
    // Stage the first 64 bytes of each row after progress. At most two rows
    // use words [256, 288), disjoint from progress words [0, 256).
    GlobalTensor<uint32_t> rowsSource;rowsSource.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->rows));
    for(uint32_t b=lane,i=0;b<runtimeShape_.batch;b+=runtimeShape_.serviceLanes,++i)
        DataCopy(incrementalOld_[256+i*16],rowsSource[b*256],16);
    }
    if(incrementalMetadataActive_) {
#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
        if (!headerIssued)
#endif
        {
            for(uint32_t b=lane,i=0;b<runtimeShape_.batch;b+=runtimeShape_.serviceLanes,i+=64)
                DataCopy(incrementalMetadata_[i],metadata[b*64],64);
        }
        SetFlag<HardEvent::MTE2_S>(EVENT_ID5);WaitFlag<HardEvent::MTE2_S>(EVENT_ID5);
    }
    uint32_t ownedQueries=lane<runtimeShape_.batch?((runtimeShape_.batch-1-lane)/runtimeShape_.serviceLanes+1)*4:0;
    uint32_t requests0=0,requests1=0;
    if(incrementalMetadataActive_) {
        if(ownedQueries>0)requests0=incrementalOld_.GetValue(256+9);
        if(ownedQueries>4)requests1=incrementalOld_.GetValue(256+16+9);
#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
        if (dedicatedProgress_) {
            for (uint32_t b = lane, index = 0; b < runtimeShape_.batch; b += runtimeShape_.serviceLanes) {
                for (uint32_t q = 0; q < 4; ++q, ++index)
                    incrementalMetadata_.SetValue(index * 16 + 4, DedicatedColdReady(b * 4 + q, epoch, half));
            }
        } else if (coldParts_) {
            KVIP_TRACE_BEGIN(cursorScope, p8Row ? KVIP_TRACE_WORKSPACE(streamingWorkspace_) : 0,
                Cold, WindowCursorLoad, epoch, lane);
            if (p8Row && option == kvip_window_probe::kBulk) {
                constexpr uint32_t kParts = 8;
                constexpr uint32_t kWords = kvip_cold_score::kQueries * kParts * kvip_cold_score::kProgressWords;
                static_assert(kWords * sizeof(uint32_t) == 2048, "P8/Q4 cursor snapshot size");
                // Use old-ID scratch after the staged row and before any TopK reads.
                auto snapshot = incrementalOld_[512];
                KVIP_TRACE_BEGIN(copyScope, KVIP_TRACE_WORKSPACE(streamingWorkspace_),
                    Cold, BulkCursorMte2, epoch, lane);
                DataCopy(snapshot, coldProgressGm_, kWords);
                SetFlag<HardEvent::MTE2_S>(EVENT_ID5);WaitFlag<HardEvent::MTE2_S>(EVENT_ID5);
                KVIP_TRACE_END(copyScope);
                for (uint32_t q = 0; q < kvip_cold_score::kQueries; ++q) {
                    uint32_t cursors[kParts];
                    for (uint32_t part = 0; part < kParts; ++part)
                        cursors[part] = snapshot.GetValue(kvip_cold_score::ProgressOffset(q, part, kParts));
                    incrementalMetadata_.SetValue(q * 16 + 4, FoldPartitionedColdReady(cursors, half));
                }
            } else {
                for (uint32_t b = lane, index = 0; b < runtimeShape_.batch; b += runtimeShape_.serviceLanes) {
                    for (uint32_t q = 0; q < 4; ++q, ++index)
                        incrementalMetadata_.SetValue(index * 16 + 4, PartitionedColdReady(b * 4 + q, half));
                }
            }
        } else
#endif
        {
        for(uint32_t i=0;i<ownedQueries;++i)
            incrementalMetadata_.SetValue(i*16+4,uint64_t(incrementalOld_.GetValue(i*32))|(uint64_t(incrementalOld_.GetValue(i*32+16))<<32));
        }
    }
    uint32_t cap=half?rotating::SequenceWindowEnd(runtimeShape_.history,runtimeShape_.resident,half,epoch):
        rotating::Trigger(runtimeShape_.history,runtimeShape_.resident,runtimeShape_.rounds,epoch)*128;
#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
    // The benchmark uses the native BS16 shape with one row per Service
    // lane. Record the staged snapshot before TopK and next-Q preloading.
    if (stagedMetadata && ownedQueries == kvip_window_frontier::kQueries) {
        const auto mode = kvip_window_frontier::ActiveMode(streamingWorkspace_);
        if (mode == kvip_window_frontier::Capture || mode == kvip_window_frontier::Replay) {
            const auto table = kvip_window_frontier::Table(streamingWorkspace_);
            const auto record = kvip_window_frontier::Row(table, runtimeShape_.batch, epoch, lane);
            auto words = reinterpret_cast<__gm__ uint64_t*>(record);
            if (mode == kvip_window_frontier::Replay) {
                kvaf_device::CleanCacheLines(record, kvip_window_frontier::kRowBytes);
            }
            for (uint32_t q = 0; q < kvip_window_frontier::kQueries; ++q) {
                const auto metadataIndex = q * 16 + 4;
                if (mode == kvip_window_frontier::Capture) {
                    words[q] = incrementalMetadata_.GetValue(metadataIndex);
                } else {
                    const auto saved = words[q];
                    const auto observed = incrementalMetadata_.GetValue(metadataIndex);
                    auto cursor = reinterpret_cast<__gm__ uint32_t*>(ctx->progress) + (lane * 4 + q) * 32;
                    kvip_window_frontier::WaitClassified(cursor, observed, saved, cap);
                    incrementalMetadata_.SetValue(metadataIndex, saved);
                    words[kvip_window_frontier::kQueries + q] = incrementalMetadata_.GetValue(metadataIndex);
                }
            }
            kvaf_device::CleanCacheLines(record, kvip_window_frontier::kRowBytes);
        }
    }
#endif
#if KVIP_SERVICE_UB_CANDIDATES
    // Fall back for a sparse/not-yet-ready row. The GM path can legitimately
    // retain a skipped Q's old count and IDs, which the private arena lacks.
    windowCandidatesActive_ = windowCandidatesAllocated_ && half != 0
        && ownedQueries == 4 && stagedMetadata && ctx->windowLimit <= 4 * rotating::kWindowCandidatePitch;
    if (windowCandidatesActive_) {
        for (uint32_t q = 0; q < 4; ++q) {
            if (!incrementalMetadata_.GetValue(q * 16) ||
                Min(uint32_t(incrementalMetadata_.GetValue(q * 16 + 4)), cap) < runtimeShape_.resident)
                windowCandidatesActive_ = false;
        }
    }
#endif
    KVIP_TRACE_END(readyScope);
    for(uint32_t b=lane;b<runtimeShape_.batch;b+=runtimeShape_.serviceLanes) for(uint32_t q=0;q<4;++q) {
        // Init clears covered; the posting path records every admission in
        // requests. The predecessor fence makes the row and bitmap visible.
        windowCoveredEmpty_=incrementalMetadataActive_ && (((b-lane)/runtimeShape_.serviceLanes)==0 ? requests0==0 : requests1==0);
        uint32_t progressIndex=(((b-lane)/runtimeShape_.serviceLanes)*4+q)*16+4;
        uint64_t ready;
        if(incrementalMetadataActive_)ready=incrementalMetadata_.GetValue(progressIndex);
#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
        else if (dedicatedProgress_) ready = DedicatedColdReady(b * 4 + q, epoch, half);
        else if (coldParts_) ready = PartitionedColdReady(b * 4 + q, half);
#endif
        else {
            auto progress=reinterpret_cast<__gm__ volatile uint32_t*>(ctx->progress)+(b*4+q)*32;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ uint32_t*>(progress)),128);
            ready=uint64_t(progress[0])|(uint64_t(progress[16])<<32);
        }
        uint32_t end0=Min(uint32_t(ready),cap),end1=half?Min(uint32_t(ready>>32),cap):0;
        rankReady_=ready;rankHalf_=half;
        if(end0<runtimeShape_.resident)continue;
        incrementalMetadataIndex_=((b-lane)/runtimeShape_.serviceLanes)*4+q;
        uint32_t nextB=q==3?b+runtimeShape_.serviceLanes:b,nextQ=(q+1)%4;
        incrementalHasNext_=incrementalPipeline_ && incrementalMetadataActive_ && nextB<runtimeShape_.batch;
        if(incrementalHasNext_) {
            incrementalNextQuery_=nextB*4+nextQ;incrementalNextIndex_=incrementalMetadataIndex_+1;
            uint64_t nextReady=incrementalMetadata_.GetValue(progressIndex+16);
            incrementalNextEnd0_=Min(uint32_t(nextReady),cap);
            incrementalNextEnd1_=half?Min(uint32_t(nextReady>>32),cap):0;
        }
        KVIP_TRACE_BEGIN(queryScope, KVIP_TRACE_WORKSPACE(streamingWorkspace_),
            Cold, WindowQueryTopK, epoch, b * 4 + q);
        AdvanceIncremental(b*4+q,end0,end1,half);
    }
    KVIP_TRACE_BEGIN(flushScope, KVIP_TRACE_WORKSPACE(streamingWorkspace_), Cold, WindowStateFlush, epoch, lane);
    JoinColdOutput();
    if(incrementalInputPending_) {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID6);incrementalInputPending_=false;
    }
    incrementalHasNext_=false;
    if(incrementalOutputPending_) {
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);incrementalOutputPending_=false;
    }
    if(incrementalMetadataActive_) {
        SetFlag<HardEvent::S_MTE3>(EVENT_ID5);WaitFlag<HardEvent::S_MTE3>(EVENT_ID5);
        for(uint32_t b=lane,i=0;b<runtimeShape_.batch;b+=runtimeShape_.serviceLanes,i+=64)
            DataCopy(metadata[b*64],incrementalMetadata_[i],64);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID5);WaitFlag<HardEvent::MTE3_S>(EVENT_ID5);
        incrementalMetadataActive_=false;
#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
        if (topkReady && epoch + 1 < runtimeShape_.rounds) {
            const uint32_t index = epoch * runtimeShape_.serviceLanes + lane;
            auto flag = topkReady + index * kvip_window_probe::kReadyWords;
            *flag = 1;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flag),
                kvip_window_probe::kReadyBytes);
        }
#endif
    }
    exportWindowCold_=false;
    // The producer's progress lookup table shares the TopK score buffer.
    // Rebuild it before LI resumes; otherwise score bytes become fake progress.
#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
    if (!dedicatedProgress_)
#endif
    PrepareRotationProgress();
    KVIP_TRACE_END(flushScope);
    if(trace) {
        auto event=reinterpret_cast<__gm__ rotating::Event*>(ctx->events)+epoch*runtimeShape_.serviceLanes+lane;
        auto stamp=reinterpret_cast<__gm__ uint64_t*>(event->pad);
        stamp[2]=begin;stamp[3]=GetSystemCycle();stamp[4]=uint64_t(incrementalCalls_) | (uint64_t(incrementalScanned_)<<16) | (uint64_t(incrementalCandidates_)<<40);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(event),128);
    }
    return stagedMetadata;
}
__aicore__ inline void FinalIncremental(uint32_t query,uint32_t valid) {
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    if(runtimeShape_.rounds) {
        uint32_t lane=(query/4)%runtimeShape_.serviceLanes;
        // Only the final consumer must join outstanding state; never spin in
        // an optional intermediate window.
        while(!PreviousServiceReady(runtimeShape_.rounds,lane)) {}
    }
    uint32_t half=rotating::UseSequenceSplit(runtimeShape_)?rotating::SequenceHalfWindow(runtimeShape_):0;
    uint32_t alignedValid=(valid+127U)&~127U;
    incrementalFinalValid_=valid;
    AdvanceIncremental(query,alignedValid,half?alignedValid:0,half);
    incrementalFinalValid_=0;
    constexpr uint32_t prefixBytes=4*2048*6+1024;
    GlobalTensor<uint32_t> ids;
    ids.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(WindowTopkPrefix()+(query/4)*prefixBytes+4*2048*2)+(query%4)*2048);
    PipeBarrier<PIPE_ALL>();DataCopy(indicesOutLocal_,ids,2048);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID5);WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
    auto initial=streamingTopkOnly_?streamingOutput_:topkSharedTmpLocal_;
    kvip_prefetch_rank::RestorePublicIds(reinterpret_cast<__ubuf__ uint32_t*>(indicesOutLocal_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(initial.GetPhyAddr()),runtimeShape_.resident);
    PipeBarrier<PIPE_V>();taggedLocations_=true;

}
