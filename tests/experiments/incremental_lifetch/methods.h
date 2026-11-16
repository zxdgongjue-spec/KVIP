// Included in QLIVector. Use the existing merge/output/TopK scratch buffers.
// Raw scores and predicates die before TopK reuses its scratch. Only ID maps
// need additional UB, sized by the unchanged LI trunk geometry.
TBuf<TPosition::VECCALC> incrementalIdsBuf_;
LocalTensor<uint32_t> incrementalOld_;
LocalTensor<uint16_t> incrementalNew_;
__aicore__ inline __gm__ uint64_t* IncrementalMeta(uint32_t query) {
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    return reinterpret_cast<__gm__ uint64_t*>(ctx->incremental)+query*16;
}
__aicore__ inline void SeedIncremental(uint32_t query,uint32_t row) {
    auto meta=IncrementalMeta(query);
    meta[0]=reinterpret_cast<uint64_t>(scoreGm.GetPhyAddr())+
        uint64_t(row)*runtimeShape_.history*2;
    meta[1]=uint64_t(runtimeShape_.resident)|(uint64_t(runtimeShape_.resident)<<32);
    meta[2]=0;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(meta),128);
}
__aicore__ inline void AdvanceIncremental(uint32_t query,uint32_t end0,uint32_t end1,uint32_t half) {
    constexpr uint32_t prefixBytes=4*2048*6+1024;
    auto meta=IncrementalMeta(query);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(meta),128);
    uint64_t source=meta[0], cursor=meta[1];
    uint32_t prior[2]={uint32_t(cursor),uint32_t(cursor>>32)};
    uint32_t ends[2]={end0,end1};
    if (!source || (end0<=prior[0] && (!half || end1<=prior[1]))) return;
    const uint32_t batch=query/4,q=query%4;
    GlobalTensor<uint16_t> stateScores,raw,bitmap;
    GlobalTensor<uint32_t> stateIds;
    stateScores.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(prefixOutput_+batch*prefixBytes)+q*2048);
    stateIds.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(prefixOutput_+batch*prefixBytes+4*2048*2)+q*2048);
    raw.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(source));
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    bitmap.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(ctx->hits)+uint64_t(query)*runtimeShape_.history/8);
    PipeBarrier<PIPE_ALL>();
    DataCopy(scoreOutLocal_,stateScores,2048);DataCopy(indicesOutLocal_,stateIds,2048);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID5);WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
    auto rawLocal=topkSharedTmpLocal_.template ReinterpretCast<uint16_t>();
    auto maskLocal=rawLocal[trunkLen_];
    auto countLocal=maskLocal[trunkLen_/8].template ReinterpretCast<uint32_t>();
    for (uint32_t base=Min(prior[0],half?prior[1]:prior[0]);base<Max(end0,half?end1:end0);) {
        uint32_t shard=0,limit=end0;
        if (half) {
            auto slice=rotating::SequenceSliceAt(base,runtimeShape_.resident,half);
            shard=slice.shard;limit=Min(slice.end,ends[shard]);
            if (base<prior[shard]) {base=Min(slice.end,prior[shard]);continue;}
            if (base>=limit) {base=slice.end;continue;}
        }
        uint32_t n=Min(trunkLen_,limit-base);
        if (!n) break;
        // Published boundaries are whole 128-score tiles. Final causal bounds
        // use the ordinary path unless aligned, avoiding padded score reads.
        PipeBarrier<PIPE_ALL>();
        DataCopy(mrgValueLocal_,scoreOutLocal_,2048);
        DataCopy(incrementalOld_,indicesOutLocal_,2048);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID5);WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
        DataCopy(rawLocal,raw[base],n);DataCopy(maskLocal,bitmap[base/8],n/8);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID5);WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
        lifetch_incremental::Count(reinterpret_cast<__ubuf__ uint16_t*>(maskLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(countLocal.GetPhyAddr()),n);
        SetFlag<HardEvent::V_S>(EVENT_ID5);WaitFlag<HardEvent::V_S>(EVENT_ID5);
        uint32_t count=countLocal.GetValue(0);
        if (count) {
            uint32_t padded=(count+255)/256*256;
            if(count%256) Duplicate(mrgValueLocal_[2048+(count/256)*256],uint16_t(0),256);
            PipeBarrier<PIPE_V>();
            lifetch_incremental::Compact(reinterpret_cast<__ubuf__ uint16_t*>(rawLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint16_t*>(maskLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint16_t*>(mrgValueLocal_[2048].GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint16_t*>(incrementalNew_.GetPhyAddr()),n);
            PipeBarrier<PIPE_V>();
            topkOp_(mrgValueLocal_,indicesOutLocal_,scoreOutLocal_,2048+padded,0,1,true);
            PipeBarrier<PIPE_V>();
            Simt::VF_CALL<lifetch_incremental::MapBitmap>(Simt::Dim3{256,1,1},
                reinterpret_cast<__ubuf__ uint32_t*>(indicesOutLocal_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(incrementalOld_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint16_t*>(incrementalNew_.GetPhyAddr()),base);
        }
        base+=n;prior[shard]=base;
    }
    PipeBarrier<PIPE_ALL>();
    lifetch_incremental::Minimum(reinterpret_cast<__ubuf__ uint16_t*>(scoreOutLocal_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(topkSharedTmpLocal_.GetPhyAddr()));
    SetFlag<HardEvent::V_S>(EVENT_ID5);WaitFlag<HardEvent::V_S>(EVENT_ID5);
    uint32_t threshold=topkSharedTmpLocal_.template ReinterpretCast<uint16_t>().GetValue(0);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID5);WaitFlag<HardEvent::V_MTE3>(EVENT_ID5);
    DataCopy(stateScores,scoreOutLocal_,2048);DataCopy(stateIds,indicesOutLocal_,2048);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID5);WaitFlag<HardEvent::MTE3_S>(EVENT_ID5);
    meta[1]=uint64_t(prior[0])|(uint64_t(prior[1])<<32);meta[2]=threshold;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(meta),128);
}
__aicore__ inline void AdvanceWindow(uint32_t epoch,uint32_t half) {
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    uint32_t lane=rotating::ServiceLane(runtimeShape_.batch,runtimeShape_.splitQ,
        runtimeShape_.serviceAics,blockId_/2,blockId_%2);
    if(epoch) {
        auto event=reinterpret_cast<__gm__ volatile rotating::Event*>(ctx->events)+
            (epoch-1)*runtimeShape_.serviceLanes+lane;
        do {kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ rotating::Event*>(event)),128);}while(!event->complete);
    }
    uint32_t cap=half?rotating::SequenceWindowEnd(runtimeShape_.history,runtimeShape_.resident,half,epoch):
        rotating::Trigger(runtimeShape_.history,runtimeShape_.resident,runtimeShape_.rounds,epoch)*128;
    for(uint32_t b=lane;b<runtimeShape_.batch;b+=runtimeShape_.serviceLanes) for(uint32_t q=0;q<4;++q) {
        auto progress=reinterpret_cast<__gm__ volatile uint32_t*>(ctx->progress)+(b*4+q)*32;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ uint32_t*>(progress)),128);
        uint32_t end0=Min(uint32_t(progress[0]),cap),end1=half?Min(uint32_t(progress[16]),cap):0;
        if(end0<runtimeShape_.resident)continue;
        AdvanceIncremental(b*4+q,end0,end1,half);
    }
    // The producer's progress lookup table shares the TopK score buffer.
    // Rebuild it before LI resumes; otherwise score bytes become fake progress.
    PrepareRotationProgress();
}
__aicore__ inline void FinalIncremental(uint32_t query,uint32_t valid) {
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    if(runtimeShape_.rounds) {
        uint32_t lane=(query/4)%runtimeShape_.serviceLanes;
        auto event=reinterpret_cast<__gm__ volatile rotating::Event*>(ctx->events)+
            (runtimeShape_.rounds-1)*runtimeShape_.serviceLanes+lane;
        do {kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ rotating::Event*>(event)),128);}while(!event->complete);
    }
    uint32_t half=rotating::UseSequenceSplit(runtimeShape_)?rotating::SequenceHalfWindow(runtimeShape_):0;
    AdvanceIncremental(query,valid,half?valid:0,half);
    constexpr uint32_t prefixBytes=4*2048*6+1024;
    GlobalTensor<uint32_t> ids;
    ids.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(prefixOutput_+(query/4)*prefixBytes+4*2048*2)+(query%4)*2048);
    PipeBarrier<PIPE_ALL>();DataCopy(indicesOutLocal_,ids,2048);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID5);WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
}
