// SPDX-License-Identifier: Apache-2.0
// Members of QLIVector. One member of every prior service epoch becomes a
// last-window worker after publishing its own final cold score tile.
__aicore__ inline bool HasLastWindowTeam() const {
    const auto& s=runtimeShape_;
    return s.splitQ && s.rounds==4 && s.serviceLanes==s.batch &&
        s.serviceLanes==2*s.serviceAics && s.liCores==4*s.serviceAics &&
        s.liCores<=s.aicCount;
}

// One aligned 128-byte TopK header per Q. Words [0..7] are its existing
// computation state. The second 64-byte line is private to this handshake.
struct LastWindowMail {
    uint64_t phase, ready, coveredEmpty, calls, scanned, candidates, snapshot;
};
static_assert(sizeof(LastWindowMail)<=64);
__aicore__ inline __gm__ volatile LastWindowMail* LastMail(uint32_t query) {
    return reinterpret_cast<__gm__ volatile LastWindowMail*>(IncrementalMeta(query)+8);
}
__aicore__ inline void FlushLastMail(__gm__ volatile LastWindowMail* mail) {
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ LastWindowMail*>(mail)),64);
}
__aicore__ inline void ResetLastWindowMail(uint32_t query) {
    if(HasLastWindowTeam()) {
        auto mail=LastMail(query);
        mail->phase=0;
        FlushLastMail(mail);
    }
}
__aicore__ inline void WaitLastWindowMail(__gm__ volatile LastWindowMail* mail, uint64_t phase) {
    do { FlushLastMail(mail); } while(mail->phase!=phase);
}
__aicore__ inline void SendLastWindowMail(__gm__ volatile LastWindowMail* mail,uint64_t phase) {
    // Flush the payload before publishing the phase word. Pollers invalidate
    // their cached line on every check; there is exactly one writer per phase.
    FlushLastMail(mail);
    mail->phase=phase;
    FlushLastMail(mail);
}

__aicore__ inline uint64_t SnapshotLastWindow(uint32_t row, bool trace) {
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    const uint32_t last=runtimeShape_.rounds-1;
    auto events=reinterpret_cast<__gm__ rotating::Event*>(ctx->events);
    auto event=events+last*runtimeShape_.serviceLanes+row;
    if(trace) event->waitBegin=GetSystemCycle();
    auto previous=reinterpret_cast<__gm__ volatile rotating::Event*>(events+(last-1)*runtimeShape_.serviceLanes+row);
    do { kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ rotating::Event*>(previous)),64); }
    while(!previous->complete);
    uint64_t begin=trace?GetSystemCycle():0;
    GlobalTensor<uint32_t> progress,rows;
    progress.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->progress)+row*128);
    rows.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->rows)+row*256);
    // Same two DMA reads and predecessor fence as the ordinary owner. Publish
    // each Q's two captured frontiers unchanged, even when a shard is late.
    DataCopy(incrementalOld_,progress,128);
    DataCopy(incrementalOld_[256],rows,16);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID5); WaitFlag<HardEvent::MTE2_S>(EVENT_ID5);
    const uint64_t snapshot=trace?GetSystemCycle():0;
    const bool empty=incrementalOld_.GetValue(256+9)==0;
    for(uint32_t q=0;q<4;++q) {
        auto mail=LastMail(row*4+q);
        mail->ready=uint64_t(incrementalOld_.GetValue(q*32))|(uint64_t(incrementalOld_.GetValue(q*32+16))<<32);
        mail->coveredEmpty=empty;
        if(trace) mail->snapshot=snapshot;
        SendLastWindowMail(mail,1);
    }
    return begin;
}

__aicore__ inline void RunLastWindowQuery(uint32_t query,uint32_t half,bool trace,uint64_t arrival) {
    auto mail=LastMail(query);
    WaitLastWindowMail(mail,1);
    const uint64_t workBegin=trace?GetSystemCycle():0;
    uint64_t ready=mail->ready;
    windowCoveredEmpty_=mail->coveredEmpty;
    GlobalTensor<uint64_t> meta;
    meta.SetGlobalBuffer(IncrementalMeta(query));
    // Transfer only this Q's old state; its second line now belongs to the
    // handshake and no sibling ever writes this Q's first line.
    DataCopy(incrementalMetadata_,meta,8);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID5); WaitFlag<HardEvent::MTE2_S>(EVENT_ID5);
    incrementalMetadataActive_=true;
    incrementalMetadataIndex_=0;
    incrementalMetadata_.SetValue(4,ready);
    rankReady_=ready;rankHalf_=half;
    rankedEpoch_=runtimeShape_.rounds-1;
    rankTrace_=trace;exportWindowCold_=true;
    incrementalCalls_=incrementalScanned_=incrementalCandidates_=0;
    incrementalHasNext_=false;
    const uint32_t cap=rotating::SequenceWindowEnd(runtimeShape_.history,runtimeShape_.resident,half,rankedEpoch_);
    const uint32_t end0=Min(uint32_t(ready),cap),end1=Min(uint32_t(ready>>32),cap);
    if(end0>=runtimeShape_.resident) AdvanceIncremental(query,end0,end1,half);
    JoinColdOutput();
    if(incrementalInputPending_) {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID6);incrementalInputPending_=false;
    }
    if(incrementalOutputPending_) {
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);incrementalOutputPending_=false;
    }
    SetFlag<HardEvent::S_MTE3>(EVENT_ID5); WaitFlag<HardEvent::S_MTE3>(EVENT_ID5);
    DataCopy(meta,incrementalMetadata_,8);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID5); WaitFlag<HardEvent::MTE3_S>(EVENT_ID5);
    incrementalMetadataActive_=false;
    exportWindowCold_=false;
    if(trace) {
        const uint64_t workDone=GetSystemCycle();
        auto ctx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
        // Four-window diagnostics use rank words [0..3]. The final cache
        // line of this Q's private 32-word trace survives terminal retirement.
        auto traceBase=ctx->incremental+runtimeShape_.batch*(512+4*2048*6+1024+(runtimeShape_.history+255)/256*32);
        auto samples=reinterpret_cast<__gm__ uint64_t*>(traceBase)+query*32+24;
        samples[0]=arrival;samples[1]=mail->snapshot;samples[2]=workBegin;samples[3]=workDone;
        samples[4]=ready;samples[5]=incrementalCalls_;samples[6]=incrementalScanned_;samples[7]=incrementalCandidates_;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(samples),64);
        mail->calls=incrementalCalls_;mail->scanned=incrementalScanned_;mail->candidates=incrementalCandidates_;
    }
    SendLastWindowMail(mail,2);
}

__aicore__ inline void CompleteLastWindow(uint32_t role,uint32_t half) {
    const uint32_t row=rotating::ServiceLane(runtimeShape_.batch,runtimeShape_.splitQ,
        runtimeShape_.serviceAics,blockId_/2,blockId_%2);
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    const bool trace=streamingWorkspace_ &&
        (reinterpret_cast<__gm__ pipeline_ab::Args*>(streamingWorkspace_)->prefetchCount & 0x80000000U);
    const uint64_t arrival=trace?GetSystemCycle():0;
    uint64_t begin=role==3?SnapshotLastWindow(row,trace):0;
    // Drain this lane's last score before reusing its TopK and progress UB.
    PipeBarrier<PIPE_ALL>();
    RunLastWindowQuery(row*4+role,half,trace,arrival);
    if(role!=3) return;
    uint64_t calls=0,scanned=0,candidates=0;
    for(uint32_t q=0;q<4;++q) {
        auto mail=LastMail(row*4+q);
        WaitLastWindowMail(mail,2);
        if(trace){calls+=mail->calls;scanned+=mail->scanned;candidates+=mail->candidates;}
    }
    if(trace) {
        auto event=reinterpret_cast<__gm__ rotating::Event*>(ctx->events)+(runtimeShape_.rounds-1)*runtimeShape_.serviceLanes+row;
        auto stamp=reinterpret_cast<__gm__ uint64_t*>(event->pad);
        stamp[2]=begin;stamp[3]=GetSystemCycle();stamp[4]=calls|(scanned<<16)|(candidates<<40);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(event),128);
    }
    PostWindow(runtimeShape_.rounds-1,half);
}
