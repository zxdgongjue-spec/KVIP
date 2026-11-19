// SPDX-License-Identifier: Apache-2.0
#pragma once

// Four AIVs own one request, one Q and one private final queue each. Init
// already clears these four cache-line-separated records at each HIGH step.
// One writer per record. Later phases never overwrite an earlier result.
struct RowRendezvous {
    enum Phase : uint32_t { Validated=1, ReadOriginals=2, Done=3 };
    __gm__ BatchFlag* lines;
    uint32_t lane;
    __aicore__ inline void Publish(Phase phase,uint32_t error) {
        auto f=lines+lane;
        f->reserved[phase]=error;
        f->reserved[0]=phase;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(f),sizeof(*f));
    }
    __aicore__ inline uint32_t Await(uint32_t peer,Phase phase) {
        auto f=lines+peer;
        uint64_t start=GetSystemCycle();
        for (;;) {
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(f),sizeof(*f));
            if(f->reserved[0]>=phase) return f->reserved[phase];
            if(GetSystemCycle()-start>5000000000ULL) return 8;
        }
    }
    __aicore__ inline uint32_t AwaitPeers(Phase phase) {
        uint32_t error=0;
        // The caller already holds its own result. Poll the three other
        // writers, starting at different peers on each lane.
        for(uint32_t distance=1;distance<4;++distance)
            error|=Await((lane+distance)&3U,phase);
        return error;
    }
    __aicore__ inline uint64_t Gathered() {
        Publish(ReadOriginals,0);
        return AwaitPeers(ReadOriginals);
    }
};

// OutputWorker has already packed and flushed these immutable Q-local arrays.
// Keep the original device bounds checks without making a second packed copy.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void CheckFinalRange(__gm__ Context* ctx,uint32_t begin,uint32_t count,
        uint64_t remote,uint64_t remoteBytes,uint64_t outputBytes,uint32_t outputRows,
        uint32_t bytes,__ubuf__ uint32_t* error) {
    if(!threadIdx.x) error[0]=0;
    asc_syncthreads();
    for(uint32_t i=threadIdx.x;i<count;i+=256) {
        uint32_t pos=begin+i;
        auto address=asc_ldcg(reinterpret_cast<__gm__ uint64_t*>(ctx->fallbackAddresses)+pos);
        auto dst=asc_ldcg(reinterpret_cast<__gm__ int32_t*>(ctx->destinations)+pos);
        if(asc_ldcg(reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks)+pos)!=int32_t(pos) ||
           asc_ldcg(reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets)+pos)!=0 ||
           !asc_ldcg(reinterpret_cast<__gm__ uint8_t*>(ctx->fallbackValid)+pos) ||
           remoteBytes<bytes || address<remote || address-remote>remoteBytes-bytes ||
           dst<0 || uint32_t(dst)>=outputRows || (uint64_t(dst)+1)*bytes>outputBytes)
            asc_atomic_or(error,1U);
    }
    asc_syncthreads();
}

__aicore__ __attribute__((noinline)) inline void RunRow(__gm__ Args* a,uint32_t block) {
    auto ctx=reinterpret_cast<__gm__ Context*>(a->context);
    auto shape=LoadRuntimeShape(ctx);
    const uint32_t batch=shape.batch,b=block%batch,q=block/batch;
    auto flags=reinterpret_cast<__gm__ BatchFlag*>(a->flags)+b*kFlagCount;
    RowRendezvous join{flags+8,q};
    auto me=join.lines+q;
    auto records=reinterpret_cast<__gm__ post_index_tail::Record*>(a->postIndex);
    auto r=records+block;
    auto update=reinterpret_cast<__gm__ uint64_t*>(a->updateBindings);
    auto info=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*kRowStatusWords;
    auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
    auto posts=FinalPostAt(shape.finalPostOffset,ctx,4*b);
    auto ticks=reinterpret_cast<__gm__ uint64_t*>(a->times)+uint64_t(batch)*256+block*32;
    const bool profile=(a->prefetchCount & 0x80000000U)!=0;
    if(profile){
        r->start=GetSystemCycle();
        r->planDone=0; r->postBegin=0; r->postDone=0; ticks[4]=0; ticks[8]=0;
    }
    uint32_t useful=0,error=0;
    // Keep Q0's private final FIFO free to issue as soon as its descriptors
    // arrive. The resolver owns Q3; its own descriptors are produced last.
    if(q==3) useful=Producer<false,true>(a,b);
    if(q==1) error|=OutputWorker<false,OutputRole::Descriptors>(a,b,profile?ticks+24:nullptr);
    if(q==2) {
        auto edges=reinterpret_cast<__gm__ Edge*>(a->highIndex+uint64_t(b)*kRowBytes);
        auto spill=reinterpret_cast<__gm__ Tuple*>(edges+kCapacity);
        PrivatePlan plan;
        Plan(a,b,edges,&spill[0].padding,false,plan);
        if(profile) r->planDone=GetSystemCycle();
        error|=plan.error;
        me->reserved[8]=plan.pairs; me->reserved[9]=plan.singles;
        me->reserved[10]=plan.invalidations;
    }
    error|=AwaitPublication(flags+1+q);
    if(profile){ticks[20]=GetSystemCycle();ticks[22]=0;}
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(posts+q),sizeof(*posts));
    const uint32_t count=posts[q].count,safe=posts[q].nonconflicting,base=b*kSlots+q*kTop;
    error|=posts[q].error;
    if(count>kTop || safe>count || posts[q].descriptor!=base) error|=kvaf_device::kInvalidArgs;

    auto args=GetArgs(ctx,FinalQueue(batch,4,b,q));
    args.core_status=a->highIndex+uint64_t(batch)*kRowBytes+uint64_t(block)*update[15];
    args.block_count=1; args.jettys_per_core=1;
    args.indices=ctx->sourceBlocks+uint64_t(base)*4;
    args.offsets=ctx->sourceOffsets+uint64_t(base)*4;
    args.output_indices=ctx->destinations+uint64_t(base)*4;
    args.address_table=ctx->fallbackAddresses; args.valid_table=ctx->fallbackValid;
    args.table_capacity=batch*kSlots; args.request_capacity=count; args.tokens_per_block=1;
    bool blocking=false,split=false;
    if(!error) error=post_index_transport::Preflight(args,0,count,blocking);
    if(!error && count) {
        auto entity=reinterpret_cast<__gm__ const KvafBatchGetDirectUrmaEntity*>(args.channel_entity);
        split=safe && safe<count && !blocking && entity->cq_depth>2;
        TPipe pipe; TBuf<TPosition::VECCALC> buf; pipe.InitBuffer(buf,32);
        auto local=buf.Get<uint32_t>();
        Simt::VF_CALL<CheckFinalRange>(Simt::Dim3{256,1,1},ctx,base,count,entity->remote_host_ddr_address,
            entity->remote_host_ddr_size,entity->local_hbm_size,args.output_rows,args.token_bytes,
            reinterpret_cast<__ubuf__ uint32_t*>(local.GetPhyAddr()));
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
        error|=local.GetValue(0); pipe.Destroy();
    }
    if(q==3) {
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(info),128);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row),64);
        error|=row->error | info[0];
        me->reserved[5]=useful;
        me->reserved[6]=info[1];
    }
    // pending is immutable from kernel entry through the row's Index join.
    // Only a range that might overwrite a speculative KV depends on its CQ.
    bool conflict=false;
    if(safe<count){
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(&row->pending),sizeof(row->pending));
        conflict=row->pending!=0;
    }
    me->count=count;
    if(profile){r->packDone=GetSystemCycle();ticks[21]=r->packDone;}
    post_index_transport::State first{},second{};
    uint32_t early=blocking ? 0 : (!conflict ? count : (split ? safe : 0));
    if(!error && early){
        if(profile){r->postBegin=GetSystemCycle();ticks[4]=r->postBegin;ticks[8]=1;}
        error|=post_index_transport::Post(args,0,early,first);
        if(profile) r->postDone=GetSystemCycle();
    }
    // WQE scratch is released on return from Post. TND reads selected
    // logical IDs and resolved locations, independently of final KV writes.
    if(!q) {
        error|=OutputWorker<false,OutputRole::Tnd>(a,b);
        if(profile) ticks[22]=GetSystemCycle();
    }
    // A rank can now submit before a peer reports its plan/TND/validation
    // error. Such an issued prefix must drain. Real Index writes require the
    // row-wide success below; finalization invalidates residency on failure.
    join.Publish(RowRendezvous::Validated,error);
    error|=join.AwaitPeers(RowRendezvous::Validated);
    uint32_t total=0;
    for(uint32_t p=0;p<4;++p) total+=join.lines[p].count;
    if(total!=join.lines[3].reserved[6]) error|=kvaf_device::kInvalidArgs;
    if(profile) ticks[23]=GetSystemCycle();
    auto plan=join.lines+2;
    uint32_t owner=ctx->rowOwners?reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b]:b;
    if(profile) ticks[3]=GetSystemCycle();
    if(!error) error|=ApplyRowIndex(a,ctx,b,owner,q,4,plan->reserved[8],plan->reserved[9],
                                   plan->reserved[10],update[7],ticks,profile,join);
    // A failed lane can reach Done without gathering originals. Mark the
    // skipped phase failed as well so its peers never scatter stale payload.
    if(error) join.Publish(RowRendezvous::ReadOriginals,error);
    if(profile){ r->indexDone=GetSystemCycle(); ticks[2]=0x52514936U; ticks[28]=0; ticks[29]=0; }
    // Only q0 consumes the prefetch queue. Only ranges that might
    // overwrite an in-flight prefetch destination depend on that completion.
    if(!q){
        if(profile) ticks[28]=GetSystemCycle();
        CompletePrefetch<true>(ctx,b,true);
        if(profile) ticks[29]=GetSystemCycle();
        error|=row->error;
        flags[7].error=error; flags[7].ready=1;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flags+7),sizeof(*flags));
    }
    if(early<count) {
        if(conflict) error|=AwaitPublication(flags+7);
        if(!error){
            if(profile){
                uint64_t begin=GetSystemCycle(); ticks[4]=begin; ticks[8]=early ? 2 : 1;
                if(!early) r->postBegin=begin;
            }
            error|=blocking ? post_index_transport::GetBlocking(args,early,count-early,second)
                            : post_index_transport::Post(args,early,count-early,second);
            if(profile) r->postDone=GetSystemCycle();
        }
    }
    // Both checkpoints, if issued, belong to this lane's FIFO. An accepted
    // prefix must be retired even on failure of the row or of the suffix.
    if(profile) ticks[30]=GetSystemCycle();
    error|=post_index_transport::RetirePair(first,second,profile ? ticks+31 : nullptr);
    if(profile) r->cqDone=GetSystemCycle();
    r->error=error; r->globalError=error; r->count=count; r->fallback=blocking?1:0;
    r->completed=first.completed+second.completed; r->active=2;
    post_index_tail::Flush(r);
    // Completed counts travel on the same cache line as this lane's Done flag.
    me->reserved[4]=first.completed+second.completed;
    join.Publish(RowRendezvous::Done,error);
    if(!q) {
        error|=join.AwaitPeers(RowRendezvous::Done);
        uint32_t completed=0;
        for(uint32_t p=0;p<4;++p) completed+=join.lines[p].reserved[4];
        if(completed!=total) error|=kvaf_device::kInvalidArgs;
        FinalizeStreaming<false,true,true>(a,b,join.lines[3].reserved[5],total,completed,error);
    }
}
