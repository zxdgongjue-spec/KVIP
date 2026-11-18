// SPDX-License-Identifier: Apache-2.0
#pragma once

// Four AIVs own one request, one Q and one private final queue each. Init
// already clears these four cache-line-separated records at each HIGH step.
// One writer per record. Later phases never overwrite an earlier result.
struct RowRendezvous {
    enum Phase : uint32_t { Validated=1, Posted=2, ReadOriginals=3, Done=4 };
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
    __aicore__ inline uint32_t All(Phase phase) {
        uint32_t error=0;
        for(uint32_t p=0;p<4;++p) error|=Await(p,phase);
        return error;
    }
    __aicore__ inline uint64_t Gathered() {
        Publish(ReadOriginals,0);
        return All(ReadOriginals);
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
    uint32_t useful=0,error=0;
    if(q==0) useful=Producer(a,b);
    if(q==1) OutputWorker<false>(a,b);
    if(q==2) {
        auto edges=reinterpret_cast<__gm__ Edge*>(a->highIndex+uint64_t(b)*kRowBytes);
        auto spill=reinterpret_cast<__gm__ Tuple*>(edges+kCapacity);
        PrivatePlan plan;
        Plan(a,b,edges,&spill[0].padding,false,plan);
        error|=plan.error;
        me->reserved[8]=plan.pairs; me->reserved[9]=plan.singles;
        me->reserved[10]=plan.invalidations;
    }
    error|=AwaitPublication(flags+1+q);
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
    if(!q) {
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(info),128);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row),64);
        error|=row->error | info[0];
        me->reserved[6]=info[1];
    }
    me->count=count;
    me->reserved[5]=(safe<count ? 1U:0U) | (blocking ? 2U:0U);
    join.Publish(RowRendezvous::Validated,error);
    error|=join.All(RowRendezvous::Validated);
    uint32_t hazards=0,total=0;
    for(uint32_t p=0;p<4;++p){ hazards|=join.lines[p].reserved[5]; total+=join.lines[p].count; }
    // Producer and OutputWorker have both returned before their Validated signal.
    if(total!=join.lines[0].reserved[6]) error|=kvaf_device::kInvalidArgs;
    post_index_transport::State first{},second{};
    if(!error && count && (safe==count || split)) {
        error|=blocking ? post_index_transport::GetBlocking(args,0,count,first)
                        : post_index_transport::Post(args,0,safe,first);
    }
    if(hazards & 1U) {
        if(!q) {
            CompletePrefetch(ctx,b,true);
            flags[7].error=row->error; flags[7].ready=1;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flags+7),sizeof(*flags));
        }
        if(safe<count) {
            error|=AwaitPublication(flags+7);
            if(!error) {
                uint32_t begin=split?safe:0;
                error|=blocking ? post_index_transport::GetBlocking(args,begin,count-begin,second)
                                : post_index_transport::Post(args,begin,count-begin,second);
            }
        }
        if(!q) error|=row->error;
    }
    // A nonblocking Post on a private preflighted queue has no synchronous
    // rejection. Only conflicts and blocking fallback can still reject a peer.
    if(hazards) {
        join.Publish(RowRendezvous::Posted,error);
        error|=join.All(RowRendezvous::Posted);
    }
    auto plan=join.lines+2;
    uint32_t owner=ctx->rowOwners?reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b]:b;
    if(!error) error|=ApplyRowIndex(a,ctx,b,owner,q,4,plan->reserved[8],plan->reserved[9],
                                   plan->reserved[10],update[7],ticks,profile,join);
    // Both checkpoints, if issued, are on this lane's exclusive FIFO.
    // Every accepted Post is drained even if another lane rejected the row.
    error|=post_index_transport::Wait(first);
    error|=post_index_transport::Wait(second);
    if(!q && !(hazards&1U)){ CompletePrefetch(ctx,b,true); error|=row->error; }
    r->error=error; r->globalError=error; r->count=count; r->fallback=blocking?1:0;
    r->completed=first.completed+second.completed; r->active=2;
    post_index_tail::Flush(r);
    join.Publish(RowRendezvous::Done,error);
    if(!q) {
        error|=join.All(RowRendezvous::Done);
        uint32_t completed=0;
        for(uint32_t p=0;p<4;++p) {
            auto record=records+b+p*batch;
            post_index_tail::Flush(record);
            completed+=record->completed;
        }
        if(completed!=total) error|=kvaf_device::kInvalidArgs;
        for(uint32_t p=0;p<4;++p){
            posts[p].pending=p==0?completed:0; posts[p].deferred=0; posts[p].error=error;
        }
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(posts),4*sizeof(*posts));
        Simt::VF_CALL<PublishStreamingUpdateStatus>(Simt::Dim3{32,1,1},reinterpret_cast<__gm__ uint32_t*>(update[6]),b,error?1U:0U);
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
        if(!error && (a->prefetchCount & 0x10000000U)){
            Simt::VF_CALL<PublishStreamingResidency>(Simt::Dim3{256,1,1},ctx,b,shape.resident);
            SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
        }
        FinalizeStreaming<false,true>(a,b,useful);
    }
}
