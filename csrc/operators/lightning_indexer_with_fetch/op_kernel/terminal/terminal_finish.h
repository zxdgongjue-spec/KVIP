#pragma once
#include "terminal_events.h"
// Join is request-local. One warp gathers completion metadata once; other
// warps clear dead per-round state concurrently instead of reloading the same
// four GM flag lines from every warp.
template<bool LargeBatch>
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void FinishTerminalRow(
    __gm__ pipeline_ab::Args* a, __gm__ rotating::Context* ctx,
    __gm__ terminal_runtime::Binding* runtime, uint32_t b,uint32_t q,
    uint32_t owner,uint32_t total,uint32_t first,uint32_t count,
    __gm__ uint32_t* flags,uint32_t generation,uint32_t scanHistory,__ubuf__ uint32_t* control) {
    uint32_t t=threadIdx.x,history=ctx->shape.history,resident=ctx->shape.resident;
    if(!t)control[4]=total>2048?asc_ldcg(flags+9*32):0U;
    if(t<32){
        // Keep the request-local completion join; overlap only dead-state
        // retirement in the other warps. All warps rendezvous below.
        if(!t){asc_threadfence();asc_atomic_add(flags+6*32,1U);asc_threadfence();}
        uint32_t readyMask;
        do{bool ready=t!=0 || asc_ldcg(flags+6*32)==generation*4;
           readyMask=asc_ballot(ready);}while(readyMask!=UINT32_MAX);
        uint32_t error=t<4?asc_ldcg(flags+(28+t)*32):0;
        uint32_t reads=0,pending=0,completed=0;
        if(!q){
            // Four contiguous words per completion record: load all records
            // with one instruction before the warp distributes their fields.
            uint32_t word=0;
            if(t<16){
                auto post=reinterpret_cast<__gm__ rotating::FinalPost*>(ctx->owners+ctx->shape.finalPostOffset)+b*4+t/4;
                word=asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(post)+t%4);
            }
            uint32_t lane=(t%4)*4;
            uint32_t n=asc_shfl(word,lane),p=asc_shfl(word,lane+1),e=asc_shfl(word,lane+2);
            if(t<4){reads=n;pending=p;completed=n!=0 && p==0 && e==0;}
        }
        for(uint32_t d=2;d;d>>=1){
            error|=asc_shfl_down(error,d);reads+=asc_shfl_down(reads,d);
            pending+=asc_shfl_down(pending,d);completed+=asc_shfl_down(completed,d);
        }
        if(!t){control[0]=error;control[1]=reads;control[2]=pending;control[3]=completed;}
    }else{
        constexpr uint32_t clearThreads=480;
        // Every admitted source is below its Q's valid end. Clear this
        // round's complete written prefix before the next replay can grow.
        uint32_t words=(scanHistory+255)/256*8;
        auto covered=reinterpret_cast<__gm__ uint32_t*>(
            kvip_prefetch_rank::CoveredAddress(ctx->incremental,ctx->shape.batch,history,b));
        for(uint32_t i=q*clearThreads+t-32;i<words;i+=4*clearThreads)asc_stcg(covered+i,0U);
        if(!LargeBatch && a->completedEvents){
            uint32_t eventWords=ctx->shape.rounds*ctx->shape.serviceLanes*(sizeof(rotating::Event)/4);
            auto live=reinterpret_cast<__gm__ uint32_t*>(ctx->events);
            auto saved=reinterpret_cast<__gm__ uint32_t*>(a->completedEvents);
            for(uint32_t i=(b*4+q)*clearThreads+t-32;i<eventWords;i+=ctx->shape.batch*4*clearThreads){
                asc_stcg(saved+i,asc_ldcg(live+i));asc_stcg(live+i,0U);
            }
        }
    }
    asc_syncthreads();
    if constexpr (LargeBatch) {
        SaveRequestEvents(a,ctx,b,q,t);
    }
    if(!q && !t){asc_stcg(flags+5*32,0U);asc_stcg(flags+7*32,0U);}

    uint32_t extendedCount=control[4];
    if(extendedCount>2048){
        auto extended=reinterpret_cast<__gm__ uint32_t*>(runtime->reserved[0])+uint64_t(b)*(rotating::kChangeCapacity+4*((resident+31)/32));
        auto inverse=reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+uint64_t(b)*resident;
        for(uint32_t i=2048+q*512+t;i<extendedCount;i+=2048)asc_stcg(inverse+asc_ldcg(extended+i),int32_t(-1));
    }
    // Snapshot the count on all workers before resetting its shared word.
    // The monotonic per-Q generation is published only after this retirement.
    if(!q && !t){asc_stcg(flags+8*32,0U);asc_stcg(flags+9*32,0U);asc_stcg(flags+10*32,0U);}
    uint32_t error=control[0];
    // Publish through the same SIMT store path as the authoritative summary.
    // Detailed per-Q records are copied only on a failed request.
    auto queryError=reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)+(b*4+q)*32;
    if(error && t<32){
        uint32_t value=0;
        bool failed=asc_ldcg(flags+(28+q)*32)!=0;
        if(t<12)value=failed?asc_ldcg(flags+(28+q)*32+t):0;
        else if(t<=16){
            bool indexFailure=failed && asc_ldcg(flags+(28+q)*32+3)!=0 && asc_ldcg(flags+(28+q)*32+8)!=0;
            value=indexFailure?asc_ldcg(flags+(12+q)*32+t):UINT32_MAX;
        }
        asc_stcg(queryError+t,value);
    }
    else if(!t)asc_stcg(queryError,0U);

    auto dst=reinterpret_cast<__gm__ uint32_t*>(ctx->changeSlots)+uint64_t(b)*rotating::kChangeCapacity;
    auto valid=reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+uint64_t(owner)*resident;
    for(uint32_t i=t;i<count;i+=512){
        uint32_t slot=asc_ldcg(dst+first+i);
        if(slot<resident)asc_stcg(valid+slot,error?0U:1U);
    }
    if(!q){
        auto row=reinterpret_cast<__gm__ rotating::Row*>(ctx->rows)+b;
        // Independent warp: status summary below must not wait for these
        // pointer-chasing row/update publications in lane zero of warp zero.
        if(t==32){
            uint32_t reads=control[1],pending=control[2],completed=control[3];
            auto info=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*32;
            asc_stcg(info,error);asc_stcg(info+1,reads);asc_stcg(info+3,pending?1U:0U);asc_stcg(info+4,completed);
            auto update=reinterpret_cast<__gm__ uint64_t*>(a->updateBindings);
            asc_stcg(reinterpret_cast<__gm__ uint32_t*>(asc_ldcg(update+6))+b,error);
            asc_stcg(reinterpret_cast<__gm__ uint32_t*>(ctx->changeCount)+b,total);
        }
        if(t<16){
            auto status=reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(ctx->finalStatus)+2*b+t/8;
            uint64_t value=0;
            switch(t%8){
                case 0:value=KVAF_AIV_BATCH_GET_STATUS_MAGIC;break;
                case 1:if(t<8)value=control[1];break;
                case 2:if(t<8)value=error;break;
                case 3:if(t<8)value=control[3];break;
                case 6:value=UINT64_MAX;break;
            }
            if(t!=4 && t!=5)asc_stcg(reinterpret_cast<__gm__ uint64_t*>(status)+t%8,value);
        }
        if(t==64 || t==65){
            auto status=reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(ctx->finalStatus)+2*b;
            auto args=reinterpret_cast<__gm__ KvafExecuteBatchGetKernelArgs*>(runtime->queueArgs)+b*4+3;
            auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(asc_ldcg(&args->queue_state));
            uint64_t value=t==64?asc_ldcg(&queue->sq_producer_index):asc_ldcg(&queue->cq_consumer_index);
            asc_stcg(reinterpret_cast<__gm__ uint64_t*>(status)+t-60,value);
        }
        // No live-row writers remain after the request-local CQ/Index join.
        // Publish the two final counters directly into the saved row instead
        // of writing, synchronizing, and reading them back from the live row.
        if(a->completedRows && t>=128){
            auto live=reinterpret_cast<__gm__ uint32_t*>(row);
            auto saved=reinterpret_cast<__gm__ uint32_t*>(a->completedRows)+b*(sizeof(rotating::Row)/4);
            for(uint32_t i=t-128;i<sizeof(rotating::Row)/4;i+=384){
                uint32_t value;
                if(i==__builtin_offsetof(rotating::Row,finalRequests)/4)value=control[1];
                else if(i==__builtin_offsetof(rotating::Row,finalPending)/4)value=control[2];
                else value=asc_ldcg(live+i);
                asc_stcg(saved+i,value);asc_stcg(live+i,0U);
            }
        }
    }
    if(!t)asc_stcg(flags+q*32+1,generation);
    asc_threadfence();
}
