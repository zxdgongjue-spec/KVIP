#pragma once
#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "context.h"
#include "pipeline_types.h"

#include <stdint.h>
#include "prefetch.h"
using namespace AscendC;
using namespace rotating;
__aicore__ inline __gm__ Context* GetContext(GM_ADDR bindings) {
 return reinterpret_cast<__gm__ Context*>(reinterpret_cast<__gm__ uint64_t*>(bindings)[14]);
}
__aicore__ inline void FillRoundWords(uint64_t address,uint32_t words,
    LocalTensor<uint32_t> local) {
    GlobalTensor<uint32_t> target;target.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(address));
    for(uint32_t offset=0;offset<words;offset+=4096) {
        uint32_t count=words-offset;if(count>4096)count=4096;
        DataCopy(target[offset],local,count);
    }
    // Caller joins the whole fill group before reusing the shared UB value.
}
__aicore__ inline void InitializeRound(uint32_t rt_batch,uint32_t rt_history,uint32_t rt_rounds,uint32_t rt_serviceLanes,uint32_t rt_finalWorkers,uint32_t rt_finalPostOffset,__gm__ Context* ctx,uint32_t b,__gm__ pipeline_ab::Args* streaming,bool resetLiState) {
    TPipe pipe;TBuf<TPosition::VECCALC> storage;pipe.InitBuffer(storage,4096*4);
    auto local=storage.Get<uint32_t>();
    Duplicate(local,uint32_t(0),4096);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    FillRoundWords(ctx->incremental+b*512,128,local);
    if(resetLiState) {
        FillRoundWords(kvip_prefetch_rank::CoveredAddress(ctx->incremental,rt_batch,rt_history,b),(rt_history+255)/256*8,local);
        FillRoundWords(ctx->progress+b*4*sizeof(Progress),4*sizeof(Progress)/4,local);
    }
    FillRoundWords(ctx->owners+b*128,32,local);
    FillRoundWords(ctx->owners+rt_batch*128+b*rt_finalWorkers*kFinalStateWords*4,rt_finalWorkers*kFinalStateWords,local);
    FillRoundWords(ctx->owners+rt_finalPostOffset+b*rt_finalWorkers*sizeof(FinalPost),rt_finalWorkers*sizeof(FinalPost)/4,local);
    FillRoundWords(ctx->rowStatus+b*kRowStatusWords*4,kRowStatusWords,local);
    FillRoundWords(ctx->finalStatus+b*2*sizeof(KvafBatchGetCoreStatus),2*sizeof(KvafBatchGetCoreStatus)/4,local);
    // Fetch status and publication counts remain observable through the end of the round.
    if(streaming) {
        uint32_t fw=(streaming->prefetchCount>>16)&0xffU,ds=rt_batch*2048;
        uint32_t meta=(3*rt_batch+31)&~31U;
        for(uint32_t g=b;g<4*fw;g+=rt_batch)
            FillRoundWords(streaming->packedFetch+uint64_t(4*fw*(3*ds+meta)+g*(6*ds+64)+6*ds)*4,64,local);
        FillRoundWords(streaming->flags+uint64_t(b)*pipeline_ab::kFlagCount*sizeof(pipeline_ab::BatchFlag),
                       pipeline_ab::kFlagCount*sizeof(pipeline_ab::BatchFlag)/4,local);
        FillRoundWords(streaming->workerStatus+uint64_t(b)*128,32,local);
        FillRoundWords(streaming->queryErrors+uint64_t(b)*4*128,4*32,local);
    }
    // Both hot and cold predicates are overwritten before progress is
    // published. Original per-query snapshot clipping excludes unready bits.
    for(uint32_t e=b;e<(rt_rounds?rt_rounds:1)*rt_serviceLanes;e+=rt_batch)
        FillRoundWords(ctx->events+e*sizeof(Event),sizeof(Event)/4,local);
    if(!b) {
        *reinterpret_cast<__gm__ uint32_t*>(ctx->count)=0;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->count),4);
    }
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    pipe.Destroy();
}
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void ResetTrackedMapping(__gm__ uint32_t* map,__gm__ MappingJournal* journal,__gm__ uint32_t* count,
                               uint32_t history,uint32_t oldBegin,uint32_t oldEnd,uint32_t dirtyCount) {
    unsigned tid=threadIdx.x;
    for(unsigned i=tid;i<dirtyCount;i+=256) {
        unsigned token=asc_ldcg(journal->tokens+i);
        if(token<history)asc_stcg(map+token,UINT32_MAX);
    }
    // A standalone full reset can precede a streaming call on the same scratch.
    for(unsigned i=tid;oldBegin<history && i<history-oldBegin && oldBegin+i<oldEnd;i+=256)
        asc_stcg(map+oldBegin+i,UINT32_MAX);
    if(!tid){
        asc_stcg(&journal->count,0U);
        asc_stcg(&journal->tailBegin,0U);
        asc_stcg(&journal->tailEnd,0U);
        if(count)asc_stcg(count,kLruSlots);
    }
    asc_threadfence();
}
__aicore__ inline void PrepareRequestPart(GM_ADDR bindings,uint32_t fullReset,GM_ADDR workspace,
    uint32_t rt_batch,uint32_t rt_history,uint32_t rt_resident,uint32_t rt_rounds,
    uint32_t rt_serviceLanes,uint32_t rt_finalWorkers,uint32_t rt_finalPostOffset,uint32_t rt_tokenBytes,uint32_t block) {
    auto ctx=GetContext(bindings);
    if ASCEND_IS_AIV {
        uint32_t b=block/4, part=(block+3)%4;
        if(part>=4) return;
        if(ctx->mappingJournal && !fullReset && ctx->prefetchLimit<=kLruSlots) {
            if(part==3 && ctx->changes) {
                TPipe pipe;
                TBuf<TPosition::VECCALC> storage;
                pipe.InitBuffer(storage,kLruSlots*sizeof(uint32_t));
                auto local=storage.Get<uint32_t>();
                Duplicate(local,UINT32_MAX,kLruSlots);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                GlobalTensor<uint32_t> changes;
                changes.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->changes)+b*kChangeCapacity);
                DataCopy(changes,local,kLruSlots);
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
                pipe.Destroy();
            }
            if(part==2) {
                auto journal=reinterpret_cast<__gm__ MappingJournal*>(ctx->mappingJournal)+b;
                auto count=reinterpret_cast<__gm__ uint32_t*>(ctx->changeCount);
                kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(&journal->tailBegin),128);
                Simt::VF_CALL<ResetTrackedMapping>(Simt::Dim3{256,1,1},
                    reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+uint64_t(b)*rt_history,journal,count?count+b:nullptr,
                    rt_history,journal->tailBegin,journal->tailEnd,journal->count);
            }
        } else {
            // Each worker owns an aligned range, including its ABC mappings.
            // No cache line is concurrently cleared and filled by another core.
            uint32_t chunk=((rt_history+127)/128)*32;
            uint32_t begin=part*chunk;
            uint32_t end=begin+chunk<rt_history ? begin+chunk : rt_history;
            if(begin<end || ctx->changes || (part==3 && ctx->changeSlots)) {
                TPipe mapPipe; TBuf<TPosition::VECCALC> mapStorage;
                mapPipe.InitBuffer(mapStorage,4096*4);
                auto local=mapStorage.Get<uint32_t>();
                Duplicate(local,UINT32_MAX,4096);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                if(begin<end) FillRoundWords(ctx->mapping+(uint64_t(b)*rt_history+begin)*4,end-begin,local);
                // All four workers clear disjoint aligned changes ranges. This
                // also runs when the corresponding mapping range is empty.
                if(ctx->changes) {
                    constexpr uint32_t changeChunk=kChangeCapacity/4;
                    static_assert(kChangeCapacity%4==0 && changeChunk%8==0,"unaligned change shards");
                    FillRoundWords(ctx->changes+(uint64_t(b)*kChangeCapacity+part*changeChunk)*4,changeChunk,local);
                }
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
                // The next kernel observes the completed LRU copy at the launch
                // boundary; it does not depend on the owner's row initialization.
                if(fullReset && part==3 && ctx->changeSlots) {
                    GlobalTensor<uint32_t> source,target;
                    source.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+b*kLruSlots);
                    target.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->changeSlots)+b*kChangeCapacity);
                    DataCopy(local,source,kLruSlots);
                    SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                    DataCopy(target,local,kLruSlots);
                    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
                }
                mapPipe.Destroy();
                if(begin<end && (fullReset || begin==0))
                    Simt::VF_CALL<InitializeTailMappings>(Simt::Dim3{256,1,1},rt_history,rt_resident,ctx,b,begin,end,fullReset!=0);
            }
        }
        // A separate shard clears the per-round state while the owner
        // initializes the row. The next kernel waits for all init blocks.
        if(part==1) {
            auto initBindings=reinterpret_cast<__gm__ uint64_t*>(bindings);
            auto streaming=reinterpret_cast<__gm__ pipeline_ab::Args*>(workspace ? reinterpret_cast<uint64_t>(workspace) : initBindings[15]);
            InitializeRound(rt_batch,rt_history,rt_rounds,rt_serviceLanes,rt_finalWorkers,rt_finalPostOffset,ctx,b,streaming,fullReset || !streaming);
        }
        if(part) return;
        TPipe rowPipe; TBuf<TPosition::VECCALC> rowStorage;
        rowPipe.InitBuffer(rowStorage,sizeof(Row));
        auto rowLocal=rowStorage.Get<uint32_t>();
        Duplicate(rowLocal,uint32_t(0),sizeof(Row)/4);
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        // Assemble the row in UB, including the zeroed counters and padding.
        // Publish the completed row with a single DMA at the kernel boundary.
        auto row=reinterpret_cast<__ubuf__ Row*>(rowLocal.GetPhyAddr());
        const auto& args=reinterpret_cast<__gm__ KvafExecuteBatchGetKernelArgs*>(ctx->args)[b*2];
        auto entity=reinterpret_cast<__gm__ KvafBatchGetDirectUrmaEntity*>(args.channel_entity);
        auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args.queue_state);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
        row->entity=args.channel_entity; row->queue=args.queue_state;
        const uint64_t prefix=4*2048*6+1024+4*uint64_t(rt_history)*2;
        const uint64_t tail=uint64_t(rt_history)*8+8192*64+512;
        auto a=reinterpret_cast<__gm__ uint64_t*>(bindings);
        uint32_t owner=ctx->rowOwners ? reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b] : b;
        uint64_t outputStride=ctx->outputStride ? ctx->outputStride : rt_resident;
        row->addresses=ctx->tokenAddresses ? ctx->tokenAddresses+uint64_t(owner)*ctx->addressStride*8 : a[13]+rt_batch*prefix+b*tail;
        if(fullReset)row->output=args.output+uint64_t(owner)*outputStride*rt_tokenBytes;
        if(fullReset)row->initialHead=queue->sq_producer_index;
        row->limit=kSlots;
        row->prefetchLimit=ctx->prefetchLimit;
        row->queueLimit=entity->sq_depth>1 ? entity->sq_depth-1 : 0;
        row->error=queue->last_error;
        if(entity->sq_depth<2 || (entity->sq_depth&(entity->sq_depth-1)) || entity->cq_depth<2 ||
           !entity->sq_va || !entity->sq_db_va || ctx->prefetchLimit>kSlots ||
           ctx->windowLimit==0 || ctx->windowLimit>480)
            row->error=kvaf_device::kInvalidArgs;
#if !V9_SIMULATION
        if(queue->initialized && queue->magic!=kvaf_device::kQueueMagic)
            row->error=kvaf_device::kInvalidArgs;
        if(!queue->initialized) {
            queue->magic=kvaf_device::kQueueMagic; queue->initialized=1;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
        }
        if(entity->sq_depth>(1U<<30) || row->initialHead>UINT64_MAX-ctx->prefetchLimit ||
           args.token_bytes!=rt_tokenBytes || args.tokens_per_block!=128 ||
           args.block_count!=1 || args.jettys_per_core!=1)
            row->error=kvaf_device::kInvalidArgs;
        if(entity->magic!=kvaf_device::kEntityMagic || entity->version!=kvaf_device::kEntityVersion ||
           entity->wqe_size!=64 || !entity->cq_va || !entity->cq_db_va ||
           entity->cqe_size<kvaf_device::kCqeProbeBytes || !entity->local_token_id ||
           entity->local_hbm_address!=args.output ||
           entity->local_hbm_size<(uint64_t(owner)+1)*outputStride*rt_tokenBytes ||
           !entity->remote_host_ddr_address || entity->remote_host_ddr_size<rt_tokenBytes)
            row->error=kvaf_device::kInvalidEntity;

#endif
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        FillRoundWords(ctx->rows+b*sizeof(Row),sizeof(Row)/4,rowLocal);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        rowPipe.Destroy();
    }
}
#endif
