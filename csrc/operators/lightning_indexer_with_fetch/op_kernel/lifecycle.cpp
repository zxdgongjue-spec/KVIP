#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "context.h"
#include "pipeline_types.h"
#include "round_init.h"
#include "auto_route.h"
#if KVIP_AUTO_ROUTE_SIMT
#include "auto_route_simt.h"
#endif

#include <stdint.h>
#include "prefetch.h"
#if KVIP_LOW_MISS_POST_INDEX && KVIP_AUTO_ROUTE_ROW_PUBLISH
#include "auto_route_staged.h"
#endif
using namespace AscendC;
using namespace rotating;
__aicore__ inline __gm__ Context* GetContext(GM_ADDR bindings) {
 return reinterpret_cast<__gm__ Context*>(reinterpret_cast<__gm__ uint64_t*>(bindings)[14]);
}
__aicore__ inline KvafExecuteBatchGetKernelArgs GetArgs(__gm__ Context* ctx,uint32_t owner) {
 KvafExecuteBatchGetKernelArgs args{};
 static_assert(sizeof(args)==14*sizeof(uint64_t),"KVAF argument ABI changed");
 auto src=reinterpret_cast<__gm__ uint64_t*>(ctx->args)+owner*14;
 auto dst=reinterpret_cast<uint64_t*>(&args);
 for(uint32_t i=0;i<14;++i)dst[i]=src[i];
 return args;
}
#endif
#if defined(__CCE_AICORE__)
#if KVIP_LOW_MISS_POST_INDEX
__simt_vf__ __aicore__ LAUNCH_BOUND(32)
inline void InitializeLowChangeCount(__gm__ uint32_t* counts, uint32_t batch) {
    if (!threadIdx.x) asc_stcg(counts + batch, kLruSlots);
    asc_threadfence();
}
#endif
__aicore__ inline void FillRoundWords(uint64_t address,uint32_t words,
    LocalTensor<uint32_t> local) {
    GlobalTensor<uint32_t> target;target.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(address));
    for(uint32_t offset=0;offset<words;offset+=kvip_round_init::kFillWords) {
        uint32_t count=words-offset;if(count>kvip_round_init::kFillWords)count=kvip_round_init::kFillWords;
        DataCopy(target[offset],local,count);
    }
    // Caller joins the whole fill group before reusing the shared UB value.
}
__aicore__ inline void InitializeRound(uint32_t rt_batch,uint32_t rt_history,uint32_t rt_rounds,uint32_t rt_serviceLanes,uint32_t rt_finalWorkers,uint32_t rt_finalPostOffset,__gm__ Context* ctx,uint32_t b,__gm__ pipeline_ab::Args* streaming,bool& noPrefetch,bool stageRoute=false,__gm__ uint64_t* stageTicks=nullptr
#if KVIP_AUTO_ROUTE_SIMT
    ,bool parallelRoute=false,__gm__ uint64_t* routeTicks=nullptr
#endif
    ) {
    TPipe pipe;TBuf<TPosition::VECCALC> storage;pipe.InitBuffer(storage,kvip_round_init::kFillWords*4);
    auto local=storage.Get<uint32_t>();
#if defined(__CCE_AICORE__) && KVIP_LOW_MISS_POST_INDEX && KVIP_AUTO_ROUTE_ROW_PUBLISH
    if(stageRoute){
        kvip_auto_route::StageAndEvaluate(ctx,b,local,noPrefetch);
        if(stageTicks){stageTicks[27]=GetSystemCycle();stageTicks[28]=GetSystemCycle();}
    }
#endif
#if KVIP_AUTO_ROUTE_SIMT
    if(parallelRoute){
        Simt::VF_CALL<kvip_auto_route::EvaluateParallel<Context>>(Simt::Dim3{32,1,1},ctx,b,
            reinterpret_cast<__ubuf__ uint32_t*>(local.GetPhyAddr()));
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        noPrefetch=local.GetValue(0)!=0;
        if(local.GetValue(1)){
            auto config=reinterpret_cast<__gm__ kvip_auto_route::Config*>(ctx->autoRoute);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(config->pending)+b*128,128);
            if(b%8==0){uint32_t rows=rt_batch-b;if(rows>8)rows=8;
                kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->admissionControls)+b*16,rows*16);}
        }
        if(routeTicks){routeTicks[27]=GetSystemCycle();routeTicks[28]=GetSystemCycle();}
    }
#endif
    const bool lowPostIndex=kvip_round_init::LowPostIndexRound(
        streaming!=nullptr,noPrefetch,streaming && streaming->postIndex);
    Duplicate(local,uint32_t(0),kvip_round_init::kFillWords);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    FillRoundWords(ctx->rows+b*sizeof(Row),sizeof(Row)/4,local);
    // A zero-window round has no progress consumer or prefetch/final worker.
    // The next high round clears the complete original protocol state again.
    if(!lowPostIndex) {
        FillRoundWords(ctx->progress+b*4*sizeof(Progress),4*sizeof(Progress)/4,local);
        FillRoundWords(ctx->owners+b*128,32,local);
        FillRoundWords(ctx->owners+rt_batch*128+b*rt_finalWorkers*kFinalStateWords*4,rt_finalWorkers*kFinalStateWords,local);
    }
    FillRoundWords(ctx->owners+rt_finalPostOffset+b*rt_finalWorkers*sizeof(FinalPost),rt_finalWorkers*sizeof(FinalPost)/4,local);
    FillRoundWords(ctx->rowStatus+b*kRowStatusWords*4,kRowStatusWords,local);
    FillRoundWords(ctx->finalStatus+b*2*sizeof(KvafBatchGetCoreStatus),2*sizeof(KvafBatchGetCoreStatus)/4,local);
    // Producer/consumer flags are per-round ephemeral state; initialize them
    // in the existing init kernel, not as an extra graph launch.
    if(streaming && !lowPostIndex) {
        uint32_t fw=(streaming->prefetchCount>>16)&0xffU,ds=rt_batch*2048;
        uint32_t meta=(3*rt_batch+31)&~31U;
        for(uint32_t g=b;g<4*fw;g+=rt_batch)
            FillRoundWords(streaming->packedFetch+uint64_t(4*fw*(3*ds+meta)+g*(6*ds+64)+6*ds)*4,64,local);
        FillRoundWords(streaming->flags+uint64_t(b)*pipeline_ab::kFlagCount*sizeof(pipeline_ab::BatchFlag),
                       pipeline_ab::kFlagCount*sizeof(pipeline_ab::BatchFlag)/4,local);
        FillRoundWords(streaming->workerStatus+uint64_t(b)*128,32,local);
    }
    // LI classification remains active on both routes and publishes these
    // per-query errors, including padding rows, in the next graph kernel.
    if(streaming)
        FillRoundWords(streaming->queryErrors+uint64_t(b)*4*128,4*32,local);
    // Both hot and cold predicates are overwritten before progress is
    // published. Original per-query snapshot clipping excludes unready bits.
    // Optional exhaustion diagnostics still consume Event counters.
    if(kvip_round_init::ClearEvents(lowPostIndex))
        for(uint32_t e=b;e<(rt_rounds?rt_rounds:1)*rt_serviceLanes;e+=rt_batch)
            FillRoundWords(ctx->events+e*sizeof(Event),sizeof(Event)/4,local);
    if(!b) {
        *reinterpret_cast<__gm__ uint32_t*>(ctx->count)=0;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->count),4);
    }
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    Duplicate(local,UINT32_MAX,kvip_round_init::kFillWords);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    if(kvip_round_init::ClearMapping(streaming != nullptr,noPrefetch)) {
#if KVIP_INVERSE_MAPPING && !V9_UBMEM
        // A stale entry is accepted only if the freshly reset inverse agrees.
        if(!KVIP_INVERSE_MAPPING_ACTIVE(ctx))
#endif
        FillRoundWords(ctx->mapping+uint64_t(b)*rt_history*4,rt_history,local);
    }
    if(ctx->changes)FillRoundWords(ctx->changes+b*kChangeCapacity*4,
        kvip_round_init::ChangeWords(streaming != nullptr,kLruSlots,kChangeCapacity),local);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    // Copy only the candidate prefix; resolver writes extended destinations.
    if(ctx->changeSlots) {
        GlobalTensor<uint32_t> source,target;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+b*kLruSlots);
        target.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->changeSlots)+b*kChangeCapacity);
        DataCopy(local,source,kLruSlots);
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataCopy(target,local,kLruSlots);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }
    pipe.Destroy();
}
#endif
extern "C" __global__ __aicore__ void v9_init_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    auto runtimeShape = rotating::LoadRuntimeShape(reinterpret_cast<__gm__ rotating::Context*>(reinterpret_cast<__gm__ uint64_t*>(bindings)[14]));
    const uint32_t rt_batch = runtimeShape.batch;
    const uint32_t rt_history = runtimeShape.history;
    const uint32_t rt_resident = runtimeShape.resident;
    const uint32_t rt_rounds = runtimeShape.rounds;
    const uint32_t rt_serviceLanes = runtimeShape.serviceLanes;
    const uint32_t rt_finalWorkers = runtimeShape.finalWorkers;
    const uint32_t rt_finalPostOffset = runtimeShape.finalPostOffset;
    const uint32_t rt_tokenBytes = runtimeShape.tokenBytes;

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        uint32_t b=GetBlockIdx(); if(b>=rt_batch) return;
        auto ctx=GetContext(bindings);
        auto initBindings=reinterpret_cast<__gm__ uint64_t*>(bindings);
        auto streaming=reinterpret_cast<__gm__ pipeline_ab::Args*>(initBindings[15]);
        // Reserved suffix of the Consumer's diagnostic row. Init precedes
        // Consumer at a graph kernel boundary; neither writer touches the
        // other's words. No timestamps are collected in headline timing.
        bool profile=streaming && streaming->times && (streaming->prefetchCount & 0x80000000U);
        auto ticks=profile?reinterpret_cast<__gm__ uint64_t*>(streaming->times)+uint64_t(b)*64+32:nullptr;
        if(profile) {
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ticks+16),128);
            ticks[25]=GetSystemCycle();
            ticks[26]=GetSystemCycle();
        }
        bool noPrefetch=false;
        bool stageRoute=false;
#if KVIP_LOW_MISS_POST_INDEX && KVIP_AUTO_ROUTE_ROW_PUBLISH
        if(ctx->autoRoute){
            auto c=reinterpret_cast<const __gm__ kvip_auto_route::Config*>(ctx->autoRoute);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->autoRoute),128);
            stageRoute=ctx->admissionControls && kvip_auto_route::Valid(c) && c->owners<=64 &&
                rt_batch && rt_batch<=16 && b<rt_batch && !(c->pending%128) && !(ctx->admissionControls%128);
        }
#endif
#if KVIP_AUTO_ROUTE_SIMT
        bool parallelRoute=false;
        if(ctx->autoRoute){
            // The VF bypasses scalar caches, but this alignment check still
            // reads Config on the scalar pipe before choosing either route.
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->autoRoute),128);
            parallelRoute=!stageRoute && kvip_auto_route::CanPublishRows(ctx);
        }
#endif
#if KVIP_AUTO_ROUTE
        if (!stageRoute && (!b || kvip_round_init::kSparseMapping) && ctx->autoRoute
#if KVIP_AUTO_ROUTE_SIMT
            && !parallelRoute
#endif
            ) {
            auto config = reinterpret_cast<__gm__ kvip_auto_route::Config*>(ctx->autoRoute);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(config),128);
            if (kvip_auto_route::Valid(config)) {
                auto requests = reinterpret_cast<__gm__ kvip_auto_route::Request*>(config->requests);
                kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(requests),rt_batch*32);
                for (uint32_t i=0;i<rt_batch;++i) if (requests[i].owner<config->owners)
                    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(config->feedback)+requests[i].owner*128,128);
                if constexpr (kvip_round_init::kSparseMapping) {
                    if(ctx->rowOwners)kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->rowOwners),rt_batch*4);
                    if(ctx->queryEnds)kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->queryEnds),rt_batch*16);
                }
#if KVIP_AUTO_ROUTE_ROW_PUBLISH
                if (kvip_auto_route::CanPublishRows(ctx)) {
                    if (kvip_auto_route::Evaluate<true,true>(ctx,noPrefetch,b)) {
                        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(config->pending)+b*128,128);
                        if (b%8==0) {
                            uint32_t rows=rt_batch-b;if(rows>8)rows=8;
                            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->admissionControls)+b*16,rows*16);
                        }
                    }
                } else
#endif
                if (!b) {
                    if (kvip_auto_route::Evaluate<true>(ctx,noPrefetch)) {
                        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(config->pending),rt_batch*128);
                        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->admissionControls),rt_batch*16);
                    }
                } else {
                    // No inter-block polling: evaluate the same immutable
                    // request/feedback inputs without touching block 0's writes.
                    kvip_auto_route::Evaluate<false>(ctx,noPrefetch);
                }
            }
        }
#endif
        if(profile && !stageRoute) { ticks[27]=GetSystemCycle(); ticks[28]=GetSystemCycle(); }
        InitializeRound(rt_batch,rt_history,rt_rounds,rt_serviceLanes,rt_finalWorkers,rt_finalPostOffset,ctx,b,streaming,noPrefetch,stageRoute,ticks
#if KVIP_AUTO_ROUTE_SIMT
            ,parallelRoute,ticks
#endif
            );
#if KVIP_LOW_MISS_POST_INDEX
        // Evaluate computes one batch-wide decision. Publish it once at the
        // existing kernel boundary; the tail does not rescan request rows.
        if(!b && streaming && streaming->postIndex) {
            auto decision=reinterpret_cast<__gm__ uint64_t*>(streaming->postIndex)+14;
            *decision=noPrefetch?1U:0U;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(decision),sizeof(uint64_t));
        }
#endif
        if(profile)ticks[29]=GetSystemCycle();
        auto args=GetArgs(ctx,b*2);
        auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
        auto entity=reinterpret_cast<__gm__ KvafBatchGetDirectUrmaEntity*>(args.channel_entity);
        auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args.queue_state);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
        row->entity=args.channel_entity; row->queue=args.queue_state;
        const uint64_t prefix=4*2048*6+1024+4*uint64_t(rt_history)*2;
        const uint64_t tail=uint64_t(rt_history)*8+8192*64+512;
        auto a=reinterpret_cast<__gm__ uint64_t*>(bindings);
#if KVIP_AUTO_ROUTE_SIMT || (KVIP_LOW_MISS_POST_INDEX && KVIP_AUTO_ROUTE_ROW_PUBLISH)
        // SIMT/MTE2 bypass the scalar cache. Refresh the owner before scalar
        // binding so a reordered request cannot reuse the previous arena.
        if((stageRoute
#if KVIP_AUTO_ROUTE_SIMT
            || parallelRoute
#endif
            ) && ctx->rowOwners)
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->rowOwners)+b*sizeof(uint32_t),sizeof(uint32_t));
#endif
        uint32_t owner=ctx->rowOwners ? reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b] : b;
        uint64_t outputStride=ctx->outputStride ? ctx->outputStride : rt_resident;
        row->addresses=ctx->tokenAddresses ? ctx->tokenAddresses+uint64_t(owner)*ctx->addressStride*8 : a[13]+rt_batch*prefix+b*tail;
        row->output=args.output+uint64_t(owner)*outputStride*rt_tokenBytes;
        const bool lowPostIndex=kvip_round_init::LowPostIndexRound(
            streaming!=nullptr,noPrefetch,streaming && streaming->postIndex);
#if KVIP_LOW_MISS_POST_INDEX
        if(lowPostIndex) {
            // Low classification resolves ABC slots arithmetically; no map
            // entry is consumed. Keep the published change-count ABI.
            if(ctx->changeCount)
                Simt::VF_CALL<InitializeLowChangeCount>(Simt::Dim3{32,1,1},
                    reinterpret_cast<__gm__ uint32_t*>(ctx->changeCount),b);
        } else
#endif
        Simt::VF_CALL<InitializeTailMappings>(Simt::Dim3{256,1,1},rt_history,rt_resident,ctx,b);
        if(profile) {
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            ticks[30]=GetSystemCycle();
        }
        row->initialHead=queue->sq_producer_index;
        row->limit=kSlots;
        row->prefetchLimit=ctx->prefetchLimit;
        row->queueLimit=entity->sq_depth>1 ? entity->sq_depth-1 : 0;
        row->error=queue->last_error;
#if KVIP_ADAPTIVE_ADMISSION
#if KVIP_AUTO_ROUTE
        if (ctx->autoRoute) {
            // Init blocks do not read controls written by another block.
            // A low round performs no prefetch, so this full limit is unused.
            auto config = reinterpret_cast<__gm__ kvip_auto_route::Config*>(ctx->autoRoute);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(config),128);
            if (kvip_auto_route::Valid(config)) row->prefetchLimit=uint32_t(config->fullTotal);
            else row->error=kvaf_device::kInvalidArgs;
        } else
#endif
        if(ctx->admissionControls) {
            auto input=reinterpret_cast<const __gm__ int32_t*>(ctx->admissionControls)+b*4;
            kvip_admission::Controls controls{input[0],input[1],input[2],input[3]};
            if(kvip_admission::Valid(controls))row->prefetchLimit=uint32_t(controls.total);
            else row->error=kvaf_device::kInvalidArgs;
        }
#endif
        if(entity->sq_depth<2 || (entity->sq_depth&(entity->sq_depth-1)) || entity->cq_depth<2 ||
           !entity->sq_va || !entity->sq_db_va || ctx->prefetchLimit>kSlots ||
           ctx->windowLimit==0 || ctx->windowLimit>480)
            row->error=kvaf_device::kInvalidArgs;
#if !V9_SIMULATION
        if(queue->initialized && queue->magic!=kvaf_device::kQueueMagic)
            row->error=kvaf_device::kInvalidArgs;
        if(!queue->initialized) {
            queue->magic=kvaf_device::kQueueMagic; queue->initialized=1;
        }
        if(entity->sq_depth>(1U<<30) ||
#if KVIP_ADAPTIVE_ADMISSION
           row->initialHead>UINT64_MAX-row->prefetchLimit ||
#else
           row->initialHead>UINT64_MAX-ctx->prefetchLimit ||
#endif
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
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
#endif
        if(lowPostIndex) {
            // InitializeRound zeroed Row by DMA. Only these two cache lines
            // were modified by the scalar bindings and unchanged validation.
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row),64);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row)+offsetof(Row,limit),
                                        offsetof(Row,initialHead)+sizeof(row->initialHead)-offsetof(Row,limit));
        } else {
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row),sizeof(*row));
        }
        if(profile) {
            ticks[31]=GetSystemCycle();
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ticks+16),128);
        }
    }
#endif
}
#if defined(__CCE_AICORE__)
#endif
// Failure-only diagnostic. Never called by the forward pipeline; queue reads
// must execute on device because driver queue VAs are not ACL allocations.
extern "C" __global__ __aicore__ void kvip_transport_snapshot_kernel(
    GM_ADDR source, GM_ADDR destination, uint32_t bytes) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        TPipe pipe;
        TBuf<TPosition::VECCALC> storage;
        constexpr uint32_t kChunkBytes = 4096;
        pipe.InitBuffer(storage, kChunkBytes);
        auto local = storage.Get<uint32_t>();
        GlobalTensor<uint32_t> input, output;
        input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(source));
        output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(destination));
        for (uint32_t offset = 0; offset < bytes; offset += kChunkBytes) {
            uint32_t count = bytes - offset;
            if (count > kChunkBytes) count = kChunkBytes;
            DataCopy(local, input[offset / 4], count / 4);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            DataCopy(output[offset / 4], local, count / 4);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        pipe.Destroy();
    }
#endif
}
