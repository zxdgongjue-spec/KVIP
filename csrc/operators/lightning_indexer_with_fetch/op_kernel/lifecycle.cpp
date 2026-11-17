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
__aicore__ inline void FillRoundWords(uint64_t address,uint32_t words,
    LocalTensor<uint32_t> local) {
    GlobalTensor<uint32_t> target;target.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(address));
    for(uint32_t offset=0;offset<words;offset+=4096) {
        uint32_t count=words-offset;if(count>4096)count=4096;
        DataCopy(target[offset],local,count);
    }
    // Caller joins the whole fill group before reusing the shared UB value.
}
__aicore__ inline void InitializeRound(uint32_t rt_batch,uint32_t rt_history,uint32_t rt_rounds,uint32_t rt_serviceLanes,uint32_t rt_finalWorkers,uint32_t rt_finalPostOffset,__gm__ Context* ctx,uint32_t b,__gm__ pipeline_ab::Args* streaming) {
    TPipe pipe;TBuf<TPosition::VECCALC> storage;pipe.InitBuffer(storage,4096*4);
    auto local=storage.Get<uint32_t>();
    Duplicate(local,uint32_t(0),4096);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    FillRoundWords(ctx->incremental+b*512,128,local);
    FillRoundWords(kvip_prefetch_rank::CoveredAddress(ctx->incremental,rt_batch,rt_history,b),(rt_history+255)/256*8,local);
    FillRoundWords(ctx->rows+b*sizeof(Row),sizeof(Row)/4,local);
    FillRoundWords(ctx->progress+b*4*sizeof(Progress),4*sizeof(Progress)/4,local);
    FillRoundWords(ctx->owners+b*128,32,local);
    FillRoundWords(ctx->owners+rt_batch*128+b*rt_finalWorkers*kFinalStateWords*4,rt_finalWorkers*kFinalStateWords,local);
    FillRoundWords(ctx->owners+rt_finalPostOffset+b*rt_finalWorkers*sizeof(FinalPost),rt_finalWorkers*sizeof(FinalPost)/4,local);
    FillRoundWords(ctx->rowStatus+b*kRowStatusWords*4,kRowStatusWords,local);
    FillRoundWords(ctx->finalStatus+b*2*sizeof(KvafBatchGetCoreStatus),2*sizeof(KvafBatchGetCoreStatus)/4,local);
    // Producer/consumer flags are per-round ephemeral state; initialize them
    // in the existing init kernel, not as an extra graph launch.
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
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    Duplicate(local,UINT32_MAX,4096);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    if(ctx->changes)FillRoundWords(ctx->changes+b*kChangeCapacity*4,kChangeCapacity,local);
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
        uint32_t block=GetBlockIdx();
        uint32_t b=block%rt_batch, part=block/rt_batch;
        if(part>=4) return;
        auto ctx=GetContext(bindings);
        // Each worker owns an aligned range, including its ABC mappings.
        // No cache line is concurrently cleared and filled by another core.
        uint32_t chunk=((rt_history+127)/128)*32;
        uint32_t begin=part*chunk;
        uint32_t end=begin+chunk<rt_history ? begin+chunk : rt_history;
        if(begin<end) {
            TPipe mapPipe; TBuf<TPosition::VECCALC> mapStorage;
            mapPipe.InitBuffer(mapStorage,4096*4);
            auto local=mapStorage.Get<uint32_t>();
            Duplicate(local,UINT32_MAX,4096);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            FillRoundWords(ctx->mapping+(uint64_t(b)*rt_history+begin)*4,end-begin,local);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            mapPipe.Destroy();
            Simt::VF_CALL<InitializeTailMappings>(Simt::Dim3{256,1,1},rt_history,rt_resident,ctx,b,begin,end);
        }
        if(part) return;
        auto initBindings=reinterpret_cast<__gm__ uint64_t*>(bindings);
        auto streaming=reinterpret_cast<__gm__ pipeline_ab::Args*>(initBindings[15]);
        InitializeRound(rt_batch,rt_history,rt_rounds,rt_serviceLanes,rt_finalWorkers,rt_finalPostOffset,ctx,b,streaming);
        auto args=GetArgs(ctx,b*2);
        auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
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
        row->output=args.output+uint64_t(owner)*outputStride*rt_tokenBytes;
        row->initialHead=queue->sq_producer_index;
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
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
#endif
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row),sizeof(*row));
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
