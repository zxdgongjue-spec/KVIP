#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "context.h"
#include "pipeline_types.h"
#include "seed_mask.h"

#include <stdint.h>
#include "prefetch.h"
#include "fetch_packets.h"
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
#if KVIP_SEED_MASK && !KVIP_SEED_PARALLEL
    TBuf<TPosition::VECCALC> seedInputs;
    if (ctx->shape.resident <= kSeedMaskSlots) {
        pipe.InitBuffer(seedInputs, kSeedChunk * 2 * sizeof(uint32_t));
        LoadSeedInputs(ctx, b, ctx->shape.resident, seedInputs.Get<uint32_t>(), 0, ctx->shape.resident);
    }
#endif
    auto local=storage.Get<uint32_t>();
    Duplicate(local,uint32_t(0),4096);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
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
    FillRoundWords(ctx->mapping+uint64_t(b)*rt_history*4,rt_history,local);
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
#if KVIP_SEED_MASK && !KVIP_SEED_PARALLEL
    if (ctx->shape.resident <= kSeedMaskSlots) {
        // The previous DMA has joined. Reuse local for the predicate output.
        PrepareSeedMask(ctx, b, 0, ctx->shape.resident, seedInputs.Get<uint32_t>(), local);
    }
#endif
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
        uint32_t b=GetBlockIdx();
        auto ctx=GetContext(bindings);
        if (b >= rt_batch) {
#if KVIP_SEED_PARALLEL
            const uint32_t worker = b - rt_batch;
            BuildSeedStripe(ctx, worker % rt_batch, rt_resident, worker / rt_batch,
                            SeedInitHelpers(runtimeShape));
#endif
            return;
        }
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
        Simt::VF_CALL<InitializeTailMappings>(Simt::Dim3{256,1,1},rt_history,rt_resident,ctx,b);
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

// Untimed deterministic validation uses completed device scores and the same rank/admission
// code as FastService. It never opens a transport queue or changes the serving cache.
#if KVIP_FETCH_PACKETS
__simd_vf__ inline void FixturePackScore(__ubuf__ uint16_t* input,__ubuf__ uint16_t* limit,
                                        __ubuf__ uint16_t* packet) {
    MicroAPI::RegTensor<uint16_t> score;MicroAPI::LoadAlign(score,input);
    auto all=MicroAPI::CreateMask<uint16_t,MicroAPI::MaskPattern::ALL>();
    kvip_fetch_packets::PackScore(score,limit,packet,all);
}
#endif
// Functional producer fixture; actual LI invokes the same register routine.
extern "C" __global__ __aicore__ void v9_fetch_packet_fixture_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__) && KVIP_FETCH_PACKETS
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        auto a=reinterpret_cast<__gm__ uint64_t*>(bindings);
        auto ctx=reinterpret_cast<__gm__ Context*>(a[14]);uint32_t b=GetBlockIdx();
        if(b>=ctx->shape.batch)return;
        const uint32_t h=ctx->shape.history,r=ctx->shape.resident;
        uint32_t ends[4];for(uint32_t q=0;q<4;++q)ends[q]=reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds)[b*4+q];
        auto order=kvip_cold_scan::Bounds(r,h,ends);
        TPipe pipe;TBuf<TPosition::VECCALC> buf;pipe.InitBuffer(buf,1024);
        auto raw=buf.Get<uint16_t>(),packet=raw[128],limit=raw[192];
        auto prefix=reinterpret_cast<__gm__ uint8_t*>(ctx->hits)-ctx->shape.batch*(4*kTop*6+1024);
        for(uint32_t q=0;q<4;++q) {
            GlobalTensor<uint16_t> seed,scores,output;
            seed.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(prefix+b*(4*kTop*6+1024))+q*kTop);
            scores.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(a[12])+uint64_t(b*4+q)*h);
            output.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(ctx->fetchPackets)+uint64_t(b*4+q)*h/128*48);
            DataCopy(limit,seed[kTop-16],16);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            limit.SetValue(0,limit.GetValue(15));
            SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
            for(uint32_t scan=r;scan<h;scan+=128) {
                DataCopy(raw,scores[kvip_cold_scan::Remap(scan,order)],128);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                FixturePackScore(RankPtr(raw),RankPtr(limit),RankPtr(packet));
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopy(output[scan/128*48],packet,48);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }
#endif
}

extern "C" __global__ __aicore__ void v9_fetch_fixture_kernel(GM_ADDR bindings, GM_ADDR frozenMap, uint32_t lazy) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if KVIP_FETCH_BITMAP || KVIP_FETCH_DENSE
    if ASCEND_IS_AIV {
        using namespace AscendC;
        using namespace rotating;
        auto a=reinterpret_cast<__gm__ uint64_t*>(bindings);
        auto ctx=reinterpret_cast<__gm__ Context*>(a[14]);
        uint32_t b=GetBlockIdx(),h=ctx->shape.history,r=ctx->shape.resident;
        if (b >= ctx->shape.batch) return;
        uint32_t ends[4];
        for(uint32_t q=0;q<4;++q)ends[q]=reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds)[b*4+q];
        auto order=kvip_cold_scan::Bounds(r,h,ends);
        TPipe pipe; TBuf<TPosition::VECCALC> storage;
        pipe.InitBuffer(storage,11264*sizeof(uint32_t));
        auto scratch=storage.Get<uint32_t>(),cfg=scratch[10240],sent=scratch[10496];
        auto map=reinterpret_cast<__gm__ uint32_t*>(frozenMap)+uint64_t(b)*h;
        uint32_t total=0;
#if KVIP_FETCH_PACKETS
        auto state=cfg[128];for(uint32_t q=0;q<4;++q)state.SetValue(12+q,0);
#endif
        for(uint32_t e=0;e<9;++e) {
            uint32_t missing=0, budget=512-total<64?512-total:64;
            uint32_t count;
#if KVIP_FETCH_PACKETS
            if(lazy)count=ServiceFetchDemand(ctx,scratch,cfg,sent,reinterpret_cast<GM_ADDR>(a[12]),map,
                b,h,r,9,state,e+1,ends,order,budget,total,missing,true);
            else
#endif
            count=ServiceFetchRank(ctx,scratch,cfg,sent,reinterpret_cast<GM_ADDR>(a[12]),map,
                b,h,r,9,e,e+1,ends,order,budget,total,missing,true);
            GlobalTensor<uint32_t> audit;
            audit.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->fetchAudit)+(b*9+e)*96);
            cfg.SetValue(0,e+1);cfg.SetValue(1,count);
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopy(audit,cfg,32);
            if(count)DataCopy(audit[32],sent,(count+7)/8*8);
            PipeBarrier<PIPE_ALL>();
            total+=count;
        }
    }
#endif
#endif
}

extern "C" __global__ __aicore__ void v9_fetch_clock_kernel(GM_ADDR data) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() != 0) return;
        auto dst=reinterpret_cast<__gm__ uint64_t*>(data);
        uint64_t duration=dst[0],start=AscendC::GetSystemCycle(),end=start;
        while(end-start<duration)end=AscendC::GetSystemCycle();
        dst[1]=end-start;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(data),64);
    }
#endif
}
