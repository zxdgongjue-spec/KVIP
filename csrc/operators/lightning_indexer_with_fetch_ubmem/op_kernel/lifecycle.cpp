#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "context.h"
#include "../../lightning_indexer_with_fetch/op_kernel/pipeline_types.h"
#include "prefetch.h"
#include "ubmem/fused_batch_get_device.h"
using namespace AscendC;
using namespace rotating;
namespace {
__aicore__ inline __gm__ Context* GetContext(GM_ADDR bindings) {
    return reinterpret_cast<__gm__ Context*>(reinterpret_cast<__gm__ uint64_t*>(bindings)[14]);
}
__aicore__ inline void FillWords(uint64_t address,uint32_t words,LocalTensor<uint32_t> local) {
    GlobalTensor<uint32_t> output;output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(address));
    for(uint32_t offset=0;offset<words;offset+=4096) {
        uint32_t count=words-offset;if(count>4096)count=4096;
        DataCopy(output[offset],local,count);
    }
}
__aicore__ inline void InitializeRound(__gm__ Context* ctx,uint32_t b) {
    const auto shape=LoadRuntimeShape(ctx);
    TPipe pipe;TBuf<TPosition::VECCALC> scratch;pipe.InitBuffer(scratch,4096*4);
    auto local=scratch.Get<uint32_t>();
    Duplicate(local,uint32_t(0),4096);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    FillWords(ctx->rows+b*sizeof(Row),sizeof(Row)/4,local);
    FillWords(ctx->progress+b*4*sizeof(Progress),4*sizeof(Progress)/4,local);
    FillWords(ctx->owners+b*128,32,local);
    FillWords(ctx->owners+shape.batch*128+b*shape.finalWorkers*kFinalStateWords*4,shape.finalWorkers*kFinalStateWords,local);
    FillWords(ctx->owners+shape.finalPostOffset+b*shape.finalWorkers*sizeof(FinalPost),shape.finalWorkers*sizeof(FinalPost)/4,local);
    FillWords(ctx->rowStatus+b*kRowStatusWords*4,kRowStatusWords,local);
    if(ctx->ubmemStatus)FillWords(ctx->ubmemStatus+b*kUbmemStatusWords*8,kUbmemStatusWords*2,local);
    for(uint32_t e=b;e<(shape.rounds?shape.rounds:1)*shape.serviceLanes;e+=shape.batch)
        FillWords(ctx->events+e*sizeof(Event),sizeof(Event)/4,local);
    if(!b) {
        *reinterpret_cast<__gm__ uint32_t*>(ctx->count)=0;
        ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->count),4);
    }
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    Duplicate(local,UINT32_MAX,4096);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    FillWords(ctx->mapping+uint64_t(b)*shape.history*4,shape.history,local);
    FillWords(ctx->selected+uint64_t(b)*kSlots*4,kSlots,local);
    if(ctx->changes)FillWords(ctx->changes+b*kChangeCapacity*4,kChangeCapacity,local);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    if(ctx->changeSlots) {
        GlobalTensor<uint32_t> source,target;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+b*kLruSlots);
        target.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->changeSlots)+b*kChangeCapacity);
        DataCopy(local,source,kLruSlots);
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataCopy(target,local,kLruSlots);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }
    Simt::VF_CALL<InitializeTailMappings>(Simt::Dim3{256,1,1},shape.history,shape.resident,ctx,b);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    pipe.Destroy();
}
}
#endif
extern "C" __global__ __aicore__ void kvip_ubmem_init_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        auto ctx=GetContext(bindings);const auto shape=LoadRuntimeShape(ctx);
        uint32_t b=GetBlockIdx();if(b>=shape.batch)return;
        InitializeRound(ctx,b);
        auto a=reinterpret_cast<__gm__ pipeline_ab::Args*>(reinterpret_cast<__gm__ uint64_t*>(bindings)[15]);
        TPipe pipe;TBuf<TPosition::VECCALC> scratch;pipe.InitBuffer(scratch,4096*4);
        auto zero=scratch.Get<uint32_t>();Duplicate(zero,0U,4096);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        FillWords(a->flags+uint64_t(b)*pipeline_ab::kFlagCount*sizeof(pipeline_ab::BatchFlag),pipeline_ab::kFlagCount*32,zero);
        FillWords(a->workerStatus+uint64_t(b)*128,32,zero);
        FillWords(a->queryErrors+uint64_t(b)*512,128,zero);
        uint32_t workers=(a->prefetchCount>>16)&255,ds=shape.batch*2048,meta=(3*shape.batch+31)&~31U;
        for(uint32_t g=b;g<4*workers;g+=shape.batch)
            FillWords(a->packedFetch+uint64_t(4*workers*(3*ds+meta)+g*(6*ds+64)+6*ds)*4,64,zero);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);pipe.Destroy();
        auto region=UbmemRegionAt(ctx);
        auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
        uint64_t owner=ctx->rowOwners?reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b]:b;
        uint64_t stride=ctx->outputStride?ctx->outputStride:shape.resident;
        uint64_t rowBytes=stride*shape.tokenBytes;
        uint32_t error=(!ctx->ubmemStatus || !UbmemRegionValid(region,shape.resident) || !ctx->tokenAddresses ||
            !ctx->lruSlots || !ctx->reservedHot || !ctx->changeSlots || !ctx->changeCount ||
            ctx->addressStride<shape.history || stride<shape.resident || (stride-shape.resident)%128 || (ctx->tailBounds && stride<shape.resident+384) ||
            rowBytes%32 || ctx->prefetchLimit>kLruSlots || !ctx->windowLimit || ctx->windowLimit>480)?kUbmemInvalidRegion:0;
        if(!error && (rowBytes>region->outputBytes || owner>region->outputBytes/rowBytes-1))error=kUbmemInvalidRegion;
        row->reserved0=0;row->reserved1=0;row->reserved2=0;
        row->addresses=ctx->tokenAddresses+owner*ctx->addressStride*8;
        row->output=error?0:region->outputBase+owner*rowBytes;
        row->limit=kSlots;row->prefetchLimit=ctx->prefetchLimit;row->copyLimit=kLruSlots;row->error=error;
        ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row),sizeof(Row));
    }
#endif
}
