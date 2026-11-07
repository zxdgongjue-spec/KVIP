#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "context.h"
#include "prefetch.h"
#include "resolver.h"
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
    TPipe pipe;TBuf<TPosition::VECCALC> scratch;pipe.InitBuffer(scratch,4096*4);
    auto local=scratch.Get<uint32_t>();
    Duplicate(local,uint32_t(0),4096);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    FillWords(ctx->rows+b*sizeof(Row),sizeof(Row)/4,local);
    FillWords(ctx->progress+b*4*sizeof(Progress),4*sizeof(Progress)/4,local);
    FillWords(ctx->owners+b*128,32,local);
    FillWords(ctx->owners+SIM_BATCH*128+b*kFinalWorkers*kFinalStateWords*4,kFinalWorkers*kFinalStateWords,local);
    FillWords(ctx->owners+kFinalPostOffset+b*kFinalWorkers*sizeof(FinalPost),kFinalWorkers*sizeof(FinalPost)/4,local);
    FillWords(ctx->rowStatus+b*kRowStatusWords*4,kRowStatusWords,local);
    if(ctx->ubmemStatus)FillWords(ctx->ubmemStatus+b*kUbmemStatusWords*8,kUbmemStatusWords*2,local);
    for(uint32_t e=b;e<(kRounds?kRounds:1)*kServiceLanes;e+=SIM_BATCH)
        FillWords(ctx->events+e*sizeof(Event),sizeof(Event)/4,local);
    if(!b) {
        *reinterpret_cast<__gm__ uint32_t*>(ctx->count)=0;
        ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->count),4);
    }
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    Duplicate(local,UINT32_MAX,4096);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    FillWords(ctx->mapping+uint64_t(b)*SIM_HISTORY*4,SIM_HISTORY,local);
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
    Simt::VF_CALL<InitializeTailMappings>(Simt::Dim3{256,1,1},SIM_HISTORY,SIM_RESIDENT,ctx,b);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    pipe.Destroy();
}
__aicore__ inline void CompleteBatch(__gm__ Context* ctx,uint32_t b) {
    auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
    auto info=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*kRowStatusWords;
    ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row),sizeof(Row));
    ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(info),kRowStatusWords*4);
    auto region=UbmemRegionAt(ctx);
    uint64_t error=info[0]|row->error;
    if(!ctx->ubmemStatus || !UbmemRegionValid(region))error|=kUbmemInvalidRegion;
    uint64_t scheduled=0,completed=0;
    for(uint32_t w=0;w<kFinalWorkers;++w) {
        auto post=FinalPostAt(ctx,b*kFinalWorkers+w);
        ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(FinalPost));
        scheduled+=post->count;completed+=post->head;error|=post->error;
        if(post->pending || post->deferred || post->count!=post->head)error|=kUbmemIncomplete;
    }
    if(row->pending || scheduled!=info[1] || scheduled!=completed)error|=kUbmemIncomplete;
    if(ctx->ubmemStatus) {
        auto status=UbmemStatusAt(ctx,b);
        ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),128);
        if(status[2]!=row->requests)error|=kUbmemIncomplete;
        status[0]=error;status[1]=row->requests;status[3]=scheduled;status[4]=completed;
        status[5]=(status[2]+completed)*kTokenBytes;
        status[6]=UbmemRegionValid(region)?region->generation:0;
        ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),128);
    }
    row->error=static_cast<uint32_t>(error);
    info[0]|=static_cast<uint32_t>(error);info[3]=0;info[4]=static_cast<uint32_t>(completed);
    if(!error) {
        row->finalPending=0;
        Simt::VF_CALL<PublishCompletedResidency>(Simt::Dim3{256,1,1},SIM_RESIDENT,ctx,b,static_cast<uint32_t>(completed));
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
    ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row),sizeof(Row));
    ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(info),32);
}
}
#endif
extern "C" __global__ __aicore__ void kvip_ubmem_init_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        uint32_t b=GetBlockIdx();if(b>=SIM_BATCH)return;
        auto ctx=GetContext(bindings);InitializeRound(ctx,b);
        auto region=UbmemRegionAt(ctx);
        auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
        uint64_t owner=ctx->rowOwners?reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b]:b;
        uint64_t stride=ctx->outputStride?ctx->outputStride:SIM_RESIDENT;
        uint64_t rowBytes=stride*kTokenBytes;
        uint32_t error=(!ctx->ubmemStatus || !UbmemRegionValid(region) || !ctx->tokenAddresses ||
            !ctx->lruSlots || !ctx->reservedHot || !ctx->changeSlots || !ctx->changeCount ||
            ctx->addressStride<SIM_HISTORY || stride<SIM_RESIDENT || (stride-SIM_RESIDENT)%128 || (ctx->tailBounds && stride<SIM_RESIDENT+384) ||
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
extern "C" __global__ __aicore__ void kvip_ubmem_resolve_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {ReplicatedFinal(GetContext(bindings));}
#endif
}
extern "C" __global__ __aicore__ void kvip_ubmem_complete_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {uint32_t b=GetBlockIdx();if(b<SIM_BATCH)CompleteBatch(GetContext(bindings),b);}
#endif
}
