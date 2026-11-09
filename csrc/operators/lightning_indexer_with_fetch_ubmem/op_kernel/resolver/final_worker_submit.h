#pragma once
#include "ubmem/fused_batch_get_device.h"
namespace rotating {
__aicore__ inline __ubuf__ uint32_t* FinalPtr(LocalTensor<uint32_t> x) {return reinterpret_cast<__ubuf__ uint32_t*>(x.GetPhyAddr());}

__aicore__ inline __gm__ FinalPost* FinalPostAt(__gm__ Context* ctx,uint32_t core) {
    return reinterpret_cast<__gm__ FinalPost*>(ctx->owners+ctx->shape.finalPostOffset)+core;
}
__aicore__ inline void PublishFinalWorker(__gm__ Context* ctx,__gm__ Row* row,
    uint32_t b,uint32_t w,uint32_t own,uint32_t batchCount,uint32_t descriptor,uint32_t error,
    LocalTensor<uint32_t> local,LocalTensor<uint32_t> remoteLow,LocalTensor<uint32_t> remoteHigh,
    LocalTensor<uint32_t> slots,LocalTensor<uint32_t> ctl) {
    const auto shape=LoadRuntimeShape(ctx);
    auto post=FinalPostAt(ctx,b*shape.finalWorkers+w);
    // Resolve may fail before constructing the slot/address vectors. Publish
    // that error without reading uninitialized UB, even when batchCount is set.
    if(error) {
        post->count=0;post->pending=0;post->error=error;
        post->deferred=0;post->descriptor=descriptor;post->head=0;
        ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
        return;
    }
    // selected/loc have finished their use; KVAF borrows selected's retired
    // [0,21504) scratch and drains its MTE events before returning.
    UbmemCopyTrace trace;trace.Begin();
    auto region=UbmemRegionAt(ctx);
    kvaf_device::UbmemBatchGetDeviceResult result{};
    if(!UbmemRegionValid(region,shape.resident))result.error=kUbmemInvalidRegion;
    else {
        kvaf_device::UbmemBatchGetDeviceArgs args{
            remoteLow,remoteHigh,slots,region->peerBase,region->mappedBase,region->bytes,
            row->output,region->outputBase,region->outputBytes,batchCount,shape.resident,
            w,shape.finalWorkers,shape.tokenBytes};
        result=kvaf_device::KvafBatchGetDevice(args,local.ReinterpretCast<uint8_t>());
    }
    trace.IssueEnd();
    uint32_t scheduled=result.scheduled,completed=result.completed;error|=result.error;
    trace.Done();trace.SaveFinal(post,completed);
    post->count=scheduled;post->pending=scheduled!=completed;post->error=error;
    post->deferred=0;post->descriptor=descriptor;
    // ABI8 head is actual completed rows, never a fabricated SQ/CQ index.
    post->head=completed;
    ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
}
}
