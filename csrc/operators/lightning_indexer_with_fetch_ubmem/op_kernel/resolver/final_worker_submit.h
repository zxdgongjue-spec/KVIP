#pragma once
#include "../device_context.h"
namespace rotating {
__aicore__ inline __ubuf__ uint32_t* FinalPtr(LocalTensor<uint32_t> x) {return reinterpret_cast<__ubuf__ uint32_t*>(x.GetPhyAddr());}

__aicore__ inline void PublishFinalWorker(__gm__ Context* ctx,__gm__ Row* row,
    uint32_t b,uint32_t w,uint32_t own,uint32_t batchCount,uint32_t descriptor,uint32_t error,
    LocalTensor<uint32_t> local,LocalTensor<uint32_t> remoteLow,LocalTensor<uint32_t> remoteHigh,
    LocalTensor<uint32_t> slots,LocalTensor<uint32_t> ctl) {
    auto post=FinalPostAt(ctx,b*kFinalWorkers+w);
    // Resolve may fail before constructing the slot/address vectors. Publish
    // that error without reading uninitialized UB, even when batchCount is set.
    if(error) {
        post->count=0;post->pending=0;post->error=error;
        post->deferred=0;post->descriptor=descriptor;post->head=0;
        ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
        return;
    }
    uint32_t scheduled=0,completed=0;
    // selected/loc have finished all SIMD/SIMT use. ReplicatedFinal joined
    // their last location-store DMA before entering here. Both banks are dead:
    // [0,65536) may hold the final ring; keys starts at 65536 and remains live.
    // The opt-in sixteen-row ring uses 41984 bytes without adding UB storage.
    // Each destination sector still belongs to one final AIV.
    static_assert(UbmemFinalCopyPipeline::kScratchBytes<=2*kSlots*sizeof(uint32_t),
                  "Final copy must not overwrite live address vectors");
    UbmemCopyTrace trace;trace.Begin();
    UbmemFinalCopyPipeline copyPipe(ctx,row->output,local.ReinterpretCast<uint8_t>());
    // 2624-byte rows are complete 32-byte sectors. The resolver proves unique
    // destination slots, so its contiguous rank ranges also balance actual
    // payload work and avoid every worker scanning the full plan. 656-byte
    // rows keep sector ownership: adjacent rows share a partial sector.
    const uint32_t begin=V9_UBMEM_ORDERED_PAIR?batchCount*w/kFinalWorkers:0;
    const uint32_t end=V9_UBMEM_ORDERED_PAIR?batchCount*(w+1)/kFinalWorkers:batchCount;
    for(uint32_t rank=begin;rank<end;++rank) {
        uint32_t slot=slots.GetValue(rank);
        if constexpr(!V9_UBMEM_ORDERED_PAIR)
            if(UbmemDestinationOwner(slot,kFinalWorkers)!=w)continue;
        ++scheduled;
        if(error)continue;
        uint64_t peer=uint64_t(remoteLow.GetValue(rank)) | (uint64_t(remoteHigh.GetValue(rank))<<32);
        bool accepted=copyPipe.Add(peer,slot);
        if(!accepted) {
            error|=kUbmemInvalidAddress;continue;
        }
        ++completed;
    }
    trace.IssueEnd();
    completed=copyPipe.Finish();
    trace.Done();trace.SaveFinal(post,completed);
    copyPipe.SaveFinalCommandCounts(reinterpret_cast<__gm__ uint64_t*>(post)+12);
    post->count=scheduled;post->pending=scheduled!=completed;post->error=error;
    post->deferred=0;post->descriptor=descriptor;
    // ABI8 head is actual completed rows, never a fabricated SQ/CQ index.
    post->head=completed;
    ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
}
}
