#pragma once
namespace rotating {
__aicore__ inline __ubuf__ uint32_t* FinalPtr(LocalTensor<uint32_t> x) {return reinterpret_cast<__ubuf__ uint32_t*>(x.GetPhyAddr());}

__aicore__ inline __gm__ FinalPost* FinalPostAt(__gm__ Context* ctx,uint32_t core) {
    return reinterpret_cast<__gm__ FinalPost*>(ctx->owners+kFinalPostOffset)+core;
}
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
    // selected/loc have finished all SIMD/SIMT use and locations have been
    // committed above. Reuse at most [0,21504) of selected, never slots or
    // address vectors. Each destination sector belongs to one final AIV.
    UbmemCopyTrace trace;trace.Begin();
    UbmemFinalCopyPipeline copyPipe(ctx,row->output,local.ReinterpretCast<uint8_t>());
    // ReplicatedResolve assigns slot=SIM_RESIDENT+row->slots+rank until
    // capacity is exhausted. Walk only this AIV's two-row sector pairs in
    // that case, rather than loading every other worker's slot in scalar.
    // Recycled slots need the original full scan because their gaps are not
    // predictable from rank. Keep the actual slot/owner and Add checks below.
    const bool consecutive=!ctx->lruSlots && !ctx->hotValid && !ctx->reservedHot && row->slots<=kSlots && batchCount<=kSlots-row->slots;
    const uint32_t firstSlot=SIM_RESIDENT+row->slots;
    uint32_t firstRank=0;
    if(consecutive) {
        const uint32_t pairOwner=(firstSlot>>1)%kFinalWorkers;
        const uint32_t pairs=(w+kFinalWorkers-pairOwner)%kFinalWorkers;
        firstRank=pairs?2*pairs-(firstSlot&1):0;
    }
    for(uint32_t rank=firstRank;rank<batchCount;
        rank+=consecutive?(((firstSlot+rank)&1)?2*kFinalWorkers-1:1):1) {
        uint32_t slot=slots.GetValue(rank);
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
