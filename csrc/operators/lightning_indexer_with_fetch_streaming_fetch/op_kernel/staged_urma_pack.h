// SPDX-License-Identifier: Apache-2.0
// Adapt the staged Resolver's deduplicated records to the existing URMA
// FinalPost/FinishFetch contract. This header is experimental, not wired into
// the production LIFetch entry point.
#pragma once
namespace rotating {
// ctl[16+2*w] and ctl[17+2*w] publish count and safe prefix per worker.
// Other ctl banks are owned by StagedPrefix. The caller keeps the original
// address table immutable until this function has finished.
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void PackStagedUrma(uint32_t history, uint32_t resident, uint32_t tokenBytes,
    uint32_t batch, uint32_t workers, uint32_t used, uint32_t count,
    __gm__ Context* ctx, __ubuf__ uint32_t* selected, __ubuf__ uint16_t* ids,
    __ubuf__ uint32_t* slots, __ubuf__ uint32_t* ctl) {
    uint32_t tid=threadIdx.x;
    auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+batch;
    auto addresses=reinterpret_cast<__gm__ uint64_t*>(row->addresses);
    auto reserved=reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+batch*resident;
    auto blocks=reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks);
    auto offsets=reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets);
    auto destinations=reinterpret_cast<__gm__ int32_t*>(ctx->destinations);
    uint32_t ownerBase=OwnerRow(ctx,batch)*OutputStride(resident,ctx);
    for(uint32_t worker=0;worker<workers;++worker){
        uint32_t begin=count*worker/workers,end=count*(worker+1)/workers;
        uint32_t own=end-begin,per=(own+511)/512,localSafe=0;
        for(uint32_t i=tid*per;i<(tid+1)*per && i<own;++i){
            uint32_t slot=slots[begin+i];
            if(slot>=resident){asc_atomic_or(ctl+1,4U);continue;}
            int32_t ordinal=asc_ldcg(reserved+slot);
            localSafe+=!(ordinal>=0 && uint32_t(ordinal)<used);
        }
        uint32_t safeBefore=StagedPrefix(localSafe,ctl);
        uint32_t safeCount=ctl[544+512/warpSize-1];
        uint32_t conflictBefore=tid*per-safeBefore;
        for(uint32_t i=tid*per;i<(tid+1)*per && i<own;++i){
            uint32_t rank=begin+i,slot=slots[rank];
            if(slot>=resident)continue;
            int32_t ordinal=asc_ldcg(reserved+slot);
            bool conflict=ordinal>=0 && uint32_t(ordinal)<used;
            uint32_t position=conflict?safeCount+conflictBefore++:safeBefore++;
            uint32_t output=batch*kSlots+begin+position;
            uint32_t token=selected[ids[rank]];
            if(token>=history){asc_atomic_or(ctl+1,1U);continue;}
            StoreFallbackSource(history,tokenBytes,ctx,batch,output,token,asc_ldcg(addresses+token),blocks,offsets);
            asc_stcg(destinations+output,int32_t(ownerBase+slot));
        }
        asc_syncthreads();
        if(!tid){ctl[16+2*worker]=own;ctl[17+2*worker]=safeCount;}
        asc_syncthreads();
    }
    if(!tid && !ctl[1])asc_atomic_add(reinterpret_cast<__gm__ uint32_t*>(ctx->count),count);
    asc_threadfence();asc_syncthreads();
}

__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void PackStagedUrmaQuery(uint32_t history, uint32_t resident, uint32_t tokenBytes,
    uint32_t batch, uint32_t query, uint32_t used, uint32_t rangeBegin, uint32_t count,
    __gm__ Context* ctx, __ubuf__ uint32_t* selected, __ubuf__ uint16_t* ids,
    __ubuf__ uint32_t* slots, __ubuf__ uint32_t* ctl) {
    uint32_t tid=threadIdx.x;
    auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+batch;
    auto addresses=reinterpret_cast<__gm__ uint64_t*>(row->addresses);
    auto reserved=reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+batch*resident;
    auto blocks=reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks);
    auto offsets=reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets);
    auto destinations=reinterpret_cast<__gm__ int32_t*>(ctx->destinations);
    uint32_t ownerBase=OwnerRow(ctx,batch)*OutputStride(resident,ctx);
    for(uint32_t worker=0;worker<1;++worker){
        uint32_t begin=rangeBegin,end=begin+count;
        uint32_t own=end-begin,per=(own+511)/512,localSafe=0;
        for(uint32_t i=tid*per;i<(tid+1)*per && i<own;++i){
            uint32_t slot=slots[begin+i];
            if(slot>=resident){asc_atomic_or(ctl+1,4U);continue;}
            int32_t ordinal=asc_ldcg(reserved+slot);
            localSafe+=!(ordinal>=0 && uint32_t(ordinal)<used);
        }
        uint32_t safeBefore=StagedPrefix(localSafe,ctl);
        uint32_t safeCount=ctl[544+512/warpSize-1];
        uint32_t conflictBefore=tid*per-safeBefore;
        for(uint32_t i=tid*per;i<(tid+1)*per && i<own;++i){
            uint32_t rank=begin+i,slot=slots[rank];
            if(slot>=resident)continue;
            int32_t ordinal=asc_ldcg(reserved+slot);
            bool conflict=ordinal>=0 && uint32_t(ordinal)<used;
            uint32_t position=conflict?safeCount+conflictBefore++:safeBefore++;
            uint32_t output=batch*kSlots+query*kTop+position;
            uint32_t token=selected[ids[rank]];
            if(token>=history){asc_atomic_or(ctl+1,1U);continue;}
            StoreFallbackSource(history,tokenBytes,ctx,batch,output,token,asc_ldcg(addresses+token),blocks,offsets);
            asc_stcg(destinations+output,int32_t(ownerBase+slot));
        }
        asc_syncthreads();
        if(!tid){ctl[16+2*worker]=own;ctl[17+2*worker]=safeCount;}
        asc_syncthreads();
    }
    if(!tid && !ctl[1])asc_atomic_add(reinterpret_cast<__gm__ uint32_t*>(ctx->count),count);
    asc_threadfence();asc_syncthreads();
}
}
