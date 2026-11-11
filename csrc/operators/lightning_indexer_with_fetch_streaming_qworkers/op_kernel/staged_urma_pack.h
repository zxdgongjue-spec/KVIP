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

#if KVIP_FUSE_QUERY_WORK
__simt_callee__
#else
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
#endif
inline void PackStagedUrmaQuery(uint32_t history, uint32_t resident, uint32_t tokenBytes,
    uint32_t batch, uint32_t query, uint32_t used, uint32_t rangeBegin, uint32_t count,
    StagedContextPointer ctx, __ubuf__ uint32_t* selected, __ubuf__ uint16_t* ids,
    __ubuf__ uint32_t* slots, __ubuf__ uint32_t* ctl,
    __gm__ int32_t* direct=nullptr,uint32_t directStride=0) {
    uint32_t tid=threadIdx.x;
#if KVIP_FUSE_QUERY_WORK
    if(!count){
        if(!tid){ctl[16]=0;ctl[17]=0;ctl[18]=0;ctl[19]=0;}
        asc_syncthreads();
        return;
    }
#endif
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
        // Reserve one contiguous range per phase, not one atomic per token.
        if(direct){
            if(!tid){
                auto counters=reinterpret_cast<__gm__ uint32_t*>(direct+6*directStride);
                ctl[18]=safeCount?asc_atomic_add(counters,safeCount):0;
                ctl[19]=(own-safeCount)?asc_atomic_add(counters+8,own-safeCount):0;
                if(ctl[18]+safeCount>directStride || ctl[19]+own-safeCount>directStride)asc_atomic_or(ctl+1,16U);
            }
            asc_syncthreads();
        }
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
            if(direct && !ctl[1]){
                uint32_t pos=(conflict?ctl[19]+position-safeCount:ctl[18]+position);
                auto target=direct+(conflict?3*directStride:0);
                asc_stcg(target+pos,ctx->fallbackAddresses?int32_t(output):int32_t(batch*(history/128)+token/128));
                asc_stcg(target+directStride+pos,ctx->fallbackAddresses?0:int32_t((token%128)*tokenBytes));
                asc_stcg(target+2*directStride+pos,int32_t(ownerBase+slot));
            }
        }
        asc_syncthreads();
        if(!tid){ctl[16+2*worker]=own;ctl[17+2*worker]=safeCount;}
        asc_syncthreads();
    }
    if(!tid && !ctl[1])asc_atomic_add(reinterpret_cast<__gm__ uint32_t*>(ctx->count),count);
    asc_threadfence();asc_syncthreads();
}
}
