// Shared prefetch-mapping, final miss planning and slot publication.
    uint32_t tid=threadIdx.x,lane=tid%warpSize,warp=tid/warpSize;
    auto lru=ctx->lruSlots ? reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+batch*kLruSlots : nullptr;
    uint32_t outputStride=OutputStride(KV_RESOLVE_RESIDENT,ctx),ownerBase=OwnerRow(ctx,batch)*outputStride;
    uint32_t count=0,candidateCount=0;
    if(!ResolveResidentPlan<Threads>(KV_RESOLVE_HISTORY,KV_RESOLVE_RESIDENT,outputStride,selected,loc,
        positions,coldCount,keys,ranks,ids,slots,cfg,mapping,lru,used,ctx->tailBounds!=0,count,candidateCount,ctx,batch))return;
    constexpr uint32_t missingTag=0x80000000U;
    uint32_t ownBegin=count*worker/KV_RESOLVE_WORKERS,ownEnd=count*(worker+1)/KV_RESOLVE_WORKERS;
    if(!tid){cfg[2]=ownBegin;cfg[3]=ownEnd-ownBegin;if(!worker)asc_atomic_add(total,count);}
    for(uint32_t i=tid;i<coldCount;i+=Threads) {
        uint32_t pos=positions[i],slot=loc[pos];
        if(pos/(kSlots/KV_RESOLVE_WORKERS)==worker && (slot&missingTag))loc[pos]=slots[slot&~missingTag];
    }
    asc_syncthreads();
