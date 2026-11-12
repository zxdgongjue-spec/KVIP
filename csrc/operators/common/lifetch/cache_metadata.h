// Shared LIFetch slot metadata helpers; included within namespace rotating.
constexpr uint32_t kLruSlots = 2048;
constexpr uint32_t kChangeCapacity = kLruSlots + kSlots;
#if KVIP_FUSE_QUERY_WORK
template<typename ContextPointer>
__simt_callee__ inline uint32_t OwnerRow(ContextPointer ctx,uint32_t b)
#else
__simt_callee__ inline uint32_t OwnerRow(__gm__ Context* ctx,uint32_t b)
#endif
{
    return ctx->rowOwners ? asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)+b) : b;
}
#if KVIP_FUSE_QUERY_WORK
template<typename ContextPointer>
__simt_callee__ inline uint32_t OutputStride(uint32_t rt_resident,ContextPointer ctx)
#else
__simt_callee__ inline uint32_t OutputStride(uint32_t rt_resident,__gm__ Context* ctx)
#endif
{
    return ctx->outputStride ? uint32_t(ctx->outputStride) : rt_resident;
}
__simt_callee__ inline bool IsTailToken(__gm__ Context* ctx,uint32_t b,uint32_t token) {
    if(!ctx->tailBounds)return false;
    auto bounds=reinterpret_cast<__gm__ uint32_t*>(ctx->tailBounds)+b*2;
    return token>=asc_ldcg(bounds) && token<asc_ldcg(bounds+1);
}
__simt_callee__ inline uint32_t QueryEnd(uint32_t rt_history,__gm__ Context* ctx,uint32_t query) {
    return ctx->queryEnds ? asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds)+query) : rt_history;
}
#if KVIP_FUSE_QUERY_WORK
template<typename ContextPointer>
__simt_callee__ inline void StoreFallbackSource(uint32_t rt_history,uint32_t rt_tokenBytes,ContextPointer ctx,uint32_t batch,uint32_t dst,
#else
__simt_callee__ inline void StoreFallbackSource(uint32_t rt_history,uint32_t rt_tokenBytes,__gm__ Context* ctx,uint32_t batch,uint32_t dst,
#endif
    uint32_t token,uint64_t remote,__gm__ int32_t* blocks,__gm__ int32_t* offsets) {
    if(ctx->fallbackAddresses) {
        asc_stcg(reinterpret_cast<__gm__ uint64_t*>(ctx->fallbackAddresses)+dst,remote);
        asc_stcg(blocks+dst,int32_t(dst));asc_stcg(offsets+dst,int32_t(0));
    } else {
        asc_stcg(blocks+dst,int32_t(batch*(rt_history/128)+token/128));
        asc_stcg(offsets+dst,int32_t((token%128)*rt_tokenBytes));
    }
}
__simt_callee__ inline bool HotValid(uint32_t rt_resident,__gm__ Context* ctx,uint32_t b,uint32_t token) {
    return !ctx->hotValid || asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+OwnerRow(ctx,b)*rt_resident+token)!=0;
}
__simt_callee__ inline uint32_t PhysicalSlot(uint32_t rt_resident,__gm__ Context* ctx,uint32_t b,uint32_t slot) {
    if(ctx->lruSlots && uint32_t(slot-rt_resident)<kLruSlots)
        return asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+b*kLruSlots+slot-rt_resident);
    return slot;
}
__simt_callee__ inline void RecordChange(uint32_t rt_resident,__gm__ Context* ctx,uint32_t b,uint32_t slot,uint32_t token) {
    if(ctx->changes && uint32_t(slot-rt_resident)<kLruSlots)
        asc_stcg(reinterpret_cast<__gm__ uint32_t*>(ctx->changes)+b*kChangeCapacity+slot-rt_resident,token);

}

// Shared residency publication; each transport chooses its safe completion point.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void PublishDestinationResidency(uint32_t rt_resident,__gm__ Context* ctx,uint32_t b,uint32_t fetched) {
    if(!ctx->hotValid)return;
    auto valid=reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+OwnerRow(ctx,b)*rt_resident;
    // The original resolver already emits exactly the final Fetch destinations.
    // Hits need no publication; do not rescan all four TopK rows.
    auto destinations=reinterpret_cast<__gm__ int32_t*>(ctx->destinations)+b*kSlots;
    uint32_t ownerBase=OwnerRow(ctx,b)*OutputStride(rt_resident,ctx);
    for(uint32_t i=threadIdx.x;i<fetched;i+=256) {
        uint32_t slot=uint32_t(asc_ldcg(destinations+i))-ownerBase;
        if(slot<rt_resident)asc_stcg(valid+slot,uint32_t(1));
    }
    if(ctx->changes && ctx->lruSlots) {
        auto changes=reinterpret_cast<__gm__ int32_t*>(ctx->changes)+b*kChangeCapacity;
        auto lru=reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+b*kLruSlots;
        for(uint32_t i=threadIdx.x;i<kLruSlots;i+=256)
            if(asc_ldcg(changes+i)>=0)asc_stcg(valid+asc_ldcg(lru+i),uint32_t(1));
    }
    asc_threadfence();
}

// Shared LIFetch hot/cold predicate and physical slot resolution.
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void AdaptColdPredicate(uint32_t rt_resident,__gm__ Context* ctx,uint32_t b,__ubuf__ uint32_t* selected,__ubuf__ uint32_t* predicate) {
    auto reserved=reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+b*rt_resident;
    for(uint32_t i=threadIdx.x;i<kSlots;i+=512) {
        uint32_t token=selected[i];
        predicate[i]=(token<rt_resident &&
            (IsTailToken(ctx,b,token) || (!HotValid(rt_resident,ctx,b,token)) || (ctx->reservedHot && asc_ldcg(reserved+token)>=0)))?rt_resident:token;
    }
}
// Bulk metadata is staged once in UB by the resolver, not fetched per TopK item.
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void AdaptColdPredicateLocal(uint32_t rt_resident,__ubuf__ uint32_t* selected,__ubuf__ uint32_t* predicate,
    __ubuf__ uint32_t* valid,__ubuf__ int32_t* reserved,bool hasValid,bool hasReserved,
    uint32_t tailBegin,uint32_t tailEnd) {
    for(uint32_t i=threadIdx.x;i<kSlots;i+=512) {
        uint32_t token=selected[i];
        predicate[i]=(token<rt_resident && ((token>=tailBegin && token<tailEnd) ||
            (hasValid && !valid[token]) || (hasReserved && reserved[token]>=0)))?rt_resident:token;
    }
}
__simt_callee__ inline uint32_t ResolvePhysicalSlot(uint32_t rt_resident,__gm__ uint32_t* lru,uint32_t slot) {
    if(lru && uint32_t(slot-rt_resident)<kLruSlots)return asc_ldcg(lru+slot-rt_resident);
    // Missing-rank markers have their high bit set and are not ABC slots.
    return slot>=rt_resident+kSlots && slot<0x80000000U ? slot-kSlots : slot;
}
