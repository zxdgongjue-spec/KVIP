// SPDX-License-Identifier: Apache-2.0
#pragma once

// Zero-window LI has already written every hit's location. Deduplicate misses
// across Q4, protect all selected resident slots, and emit final Get buffers at
// the point of allocation. A dense input differs only in how misses are read.
namespace adaptive_low {

constexpr uint32_t kThreads = 512;
constexpr uint32_t kBucket = 0x80000000U;
constexpr uint32_t kFirst = 0x40000000U;
constexpr uint32_t kMask = kSlots - 1;
static_assert(kSlots == 8192 && kTop == 2048, "Q4 rank encoding");

struct Group {
    __gm__ int32_t* descriptors;
    __gm__ uint64_t* addresses;
    __gm__ uint32_t* counters;
};

__simt_callee__ inline Group FetchGroup(__ubuf__ Args* a, uint32_t batch,
                                        uint32_t b, uint32_t q) {
    const uint32_t workers = (a->prefetchCount >> 16) & 0xffU;
    const uint32_t w = q*workers + b%workers, stride = batch*kTop;
    const uint32_t blockWords = 3*stride + ((3*batch+31)&~31U);
    auto base = reinterpret_cast<__gm__ int32_t*>(a->packedFetch);
    auto descriptors = base + 4*workers*blockWords + w*(6*stride+64);
    return {descriptors, reinterpret_cast<__gm__ uint64_t*>(base+w*blockWords),
            reinterpret_cast<__gm__ uint32_t*>(descriptors+6*stride)};
}

// Contiguous request-local ranks preserve TopK order. Holes are selected in
// LRU order first, then in resident-slot order when the candidate list is full.
__simt_callee__ inline void AllocateSlots(uint32_t resident, uint32_t needed,
    __ubuf__ uint32_t* slots, __ubuf__ uint32_t* scratch, __ubuf__ uint32_t* ctl) {
    if (!needed) return;
    const uint32_t tid = threadIdx.x, per = kLruSlots/kThreads;
    auto bitmap = ctl+1024, lru = scratch+4096;
    uint32_t available = 0;
    for (uint32_t i=tid*per; i<(tid+1)*per; ++i) {
        const uint32_t slot = lru[i];
        if (slot >= resident) { asc_atomic_or(ctl+1,4U); continue; }
        available += !(bitmap[slot/32] & (1U<<(slot%32)));
    }
    uint32_t rank = StagedPrefix(available,ctl);
    if (ctl[1]) return;
    const uint32_t n = ctl[544+kThreads/warpSize-1];
    const uint32_t allocated = n < needed ? n : needed;
    for (uint32_t i=tid*per; i<(tid+1)*per; ++i) {
        const uint32_t slot = lru[i];
        if (!(bitmap[slot/32] & (1U<<(slot%32)))) {
            if (rank < allocated) slots[rank] = slot;
            ++rank;
        }
    }
    asc_syncthreads();
    if (allocated == needed) return;
    for (uint32_t i=tid; i<allocated; i+=kThreads)
        asc_atomic_or(bitmap+slots[i]/32,1U<<(slots[i]%32));
    asc_syncthreads();
    const uint32_t words = (resident+31)/32, perThread = (words+kThreads-1)/kThreads;
    auto counts = reinterpret_cast<__ubuf__ int32_t*>(scratch), offsets = counts+words;
    kvip_hot_slots::CountAvailableBitmapWordsBody(bitmap,counts,resident,0,words,0);
    asc_syncthreads();
    uint32_t local = 0;
    for (uint32_t i=tid*perThread; i<(tid+1)*perThread && i<words; ++i) local += counts[i];
    uint32_t offset = StagedPrefix(local,ctl);
    if (!tid && ctl[544+kThreads/warpSize-1] < needed-allocated) ctl[1] = 4;
    for (uint32_t i=tid*perThread; i<(tid+1)*perThread && i<words; ++i) {
        offsets[i] = offset;
        offset += counts[i];
    }
    asc_syncthreads();
    if (ctl[1]) return;
    kvip_hot_slots::WriteAvailableBitmapSlotsBody(bitmap,counts,offsets,
        reinterpret_cast<__ubuf__ int32_t*>(slots),resident,0,words,0,needed-allocated,allocated);
    asc_syncthreads();
}

template<bool Compact>
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void Resolve(__ubuf__ Args* a, StagedContextPointer ctx, uint32_t b,
    __ubuf__ uint32_t* selected, __ubuf__ uint32_t* loc, __ubuf__ uint32_t* keys,
    __ubuf__ uint32_t* ranks, __ubuf__ uint32_t* slots, __ubuf__ uint16_t* positions,
    __ubuf__ uint32_t* scratch, __ubuf__ uint32_t* ctl) {
    const uint32_t tid=threadIdx.x, history=ctx->shape.history, resident=ctx->shape.resident;
    constexpr uint32_t span = Compact ? kThreads : kSlots, per=span/kThreads, qWidth=span/4;
    const uint32_t words=(resident+31)/32, buckets=Compact ? 512 : kSlots;
    auto bitmap=ctl+1024;
    auto protection=reinterpret_cast<__gm__ uint32_t*>(a->queryBitmaps)+uint64_t(b)*4*words;
    auto blocks=reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks);
    auto locations=reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*kSlots;
    // Zero mode has no prefetch events. Invalidate the public LRU change
    // journal here instead of dispatching a separate empty prefetch VF.
    auto changes=reinterpret_cast<__gm__ uint32_t*>(ctx->changes)+b*kChangeCapacity;
    for (uint32_t i=tid; i<kLruSlots; i+=kThreads) asc_stcg(changes+i,UINT32_MAX);
    for (uint32_t i=tid; i<words; i+=kThreads)
        bitmap[i]=asc_ldcg(protection+i)|asc_ldcg(protection+words+i)|
                  asc_ldcg(protection+2*words+i)|asc_ldcg(protection+3*words+i);
    if constexpr (Compact) {
        const uint32_t q=tid/qWidth, pos=tid%qWidth;
        auto header=reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)+(b*4+q)*32;
        uint32_t token=UINT32_MAX;
        if (pos < asc_ldcg(header+1)-1) {
            uint32_t entry=uint32_t(asc_ldcg(blocks+(b*4+q)*kTop+pos));
            positions[tid]=entry&kPositionMask;
            token=entry>>kPositionBits;
        }
        selected[tid]=token;
        loc[tid]=UINT32_MAX;
    }
    asc_threadfence(); asc_syncthreads();
    for (uint32_t i=tid; i<span; i+=kThreads) {
        uint32_t token=selected[i];
        if (token==UINT32_MAX || loc[i]!=UINT32_MAX) continue;
        if (token>=history) { asc_atomic_or(ctl+1,1U); continue; }
        // An invalid resident hit is refilled into its own slot. An unrelated
        // cold miss must never evict it, even if the slot occurs in the LRU.
        if (token<resident) asc_atomic_or(bitmap+token/32,1U<<(token%32));
        uint32_t h=SparseHash(token)&(buckets-1), probe=0;
        for (; probe<buckets; ++probe) {
            uint32_t prev=asc_atomic_cas(keys+h,UINT32_MAX,token);
            if (prev==UINT32_MAX || prev==token) {
                asc_atomic_min(ranks+h,i); loc[i]=kBucket|h; break;
            }
            h=(h+1)&(buckets-1);
        }
        if (probe==buckets) asc_atomic_or(ctl+1,8U);
    }
    asc_syncthreads();
    if (ctl[1]) return;

    uint32_t unique=0, victims=0;
    for (uint32_t i=tid*per; i<(tid+1)*per; ++i) {
        if (selected[i]==UINT32_MAX || !(loc[i]&kBucket)) continue;
        if (ranks[loc[i]&kMask]==i) {
            loc[i]|=kFirst; ++unique; victims+=selected[i]>=resident;
        }
    }
    uint32_t prefix=StagedPrefix(unique+(victims<<16),ctl);
    const uint32_t total=ctl[544+kThreads/warpSize-1], needed=total&0xffffU;
    uint32_t ordinal=prefix&0xffffU, victim=prefix>>16;
    // ctl[24..27] and [28..31] hold the row-local and group-wide Q offsets.
    if (tid<4) {
        constexpr uint32_t warps=kThreads/(4*warpSize);
        const uint32_t start=tid ? ctl[544+tid*warps-1]&0xffffU : 0;
        ctl[24+tid]=start;
        ctl[2+tid]=(ctl[544+(tid+1)*warps-1]&0xffffU)-start;
    }
    if (!tid) ctl[0]=needed;
    for (uint32_t i=tid*per; i<(tid+1)*per; ++i) {
        if (selected[i]==UINT32_MAX || !(loc[i]&kFirst)) continue;
        ranks[loc[i]&kMask]=(ordinal++<<13)|victim;
        victim+=selected[i]>=resident;
    }
    asc_syncthreads();
    AllocateSlots(resident,total>>16,slots,scratch,ctl);
    if (ctl[1]) return;

    const uint32_t batch=ctx->shape.batch, stride=batch*kTop;
    auto inverse=reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+b*resident;
    auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
    auto changeSlots=reinterpret_cast<__gm__ uint32_t*>(ctx->changeSlots)+b*kChangeCapacity;
    auto eventSrc=reinterpret_cast<__gm__ uint32_t*>(a->eventsSrc)+b*a->maxEvents+kPrefetch;
    auto eventDst=reinterpret_cast<__gm__ uint32_t*>(a->eventsDst)+b*a->maxEvents+kPrefetch;
    const uint32_t ownerBase=OwnerRow(ctx,b)*OutputStride(resident,ctx);
    if (tid<4) {
        Group g=FetchGroup(a,batch,b,tid);
        uint32_t n=ctl[2+tid], offset=n ? asc_atomic_add(g.counters,n) : 0;
        // This flag must be distinct from the completed allocator error:
        // other warps may still be checking ctl[1] before joining us.
        if (offset>stride || n>stride-offset) asc_atomic_or(ctl+14,16U);
        ctl[28+tid]=offset;
    }
    asc_syncthreads();
    if (ctl[14]) {
        if (!tid) ctl[1]=16U;
        return;
    }
    for (uint32_t i=tid; i<span; i+=kThreads) {
        const uint32_t token=selected[i], handle=loc[i];
        if (token==UINT32_MAX || !(handle&kBucket)) continue;
        const uint32_t pair=ranks[handle&kMask], rank=pair>>13;
        const uint32_t slot=token<resident ? token : slots[pair&kMask];
        const uint32_t q=i/qWidth, inQ=rank-ctl[24+q];
        const uint32_t topPosition=Compact ? q*kTop+positions[i] : i;
        asc_stcg(locations+topPosition,slot);
        if (!(handle&kFirst)) continue;
        int32_t oldOrdinal=asc_ldcg(inverse+slot);
        uint32_t record=kLruSlots+rank;
        if (oldOrdinal>=0) {
            if (uint32_t(oldOrdinal)>=kLruSlots) {asc_atomic_or(ctl+1,4U);continue;}
            asc_stcg(changes+record,UINT32_MAX);
            record=uint32_t(oldOrdinal);
        }
        asc_stcg(changes+record,token); asc_stcg(changeSlots+record,slot);
        asc_stcg(eventSrc+rank,token); asc_stcg(eventDst+rank,slot);
        Group g=FetchGroup(a,batch,b,q);
        uint32_t pos=ctl[28+q]+inQ, out=(b*4+q)*kTop+inQ;
        uint64_t address=asc_ldcg(reinterpret_cast<__gm__ uint64_t*>(row->addresses)+token);
        // The per-row view is the public frozen descriptor used for replay;
        // Join also uses its destinations to publish hotValid after successful Get.
        StoreFallbackSource(history,ctx->shape.tokenBytes,ctx,b,out,token,address,blocks,
                            reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets));
        asc_stcg(reinterpret_cast<__gm__ int32_t*>(ctx->destinations)+out,int32_t(ownerBase+slot));
        asc_stcg(g.addresses+pos,address);
        asc_stcg(g.descriptors+pos,int32_t(pos));
        asc_stcg(g.descriptors+stride+pos,0);
        asc_stcg(g.descriptors+2*stride+pos,int32_t(ownerBase+slot));
    }
    asc_threadfence(); asc_syncthreads();
    if (ctl[1]) return;
    if (tid<32) {
        const uint32_t q=tid/8, field=tid%8, count=ctl[2+q];
        auto posts=reinterpret_cast<__gm__ FinalPost*>(ctx->owners+ctx->shape.finalPostOffset);
        auto post=reinterpret_cast<__gm__ uint32_t*>(posts+b*4+q);
        uint32_t v=field==0 || field==5 ? count : field==3 ? uint32_t(count!=0) :
                   field==4 ? (b*4+q)*kTop : 0;
        asc_stcg(post+field,v);
    }
    auto flags=reinterpret_cast<__gm__ BatchFlag*>(a->flags)+b*kFlagCount+kRawQueryFlag;
    if (tid<4) {
        auto status=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*32;
        asc_stcg(status+4+tid,ctl[2+tid]); asc_stcg(status+8+tid,ctl[24+tid]+ctl[2+tid]);
        asc_stcg(&flags[tid].reserved[0],1U); asc_stcg(&flags[tid].count,ctl[2+tid]);
        asc_stcg(&flags[tid].error,0U);
    }
    if (!tid) {
        asc_stcg(reinterpret_cast<__gm__ uint32_t*>(ctx->changeCount)+b,kLruSlots+needed);
        if (needed) asc_atomic_add(reinterpret_cast<__gm__ uint32_t*>(a->liveCount?a->liveCount:ctx->count),needed);
    }
    asc_threadfence(); asc_syncthreads();
    if (tid<4) {
        Group g=FetchGroup(a,batch,b,tid);
        asc_atomic_add(g.counters+kGroupReadyWord,1U);
        if (tid<3) asc_stcg(&flags[tid].ready,1U);
    }
    // Index/outputs acquire Q4 to consume the entire row.
    asc_threadfence(); asc_syncthreads();
    if (!tid) asc_stcg(&flags[3].ready,1U);
    asc_threadfence();
}

} // namespace adaptive_low
