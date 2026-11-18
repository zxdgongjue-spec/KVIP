// SPDX-License-Identifier: Apache-2.0
#pragma once

// Entry requires this row's four CQ owners returned and all Index
// invalidations finished. Blocks from different rows have distinct owners.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void CommitJoinedRow(__gm__ Args* a,__gm__ Context* ctx,uint32_t b,
        uint32_t resident,uint32_t postOffset,uint32_t prefetched,uint32_t total,
        uint32_t completed,uint32_t joinedError,uint32_t error) {
    const uint32_t tid=threadIdx.x;
    auto bind=reinterpret_cast<__gm__ uint64_t*>(a->updateBindings);
    auto info=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*kRowStatusWords;
    auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
    auto posts=reinterpret_cast<__gm__ FinalPost*>(ctx->owners+postOffset)+4*b;
    auto status=reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(ctx->finalStatus)+2*b;
    if(error) {
        // CompleteEarlyUpdate's failure contract: no partial owner is usable.
        uint32_t owner=asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(bind[9])+b);
        uint32_t size=uint32_t(bind[25]);
        auto valid=reinterpret_cast<__gm__ uint32_t*>(bind[16])+uint64_t(owner)*size;
        for(uint32_t i=tid;i<size;i+=256)asc_stcg(valid+i,0U);
    } else if(ctx->hotValid) {
        auto valid=reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+uint64_t(OwnerRow(ctx,b))*resident;
        auto events=reinterpret_cast<__gm__ uint32_t*>(a->eventsDst)+uint64_t(b)*a->maxEvents;
        // Reserved-slot changes are the union of the committed prefetch prefix
        // and final destinations. Raw final destinations may have been reordered
        // for posting but mark precisely the same physical hot slots.
        for(uint32_t i=tid;i<prefetched+total;i+=256) {
            uint32_t pos=i<prefetched ? i : kPrefetch+(i-prefetched);
            uint32_t slot=asc_ldcg(events+pos);
            if(slot<resident)asc_stcg(valid+slot,1U);
        }
    }
    if(tid<4) {
        asc_stcg(&posts[tid].pending,0U);asc_stcg(&posts[tid].deferred,0U);
        asc_stcg(&posts[tid].error,joinedError);
    }
    if(tid<16) {
        // The native WriteStatus ABI has eight consecutive words per header;
        // the optional profile after those eight words is caller-owned.
        static_assert(offsetof(KvafBatchGetCoreStatus,profile)==8*sizeof(uint64_t),"KVAF status header changed");
        uint32_t field=tid%8;
        uint64_t word=field==0 ? KVAF_AIV_BATCH_GET_STATUS_MAGIC : field==6 ? UINT64_MAX : 0;
        if(tid==1)word=total;
        if(tid==2)word=error;
        if(tid==3)word=completed;
        asc_stcg(reinterpret_cast<__gm__ uint64_t*>(status+tid/8)+field,word);
    }
    if(!tid) {
        asc_stcg(reinterpret_cast<__gm__ uint32_t*>(bind[6])+b,(joinedError?1U:0U)|(error?4U:0U));
        asc_stcg(info,asc_ldcg(info)|error);
        asc_stcg(info+3,0U);asc_stcg(info+4,completed);
        if(!error)asc_stcg(&row->finalPending,0U);
    }
    asc_threadfence();
}
