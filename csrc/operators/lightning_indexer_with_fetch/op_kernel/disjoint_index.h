// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace pipeline_ab {

// Eligible rows have no speculative moves, cold-only missing sources, and
// resolver-deduplicated sources and destinations across all queries. Each
// cold source therefore exchanges with one unique hot destination. No other
// exchange can read either position, so a shadow permutation is unnecessary.
// Unique LRU candidates are an existing input contract: SlotLRU walks a unique
// slot permutation, and snapshot validation rejects duplicate candidates.
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS)
inline void ApplyDisjointIndex(
    __gm__ uint32_t* sources, __gm__ uint32_t* destinations,
    uint32_t begin, uint32_t count, uint32_t resident, uint32_t history,
    __gm__ uint64_t* keys, __gm__ uint32_t* scales, __ubuf__ uint32_t* work) {
    uint32_t tid=threadIdx.x;
    auto error=work+CONTROL,sourceRows=work+SRCSET,destRows=work+DSTSET;
    if(!tid)*error=count>SET?1U:0U;
    asc_syncthreads();
    if(*error)return;
    // Reuse dead membership scratch for compact row IDs. Read each GM
    // descriptor once rather than repeating it for all 16 key fields.
    for(uint32_t i=tid;i<count;i+=THREADS){
        uint32_t src=asc_ldcg(sources+begin+i),dst=asc_ldcg(destinations+begin+i);
        sourceRows[i]=src;destRows[i]=dst;
        if(src<resident || src>=history || dst>=resident)asc_atomic_or(error,1U);
    }
    asc_syncthreads();
    if(*error)return;
    for(uint32_t x=tid;x<count*16;x+=THREADS){
        uint32_t i=x/16,field=x%16;
        uint64_t src=uint64_t(sourceRows[i])*16+field;
        uint64_t dst=uint64_t(destRows[i])*16+field;
        uint64_t a=asc_ldcg(keys+src),b=asc_ldcg(keys+dst);
        asc_stcg(keys+src,b);asc_stcg(keys+dst,a);
    }
    for(uint32_t i=tid;i<count;i+=THREADS){
        uint32_t src=sourceRows[i],dst=destRows[i];
        uint32_t a=asc_ldcg(scales+src),b=asc_ldcg(scales+dst);
        asc_stcg(scales+src,b);asc_stcg(scales+dst,a);
    }
    asc_threadfence();asc_syncthreads();
}

// Descriptor publication needs the original address and logical-ID tables.
// Only the existing OutputWorker completion join permits these swaps.
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS)
inline void CommitDisjointMetadata(
    __gm__ uint32_t* sources, __gm__ uint32_t* destinations,
    uint32_t begin, uint32_t count,
    __gm__ uint64_t* addresses, __gm__ uint64_t* logical) {
    for(uint32_t i=threadIdx.x;i<count;i+=THREADS){
        uint32_t src=asc_ldcg(sources+begin+i),dst=asc_ldcg(destinations+begin+i);
        uint64_t a=asc_ldcg(addresses+src),b=asc_ldcg(addresses+dst);
        uint64_t x=asc_ldcg(logical+src),y=asc_ldcg(logical+dst);
        asc_stcg(addresses+src,b);asc_stcg(addresses+dst,a);
        asc_stcg(logical+src,y);asc_stcg(logical+dst,x);
    }
    asc_threadfence();asc_syncthreads();
}

}  // namespace pipeline_ab
