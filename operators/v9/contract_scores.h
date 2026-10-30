#pragma once
#include "rotation_device.h"
#include "contract_device.h"
#include "contract_parallel.h"
namespace rotating {
__simt_callee__ inline uint16_t LoadContractScore(const __gm__ uint16_t* address) {
    // CANN 9.1 asc_ldcg takes a mutable pointer even though it only loads.
    return asc_ldcg(const_cast<__gm__ uint16_t*>(address));
}
// Keep per-tile payloads contiguous for the native MTE3 pipeline. Only
// causal masking depends on logical IDs; map only the final TopK IDs.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void MaskContractScores(__gm__ Context* ctx, uint32_t b, uint32_t q0,
    uint32_t nq, uint32_t start, uint32_t count, bool hot,
    __ubuf__ uint16_t* values, uint32_t stride, __ubuf__ uint32_t* predicate) {
    auto c=contract_simt::ContractOf(ctx);
    uint32_t length=contract_simt::ContractLength(c,b);
    for(uint32_t j=threadIdx.x;j<nq*128;j+=256) {
        uint32_t q=j/128,lane=j%128,position=start+lane;
        bool real=lane<count && (hot || position<contract_simt::ColdCount(c,b));
        int32_t token=real?contract_simt::IndexIds(c,b,hot)[position]:int32_t(length);
        if(!real || int64_t(token)>=int64_t(length)-4+q0+q+1)values[q*stride+lane]=0;
    }
    if(!hot)for(uint32_t j=threadIdx.x;j<nq*8;j+=256) {
        uint32_t q=j/8,word=j%8,bits=predicate[q*64+word];
        for(uint32_t k=0;k<16;++k) {
            uint32_t lane=word*16+k,position=start+lane;
            bool real=lane<count && position<contract_simt::ColdCount(c,b);
            int32_t token=real?contract_simt::IndexIds(c,b,false)[position]:int32_t(length);
            if(!real || int64_t(token)>=int64_t(length)-4+q0+q+1)bits&=~(3U<<(k*2));
        }
        predicate[q*64+word]=bits;
    }
}
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void ContractThreshold(__gm__ Context* ctx,uint32_t b,uint32_t q,
    __ubuf__ uint32_t* scratch,__ubuf__ uint16_t* threshold,
    const __gm__ uint16_t* compact,uint32_t count) {
    auto c=contract_simt::ContractOf(ctx); uint32_t tid=threadIdx.x;
    uint32_t length=contract_simt::ContractLength(c,b),visible=count;
    for(uint32_t t=length>3-q?length-(3-q):0;t<length;++t)
        visible-=contract_simt::ContractHot(c,b,t);
    // Native MTE3 -> MTE2 TopK join has completed the contiguous hot scores.
    asc_threadfence(); asc_syncthreads();
    if(!tid) {
        if(visible<2048)threshold[0]=0;
        asc_stcg(contract_simt::ContractPtr<int32_t>(c->threshold_valid)+b*4+q,int32_t(visible>=2048));
        asc_stcg(contract_simt::ContractPtr<int32_t>(c->hot_thresholds)+b*4+q,int32_t(threshold[0]));
        asc_threadfence();
        scratch[0]=asc_atomic_add(contract_simt::ContractPtr<int32_t>(c->hot_ready)+b*32,1)==3;
    }
    asc_syncthreads();
    if(!scratch[0])return;
    asc_syncthreads(); // all lanes consumed the decision before prefix reuses scratch
    uint32_t pool=contract_simt::ContractPool(c,b),keep=0,mask=0;
    auto forward=contract_simt::ContractPtr<int32_t>(c->forward)+uint64_t(pool)*c->candidate_capacity;
    auto reverse=contract_simt::ContractPtr<int32_t>(c->reverse)+uint64_t(pool)*c->resident_capacity;
    for(uint32_t j=0;j<8;++j) {
        uint32_t i=tid*8+j;
        int32_t slot=contract_simt::ContractPtr<int32_t>(c->victims)[b*2048+i];
        int32_t t=asc_ldcg(reverse+slot); bool protect=false;
        // Preserve the valid-owner check: stale reverse aliases are reusable.
        if(t>=0 && uint32_t(t)<length && asc_ldcg(forward+t)==slot && contract_simt::ContractHot(c,b,uint32_t(t)))
            for(uint32_t query=0;query<4;++query) {
                bool visible=int64_t(t)<int64_t(length)-4+query+1;
                uint16_t score=LoadContractScore(contract_simt::ContractPtr<uint16_t>(c->scores)+(b*4+query)*c->score_stride+uint32_t(-contract_simt::IndexPositions(c,b)[t]-2));
                protect|=visible && score>=asc_ldcg(contract_simt::ContractPtr<int32_t>(c->hot_thresholds)+b*4+query);
            }
        if(!protect){mask|=1U<<j;++keep;}
    }
    auto prefix=ContractPrefixUB(keep,scratch);uint32_t at=prefix.begin;
    for(uint32_t j=0;j<8;++j)if(mask&(1U<<j))
        asc_stcg(contract_simt::ContractPtr<int32_t>(c->eligible_victims)+b*2048+at++,
            contract_simt::ContractPtr<int32_t>(c->victims)[b*2048+tid*8+j]);
    asc_threadfence();asc_syncthreads();
    if(!tid) {
        asc_stcg(contract_simt::ContractPtr<int32_t>(c->eligible_counts)+b,int32_t(prefix.total));
        asc_threadfence();asc_stcg(contract_simt::ContractPtr<int32_t>(c->hot_ready)+b*32+1,1);
        asc_threadfence();
    }
}
// Native BF16 sortable zero is 0x8000 (padding is 0). Replace only invalid
// or repeated IDs, preserving every strictly-better element of native TopK.
__simt_callee__ inline void RepairContractTopK(__ubuf__ uint32_t* indices,
    __ubuf__ uint32_t* hash, const __gm__ uint16_t* scores, uint32_t valid, uint32_t visible = UINT32_MAX) {
    uint32_t tid=threadIdx.x;
    for(uint32_t i=tid;i<4096;i+=256)hash[i]=UINT32_MAX;
    asc_syncthreads();
    for(uint32_t i=tid;i<2048;i+=256) {
        uint32_t t=indices[i];
        if(t>=valid || LoadContractScore(scores+t)==0){indices[i]=UINT32_MAX;continue;}
        uint32_t h=(t*0x9e3779b1U)&4095;
        while(true) {
            uint32_t old=asc_atomic_cas(hash+h,UINT32_MAX,t);
            if(old==UINT32_MAX)break;
            if(old==t){indices[i]=UINT32_MAX;break;}
            h=(h+1)&4095;
        }
    }
    asc_syncthreads();
    uint32_t have=0;
    for(uint32_t i=tid;i<2048;i+=256)have+=indices[i]!=UINT32_MAX;
    // Native TopK shared scratch exceeds 4104 words for both Q2 and Q4.
    have=ContractPrefixUB(have,hash+4096).total;
    if(visible==UINT32_MAX)visible=valid;
    uint32_t wanted=visible<2048?visible:2048;
    if(tid || have==wanted)return;
    uint32_t next=0;
    for (uint32_t i = 0; i < 2048 && have < wanted; ++i) {
        if (indices[i] != UINT32_MAX) continue;
        while (next < valid) {
            uint32_t t = next++, h = (t * 0x9e3779b1U) & 4095;
            while (hash[h] != UINT32_MAX && hash[h] != t) h = (h + 1) & 4095;
            if (hash[h] != t && (LoadContractScore(scores+t)!=0 && (visible < 2048 || LoadContractScore(scores + t) == uint16_t(0x8000)))) {
                hash[h] = t; indices[i] = t; ++have; break;
            }
        }
    }
}
// Physical rows: [hot, zero padding to 128, cold]. Only 2048 selected IDs
// cross the physical/logical boundary; scores never undergo an O(C) scatter.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void FinalizeContractTopK(__gm__ Context* ctx,uint32_t b,uint32_t q,
    __ubuf__ uint32_t* indices,__ubuf__ uint32_t* hash,
    const __gm__ uint16_t* scores,uint32_t extent) {
    auto c=contract_simt::ContractOf(ctx);
    uint32_t nh=contract_simt::ContractPtr<int32_t>(c->hot_counts)[b];
    uint32_t base=(nh+127)/128*128;
    uint32_t length=contract_simt::ContractLength(c,b);
    uint32_t visible=length>3-q?length-(3-q):0;
    RepairContractTopK(indices,hash,scores,extent,visible);
    asc_syncthreads();
    for(uint32_t i=threadIdx.x;i<2048;i+=256) {
        uint32_t p=indices[i];
        if(p==UINT32_MAX)continue;
        indices[i]=uint32_t(contract_simt::IndexIds(c,b,p<nh)[p<nh?p:p-base]);
    }
}

}
