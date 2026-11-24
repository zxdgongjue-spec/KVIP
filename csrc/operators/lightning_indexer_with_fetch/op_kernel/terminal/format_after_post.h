#pragma once
// Called after the four owners have published their mappings. Each lane
// resolves four independent output entries. Preserve invalid entries and
// the exact selected-token, public logical ID, physical slot and error ABI.
__simt_callee__ inline void FormatAfterPost(
    __ubuf__ uint32_t* ids,__ubuf__ uint32_t* slots,__ubuf__ uint32_t* error,
    __gm__ uint32_t* mapping,__gm__ uint32_t* lru,__gm__ uint64_t* logical,
    __gm__ uint32_t* selected,uint32_t history,uint32_t resident,
    uint32_t tailBegin,uint32_t tailEnd,uint32_t tailCapacity) {
    uint32_t token[4],direct[4],lookup[4],logicalId[4],invalid=0;
    bool live[4];
    #pragma unroll
    for(uint32_t j=0;j<4;++j){
        const uint32_t i=threadIdx.x+j*512,tag=ids[i],t=tag&0x3fffffffU;
        token[j]=t;live[j]=tag!=UINT32_MAX && t<history;
        if(tag!=UINT32_MAX && t>=history)invalid|=1U;
        asc_stcg(selected+i,tag==UINT32_MAX?UINT32_MAX:t);
        uint32_t slot=UINT32_MAX;
        if((tag&0x80000000U) && t<resident)slot=t;
        else slot=rotating::DirectTailSlot(t,resident,tailBegin,tailEnd,tailCapacity);
        direct[j]=slot;
    }
    #pragma unroll
    for(uint32_t j=0;j<4;++j){
        const uint32_t t=live[j]?token[j]:0;
        logicalId[j]=live[j]?asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(logical)+2*t):UINT32_MAX;
        lookup[j]=(live[j] && direct[j]==UINT32_MAX)?asc_ldcg(mapping+t):UINT32_MAX;
    }
    #pragma unroll
    for(uint32_t j=0;j<4;++j){
        uint32_t s=direct[j],m=lookup[j];
        if(live[j] && s==UINT32_MAX){
            if(m<resident)s=m;
            else if(m-resident<2048)s=asc_ldcg(lru+(m-resident));
            if(s==UINT32_MAX)invalid|=2U;
        }
        const uint32_t i=threadIdx.x+j*512;
        ids[i]=live[j]?logicalId[j]:UINT32_MAX;
        slots[i]=live[j]?s:UINT32_MAX;
    }
    if(invalid)asc_atomic_or(error,invalid);
    asc_threadfence();
}

__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void FinishOutputAfterPost(
    __ubuf__ uint32_t* ids,__ubuf__ uint32_t* slots,__ubuf__ uint32_t* control,
    __gm__ uint32_t* mapping,__gm__ uint32_t* lru,__gm__ uint64_t* logical,
    __gm__ uint32_t* selected,uint32_t history,uint32_t resident,
    uint32_t tailBegin,uint32_t tailEnd,uint32_t tailCapacity,
    __gm__ uint32_t* flags,uint32_t generation,__gm__ rotating::Row* row){
    uint32_t t=threadIdx.x;
    if(t<32){
        uint32_t ready;
        do{ready=asc_ballot(t>=4 || asc_ldcg(flags+24*32+t)>=generation);}while(ready!=UINT32_MAX);
        if(!t){control[0]=0;control[1]=asc_ldcg(&row->slots);}
    }
    asc_syncthreads();
    FormatAfterPost(ids,slots,control,mapping,lru,logical,selected,
                           history,resident,tailBegin,tailEnd,tailCapacity);
}
