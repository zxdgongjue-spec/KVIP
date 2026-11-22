#if defined(__CCE_AICORE__)
__simt_callee__ inline void Emit(
 __ubuf__ uint32_t* ids,__ubuf__ uint32_t* loc,__ubuf__ uint32_t* error,
 __gm__ uint32_t* mapping,__gm__ uint32_t* lru,__gm__ uint64_t* logical,
 uint32_t history,uint32_t resident,uint32_t begin,uint32_t size,uint32_t capacity,__gm__ uint32_t* header,__gm__ uint32_t* progress,__gm__ uint32_t* flags,uint32_t generation,uint32_t producers){
 if(threadIdx.x<32){uint32_t mask;do{bool ready=threadIdx.x>=producers || asc_ldcg(flags+24*32+threadIdx.x)>=generation;mask=asc_ballot(ready);}while(mask!=UINT32_MAX); }
 asc_syncthreads();
 uint32_t i0=threadIdx.x,i1=threadIdx.x+1024;
 bool use0=loc[i0]==UINT32_MAX && ids[i0]!=UINT32_MAX;
 bool use1=loc[i1]==UINT32_MAX && ids[i1]!=UINT32_MAX;
 uint32_t t0=ids[i0]&0x3fffffffU,t1=ids[i1]&0x3fffffffU;
 uint32_t m0=use0?asc_ldcg(mapping+t0):UINT32_MAX;
 uint32_t m1=use1?asc_ldcg(mapping+t1):UINT32_MAX;
 uint32_t s0=m0<resident?m0:UINT32_MAX,s1=m1<resident?m1:UINT32_MAX;
 if(use0 && m0-resident<2048){s0=asc_ldcg(lru+m0-resident);t0=s0;}
 if(use1 && m1-resident<2048){s1=asc_ldcg(lru+m1-resident);t1=s1;}
 uint32_t l0=use0?asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(logical)+2*t0):0;
 uint32_t l1=use1?asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(logical)+2*t1):0;
 if(use0){if(s0==UINT32_MAX)asc_atomic_or(error,2U);loc[i0]=s0;ids[i0]=l0;}
 if(use1){if(s1==UINT32_MAX)asc_atomic_or(error,2U);loc[i1]=s1;ids[i1]=l1;}
 asc_threadfence();
 // VF completion is the next consumer boundary; no work follows in this task.
}
#endif
