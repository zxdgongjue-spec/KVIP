#if defined(__CCE_AICORE__)
__simt_callee__ inline void Emit(
 __ubuf__ uint32_t* ids,__ubuf__ uint32_t* loc,__ubuf__ uint32_t* error,
 __gm__ uint32_t* mapping,__gm__ uint32_t* lru,__gm__ uint64_t* logical,
 uint32_t history,uint32_t resident,uint32_t begin,uint32_t size,uint32_t capacity,__gm__ uint32_t* header,__gm__ uint32_t* progress,__gm__ uint32_t* flags,uint32_t generation,uint32_t producers){
 bool use[4];uint32_t token[4],logicalId[4];
 #pragma unroll
 for(uint32_t part=0;part<4;++part){
  uint32_t i=threadIdx.x+part*512;
  use[part]=loc[i]==UINT32_MAX && ids[i]!=UINT32_MAX;
  token[part]=ids[i]&0x3fffffffU;
  logicalId[part]=use[part]?asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(logical)+2*token[part]):0;
 }
 if(threadIdx.x<32){uint32_t mask;do{bool ready=threadIdx.x>=producers || asc_ldcg(flags+24*32+threadIdx.x)>=generation;mask=asc_ballot(ready);}while(mask!=UINT32_MAX);}
 asc_syncthreads();
 #pragma unroll
 for(uint32_t part=0;part<4;++part){
  uint32_t i=threadIdx.x+part*512;
  uint32_t m=use[part]?asc_ldcg(mapping+token[part]):UINT32_MAX;
  uint32_t slot=m<resident?m:UINT32_MAX;
  if(use[part] && m-resident<2048)slot=asc_ldcg(lru+m-resident);
  if(use[part]){if(slot==UINT32_MAX)asc_atomic_or(error,2U);loc[i]=slot;ids[i]=logicalId[part];}
 }
 asc_threadfence();
 // VF completion is the next consumer boundary; no work follows in this task.
}
#endif
