template<bool Restore=false>
__simt_callee__ inline void SwapColdIndexRows(
 __gm__ uint32_t* sources,__gm__ uint32_t* dest,uint32_t begin,uint32_t count,
 uint32_t resident,uint32_t history,__gm__ uint64_t* keys,__gm__ uint32_t* scales,
 __ubuf__ uint32_t* m){
 uint32_t t=threadIdx.x;
 if constexpr(!Restore){
 if(!t)m[TAIL_CONTROL]=0;
 if(!begin)for(uint32_t i=t;i<SET;i+=THREADS){m[TAIL_SRCSET+i]=EMPTY;m[TAIL_DSTSET+i]=EMPTY;}
 asc_syncthreads();
 if(t<count){
  uint32_t i=t;
  uint32_t src=asc_ldcg(sources+i),dst=asc_ldcg(dest+i);
  m[TAIL_PLAN_SRC+begin+t]=src;m[TAIL_PLAN_DST+begin+t]=dst;
  if(src<resident || src>=history || dst>=resident ||
     !insert(m+TAIL_SRCSET,src) || !insert(m+TAIL_DSTSET,dst))asc_atomic_or(m+TAIL_CONTROL,1U);
 }
 asc_syncthreads();
 if(m[TAIL_CONTROL])return;
 }
 for(uint32_t i=t;i<count*16;i+=THREADS){
  uint32_t src=m[TAIL_PLAN_SRC+begin+i/16],dst=m[TAIL_PLAN_DST+begin+i/16],j=i%16;
  auto x=asc_ldcg(keys+16ULL*src+j),y=asc_ldcg(keys+16ULL*dst+j);
  asc_stcg(keys+16ULL*src+j,y);asc_stcg(keys+16ULL*dst+j,x);
 }
 if(t<count){
  uint32_t src=m[TAIL_PLAN_SRC+begin+t],dst=m[TAIL_PLAN_DST+begin+t];
  auto x=asc_ldcg(scales+src),y=asc_ldcg(scales+dst);
  asc_stcg(scales+src,y);asc_stcg(scales+dst,x);
 }
 // The following metadata phase fences both sets of stores.
}
__simt_callee__ inline void SwapColdMetadata(
 __gm__ uint64_t* addr,__gm__ uint64_t* logical,uint32_t n,__ubuf__ uint32_t* m){
 uint32_t t=threadIdx.x;
 if(t<n){
  uint32_t src=m[TAIL_PLAN_SRC+t],dst=m[TAIL_PLAN_DST+t];
  auto x=asc_ldcg(addr+src),y=asc_ldcg(addr+dst);
  auto a=asc_ldcg(logical+src),b=asc_ldcg(logical+dst);
  asc_stcg(addr+src,y);asc_stcg(addr+dst,x);asc_stcg(logical+src,b);asc_stcg(logical+dst,a);
 }
 asc_threadfence();
}
