__simt_callee__ inline bool InsertTerminalRecord(__ubuf__ uint32_t* table,uint32_t key,uint32_t size){
 uint32_t h=((key*0x9e3779b9U)>>21)&(size-1);
 for(uint32_t i=0;i<size;++i){uint32_t old=asc_atomic_cas(table+h,EMPTY,key);
  if(old==EMPTY)return true;if(old==key)return false;h=(h+1)&(size-1);}
 return false;
}
template<bool Restore=false>
__simt_callee__ inline void SwapColdIndexRows(
 __gm__ uint32_t* sources,__gm__ uint32_t* dest,uint32_t begin,uint32_t count,
 uint32_t resident,uint32_t history,__gm__ uint64_t* keys,__gm__ uint32_t* scales,
 __ubuf__ uint32_t* m,__gm__ uint64_t* addresses,__gm__ uint64_t* logical,
 __gm__ int32_t* inverse,uint32_t total,__gm__ uint32_t* flags){
 uint32_t t=threadIdx.x,setSize=count<=512?SET:2048;
 if constexpr(!Restore){
 if(!t){m[TAIL_CONTROL]=0;m[TAIL_CONTROL+1]=0;}
 if(!begin)for(uint32_t i=t;i<setSize;i+=THREADS){m[TAIL_SRCSET+i]=EMPTY;m[TAIL_DSTSET+i]=EMPTY;}
 asc_syncthreads();
 for(uint32_t i=t;i<count;i+=THREADS){
  uint32_t src=asc_ldcg(sources+i),dst=asc_ldcg(dest+i);
  m[TAIL_PLAN_SRC+begin+i]=src;m[TAIL_PLAN_DST+begin+i]=dst;
  if(src<resident && (!inverse || uint32_t(asc_ldcg(inverse+src))<total))asc_atomic_or(m+TAIL_CONTROL+1,1U);
  if(src>=history || dst>=resident ||
     !(count<=512?insert(m+TAIL_SRCSET,src):InsertTerminalRecord(m+TAIL_SRCSET,src,setSize)) || !(count<=512?insert(m+TAIL_DSTSET,dst):InsertTerminalRecord(m+TAIL_DSTSET,dst,setSize)))asc_atomic_or(m+TAIL_CONTROL,1U);
 }
 asc_syncthreads();
 // This replaces the old output-reader join. Do not write any Index row
 // before every Q has finished its output reads and validated its partition.
 if(!t){asc_threadfence();asc_atomic_add(flags+5*32,m[TAIL_CONTROL+1]?257U:1U);asc_threadfence();}
 if(t<32){
  uint32_t value=0,ready;
  do{if(!t)value=asc_ldcg(flags+5*32);ready=asc_ballot(t!=0 || (value&255U)==4U);}while(ready!=UINT32_MAX);
  if(!t && (value>>8))m[TAIL_CONTROL]|=0x80000000U;
 }
 asc_syncthreads();
 if(m[TAIL_CONTROL])return;
 }
 // The validated plan is immutable; keys and metadata occupy disjoint arrays.
 if(t<448){
  for(uint32_t i=t;i<count*16;i+=448){
   uint32_t src=m[TAIL_PLAN_SRC+begin+i/16],dst=m[TAIL_PLAN_DST+begin+i/16],j=i%16;
   auto x=asc_ldcg(keys+16ULL*src+j),y=asc_ldcg(keys+16ULL*dst+j);
   asc_stcg(keys+16ULL*src+j,y);asc_stcg(keys+16ULL*dst+j,x);
  }
 }else{
  for(uint32_t i=t-448;i<count;i+=64){
   uint32_t src=m[TAIL_PLAN_SRC+begin+i],dst=m[TAIL_PLAN_DST+begin+i];
   auto sx=asc_ldcg(scales+src),sy=asc_ldcg(scales+dst);
   auto ax=asc_ldcg(addresses+src),ay=asc_ldcg(addresses+dst);
   auto lx=asc_ldcg(logical+src),ly=asc_ldcg(logical+dst);
   asc_stcg(scales+src,sy);asc_stcg(scales+dst,sx);
   asc_stcg(addresses+src,ay);asc_stcg(addresses+dst,ax);
   asc_stcg(logical+src,ly);asc_stcg(logical+dst,lx);
  }
 }
 asc_threadfence();
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
