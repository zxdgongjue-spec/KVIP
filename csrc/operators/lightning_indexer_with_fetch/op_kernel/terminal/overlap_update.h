#pragma once
// Replicate the small immutable assignment plan; distribute the payload work.
// All four workers snapshot their old rows before any worker writes Index.
// The closure retains input-order pairing used by the original planner.
constexpr uint32_t kParallelIndexWords=223232/4;
constexpr uint32_t kIndexRowWords=38;
constexpr uint32_t kParallelSet=4096;
__simt_callee__ inline bool ParallelInsert(__ubuf__ uint32_t* set,uint32_t key,uint32_t size){
 uint32_t h=key*0x9e3779b9U,slot=(h^(h>>16))&(size-1);
 for(uint32_t i=0;i<size;++i){uint32_t old=asc_atomic_cas(set+slot,UINT32_MAX,key);
  if(old==UINT32_MAX)return true;if(old==key)return false;slot=(slot+1)&(size-1);}
 return false;
}
__simt_callee__ inline bool ParallelContains(__ubuf__ uint32_t* set,uint32_t key,uint32_t size){
 uint32_t h=key*0x9e3779b9U,slot=(h^(h>>16))&(size-1);
 for(uint32_t i=0;i<size;++i){uint32_t old=set[slot];
  if(old==key)return true;if(old==UINT32_MAX)return false;slot=(slot+1)&(size-1);}
 return false;
}
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void ApplyParallelIndex(
 __gm__ uint32_t* sources,__gm__ uint32_t* destinations,uint32_t count,
 uint32_t resident,uint32_t history,uint32_t q,uint32_t generation,
 __gm__ uint64_t* keys,__gm__ uint32_t* scales,__gm__ uint64_t* addresses,
 __gm__ uint64_t* logical,__gm__ uint32_t* valid,__gm__ uint32_t* flags,
 __gm__ int32_t* inverse,__ubuf__ uint32_t* work){
 uint32_t t=threadIdx.x,words=128;
 while(words<count*2 && words<kParallelSet)words*=2;
 auto ctl=work;auto sourceBits=work+128;auto destBits=sourceBits+words;
 auto ps=destBits+words;auto pd=ps+2*count;
 auto payload=pd+2*count;
 if(!t){ctl[0]=0;ctl[1]=0;ctl[2]=0;}
 for(uint32_t i=t;i<(inverse?words:2*words);i+=512)sourceBits[i]=UINT32_MAX;
 asc_syncthreads();
 for(uint32_t i=t;i<count;i+=512){
  uint32_t src=asc_ldcg(sources+i),dst=asc_ldcg(destinations+i);
  ps[i]=src;pd[i]=dst;
  if(src>=history || dst>=resident){asc_atomic_or(ctl,1U);continue;}
  if(src<resident && !ParallelInsert(sourceBits,src,words))asc_atomic_or(ctl,1U);
  if(!inverse && !ParallelInsert(destBits,dst,words))asc_atomic_or(ctl,1U);
 }
 asc_syncthreads();
 if(ctl[0])return;
 for(uint32_t base=0;base<count;base+=512){
  uint32_t i=base+t;
  uint32_t out=i<count && !ParallelContains(sourceBits,pd[i],words);
  uint32_t freed=i<count && (ps[i]>=resident || (inverse?uint32_t(asc_ldcg(inverse+ps[i]))>=count:!ParallelContains(destBits,ps[i],words)));
  uint32_t packed=prefix(out|(freed<<16),ctl+32);
  if(out)ps[count+ctl[1]+(packed&65535U)]=pd[i];
  if(freed)pd[count+ctl[2]+(packed>>16)]=ps[i];
  asc_syncthreads();
  if(!t){ctl[1]+=ctl[47]&65535U;ctl[2]+=ctl[47]>>16;}
  asc_syncthreads();
 }
 uint32_t moves=count+ctl[1],begin=moves*q/4,end=moves*(q+1)/4;
 if(ctl[1]!=ctl[2] || uint32_t(payload-work)+((moves+3)/4)*kIndexRowWords>kParallelIndexWords){
  if(!t)ctl[0]=2U;return;
 }
 auto keyData=reinterpret_cast<__ubuf__ uint64_t*>(payload);
 auto scaleData=payload+(end-begin)*32;
 auto addressData=reinterpret_cast<__ubuf__ uint64_t*>(scaleData+((end-begin+1)/2)*2);
 auto logicalData=addressData+(end-begin);
 if(t<448){
  for(uint32_t x=t;x<(end-begin)*16;x+=448){uint32_t src=ps[begin+x/16];keyData[x]=asc_ldcg(keys+uint64_t(src)*16+x%16);}
 }else{
  for(uint32_t i=t-448;i<end-begin;i+=64){uint32_t src=ps[begin+i];
   scaleData[i]=asc_ldcg(scales+src);addressData[i]=asc_ldcg(addresses+src);logicalData[i]=asc_ldcg(logical+src);}
 }
 asc_syncthreads();
 if(!t){asc_threadfence();asc_atomic_add(flags+7*32,1U);asc_threadfence();}
 if(t<32){uint32_t done;do{done=asc_ballot(t!=0 || asc_ldcg(flags+7*32)==generation*4);}while(done!=UINT32_MAX);}
 asc_syncthreads();
 if(t<448){
  for(uint32_t x=t;x<(end-begin)*16;x+=448){uint32_t dst=pd[begin+x/16];asc_stcg(keys+uint64_t(dst)*16+x%16,keyData[x]);}
 }else{
  for(uint32_t i=t-448;i<end-begin;i+=64){uint32_t dst=pd[begin+i];
   asc_stcg(scales+dst,scaleData[i]);asc_stcg(addresses+dst,addressData[i]);asc_stcg(logical+dst,logicalData[i]);}
 }
 for(uint32_t i=begin+t;i<end;i+=512)if(i>=count && pd[i]<resident)asc_stcg(valid+pd[i],0U);
 asc_threadfence();
 asc_syncthreads();
}
