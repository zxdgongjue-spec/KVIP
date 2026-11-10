// SPDX-License-Identifier: Apache-2.0
// Experimental partial-permutation batches; one AIV owns all mutable maps.
#pragma once
namespace pipeline_ab {
constexpr uint32_t EMPTY=UINT32_MAX, MAP=kMapSize, THREADS=512, SET=2048, CHUNK=512;
constexpr uint32_t KEYS_POS=0, VALUES_POS=MAP, KEYS_ROW=2*MAP, VALUES_ROW=3*MAP;
constexpr uint32_t SRCSET=4*MAP, DSTSET=SRCSET+SET, PLAN_SRC=DSTSET+SET;
constexpr uint32_t PLAN_DST=PLAN_SRC+2*CHUNK, CONTROL=PLAN_DST+2*CHUNK, STAGING=CONTROL+128;
constexpr uint32_t SCRATCH_WORDS=STAGING+4*CHUNK*16;
static_assert(SCRATCH_WORDS*4<=223232,"UB budget");
__simt_callee__ inline uint32_t hash(uint32_t x){return (x*0x9e3779b9U)>>20;}
__simt_callee__ inline uint32_t get(__ubuf__ uint32_t* key,__ubuf__ uint32_t* val,uint32_t k){uint32_t h=hash(k);for(uint32_t i=0;i<MAP;++i){uint32_t v=key[h];if(v==k)return val[h];if(v==EMPTY)return k;h=(h+1)&(MAP-1);}return EMPTY;}
__simt_callee__ inline bool put(__ubuf__ uint32_t* key,__ubuf__ uint32_t* val,uint32_t k,uint32_t v){uint32_t h=hash(k);for(uint32_t i=0;i<MAP;++i){uint32_t old=asc_atomic_cas(key+h,EMPTY,k);if(old==EMPTY||old==k){val[h]=v;return true;}h=(h+1)&(MAP-1);}return false;}
__simt_callee__ inline bool insert(__ubuf__ uint32_t* table,uint32_t k){uint32_t h=(k*0x9e3779b9U)>>21;for(uint32_t i=0;i<SET;++i){uint32_t old=asc_atomic_cas(table+h,EMPTY,k);if(old==EMPTY)return true;if(old==k)return false;h=(h+1)&(SET-1);}return false;}
__simt_callee__ inline bool contains(__ubuf__ uint32_t* table,uint32_t k){uint32_t h=(k*0x9e3779b9U)>>21;for(uint32_t i=0;i<SET;++i){uint32_t old=table[h];if(old==k)return true;if(old==EMPTY)return false;h=(h+1)&(SET-1);}return false;}
__simt_callee__ inline uint32_t prefix(uint32_t x,__ubuf__ uint32_t* scan){uint32_t tid=threadIdx.x,lane=tid%32,warp=tid/32,v=x;for(uint32_t d=1;d<32;d*=2){uint32_t p=asc_shfl_up(v,d);if(lane>=d)v+=p;}if(lane==31)scan[warp]=v;asc_syncthreads();if(!warp){uint32_t p=lane<16?scan[lane]:0;for(uint32_t d=1;d<32;d*=2){uint32_t t=asc_shfl_up(p,d);if(lane>=d)p+=t;}scan[lane]=p;}asc_syncthreads();return v-x+(warp?scan[warp-1]:0);}
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void InitShadow(__ubuf__ uint32_t* m){for(uint32_t i=threadIdx.x;i<4*MAP;i+=THREADS)m[i]=EMPTY;if(threadIdx.x<128)m[CONTROL+threadIdx.x]=0;asc_syncthreads();}
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void ApplyBatch(__gm__ uint32_t* sources,__gm__ uint32_t* destinations,uint32_t begin,uint32_t count,__gm__ uint64_t* keys,__gm__ uint32_t* scales,__ubuf__ uint32_t* m){
 uint32_t tid=threadIdx.x;auto ctl=m+CONTROL;auto ps=m+PLAN_SRC;auto pd=m+PLAN_DST;auto data=reinterpret_cast<__ubuf__ uint64_t*>(m+STAGING);
 for(uint32_t base=0;base<count;base+=CHUNK){uint32_t n=(count-base)<CHUNK?count-base:CHUNK;
  for(uint32_t i=tid;i<2*SET;i+=THREADS)m[SRCSET+i]=EMPTY;asc_syncthreads();
  if(tid<n){uint32_t orig=asc_ldcg(sources+begin+base+tid),dst=asc_ldcg(destinations+begin+base+tid);uint32_t src=get(m+KEYS_ROW,m+VALUES_ROW,orig);ps[tid]=src;pd[tid]=dst;if(orig>=kHistory||dst>=kResident||src==EMPTY||!insert(m+SRCSET,src)||!insert(m+DSTSET,dst))asc_atomic_or(ctl,1U);}asc_syncthreads();if(ctl[0])return;
  uint32_t out=tid<n&&!contains(m+SRCSET,pd[tid]),freed=tid<n&&!contains(m+DSTSET,ps[tid]);uint32_t oi=prefix(out,ctl+32),fi=prefix(freed,ctl+64);if(out)ps[CHUNK+oi]=pd[tid];if(freed)pd[CHUNK+fi]=ps[tid];asc_syncthreads();uint32_t extras=ctl[32+15];uint32_t moves=n+extras;
  // Each lane handles individual fields; snapshot all rows before any write.
  for(uint32_t x=tid;x<moves*16;x+=THREADS){uint32_t i=x/16,j=i<n?i:CHUNK+i-n;data[x]=asc_ldcg(keys+uint64_t(ps[j])*16+x%16);}
  asc_syncthreads();
  for(uint32_t x=tid;x<moves*16;x+=THREADS){uint32_t i=x/16,j=i<n?i:CHUNK+i-n;asc_stcg(keys+uint64_t(pd[j])*16+x%16,data[x]);}
  asc_threadfence();asc_syncthreads();
  for(uint32_t i=tid;i<moves;i+=THREADS){uint32_t j=i<n?i:CHUNK+i-n;data[i]=asc_ldcg(scales+ps[j]);}
  asc_syncthreads();
  for(uint32_t i=tid;i<moves;i+=THREADS){uint32_t j=i<n?i:CHUNK+i-n;asc_stcg(scales+pd[j],uint32_t(data[i]));}
  asc_threadfence();asc_syncthreads();
  for(uint32_t i=tid;i<moves;i+=THREADS){uint32_t j=i<n?i:CHUNK+i-n,src=ps[j],dst=pd[j],row=get(m+KEYS_POS,m+VALUES_POS,src);data[i]=row;}asc_syncthreads();
  for(uint32_t i=tid;i<moves;i+=THREADS){uint32_t j=i<n?i:CHUNK+i-n,dst=pd[j],row=uint32_t(data[i]);if(!put(m+KEYS_POS,m+VALUES_POS,dst,row)||!put(m+KEYS_ROW,m+VALUES_ROW,row,dst))asc_atomic_or(ctl,4U);}asc_syncthreads();if(!tid)ctl[4]++;asc_syncthreads();
 }
}
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void CommitMetadata(__gm__ uint64_t* addresses,__gm__ uint64_t* logical,__ubuf__ uint32_t* m){auto key=m+KEYS_POS,val=m+VALUES_POS;auto addr=reinterpret_cast<__ubuf__ uint64_t*>(m+STAGING);auto id=addr+MAP;for(uint32_t i=threadIdx.x;i<MAP;i+=THREADS)if(key[i]!=EMPTY){addr[i]=asc_ldcg(addresses+val[i]);id[i]=asc_ldcg(logical+val[i]);}asc_syncthreads();for(uint32_t i=threadIdx.x;i<MAP;i+=THREADS)if(key[i]!=EMPTY){asc_stcg(addresses+key[i],addr[i]);asc_stcg(logical+key[i],id[i]);}asc_threadfence();asc_syncthreads();}
}
