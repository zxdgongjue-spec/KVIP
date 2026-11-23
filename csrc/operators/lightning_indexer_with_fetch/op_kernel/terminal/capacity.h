#pragma once
__simt_vf__ __aicore__ LAUNCH_BOUND(32) inline void ReadTerminalCapacity(
 __gm__ uint32_t* flags,uint32_t generation,__gm__ uint32_t* headers,
 __gm__ rotating::Row* row,uint32_t prior,bool finalCount,__ubuf__ uint32_t* result){
 uint32_t t=threadIdx.x,ready;
 do{bool done=t>=4 || asc_ldcg(flags+(finalCount?24*32+t:t*32))>=generation;
    ready=asc_ballot(done);}while(ready!=UINT32_MAX);
 uint32_t count=t<4 && !finalCount?asc_ldcg(headers+t*32+6):0;
 count+=asc_shfl_down(count,2);count+=asc_shfl_down(count,1);
 if(!t)result[0]=finalCount?asc_ldcg(&row->slots):prior+count;
}
// Only the ambiguous capacity boundary needs an exact cross-Q count.
// Claims are idempotent and are consumed by the existing pack path afterwards.
__simt_callee__ inline void ExactTerminalCapacity(
 __ubuf__ uint32_t* ids,uint32_t count,__gm__ uint32_t* mapping,
 __gm__ uint32_t* flags,uint32_t generation,uint32_t q,uint32_t prior,
 __ubuf__ uint32_t* result){
 uint32_t t=threadIdx.x,claim=0x80000000U+q;
 for(uint32_t i=t;i<count;i+=32)asc_atomic_min(mapping+ids[i],claim);
 asc_threadfence();
 if(!t)asc_stcg(flags+11*32+q,generation);
 uint32_t ready;
 do{ready=asc_ballot(t>=4 || asc_ldcg(flags+11*32+t)>=generation);}while(ready!=UINT32_MAX);
 uint32_t own=0;
 for(uint32_t base=0;base<count;base+=32){
  uint32_t i=base+t,token=i<count?ids[i]:0;
  bool fresh=i<count && asc_ldcg(mapping+token)==claim;
  uint32_t mask=asc_ballot(fresh);
  if(fresh)ids[own+__popc(mask&((1U<<t)-1U))]=token;
  own+=__popc(mask);
 }
 if(!t){asc_stcg(flags+11*32+4+q,own);asc_threadfence();asc_stcg(flags+11*32+8+q,generation);}
 do{ready=asc_ballot(t>=4 || asc_ldcg(flags+11*32+8+t)>=generation);}while(ready!=UINT32_MAX);
 uint32_t total=t<4?asc_ldcg(flags+11*32+4+t):0;
 total+=asc_shfl_down(total,2);total+=asc_shfl_down(total,1);
 if(!t){result[0]=prior+total;result[1]=own;}
}
// One device call owns the count decision; no scalar rendezvous between
// reading headers and resolving the ambiguous boundary.
__simt_vf__ __aicore__ LAUNCH_BOUND(32) inline void PrepareTerminalCapacity(
 __ubuf__ uint32_t* ids,uint32_t count,__gm__ uint32_t* mapping,
 __gm__ uint32_t* headers,__gm__ uint32_t* flags,uint32_t generation,
 uint32_t q,uint32_t prior,__ubuf__ uint32_t* result){
 uint32_t t=threadIdx.x,ready;
 do{ready=asc_ballot(t>=4 || asc_ldcg(flags+t*32)>=generation);}while(ready!=UINT32_MAX);
 uint32_t total=t<4?asc_ldcg(headers+t*32+6):0;
 total+=asc_shfl_down(total,2);total+=asc_shfl_down(total,1);
 total=asc_shfl(total,0)+prior;
 if(!t){result[0]=total;result[1]=count;result[2]=total>2048;}
 if(total>2048)ExactTerminalCapacity(ids,count,mapping,flags,generation,q,prior,result);
}
__simt_callee__ inline void MarkTerminalSlot(__ubuf__ uint32_t* bits,uint32_t slot,uint32_t resident){
 uint32_t lane=threadIdx.x%32,word=slot<resident?slot/32:UINT32_MAX;
 uint32_t mask=slot<resident?1U<<(slot%32):0;
 for(uint32_t d=1;d<32;d*=2){uint32_t previous=asc_shfl_up(mask,d),key=asc_shfl_up(word,d);
  if(lane>=d && key==word)mask|=previous;}
 uint32_t next=asc_shfl_down(word,1);
 if(word!=UINT32_MAX && (lane==31 || word!=next))asc_atomic_or(bits+word,mask);
}
// Called only when the four compact lists can exceed the supplied LRU prefix.
// The saved final TopK carries authoritative hot-hit tags. Do not evict hits
// belonging to a later Q, and do not change the original LRU prefix.
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void ExtendTerminalLru(
 __gm__ uint32_t* lru,__gm__ uint32_t* extended,__gm__ uint32_t* savedIds,
 __gm__ int32_t* inverse,__gm__ uint32_t* queryEnds,uint32_t resident,uint32_t requested,uint32_t q,
 __gm__ uint32_t* flags,__gm__ uint32_t* protection,__ubuf__ uint32_t* work){
 uint32_t t=threadIdx.x,words=(resident+31)/32;
 auto bits=work;auto offsets=bits+words;auto scan=offsets+words;auto ctl=scan+32;
 for(uint32_t i=t;i<words;i+=512)bits[i]=0;
 if(!t){ctl[0]=0;ctl[1]=!queryEnds || asc_ldcg(queryEnds+q)!=0;}
 asc_syncthreads();
 // Use one lane per bitmap word instead of contended per-slot atomics for
 // the candidate scan below; selected-hit insertion remains sparse.
 for(uint32_t i=t;i<2048;i+=512)MarkTerminalSlot(bits,asc_ldcg(lru+i),resident);
 bool active=ctl[1]!=0;
 for(uint32_t i=t;i<2048;i+=512){uint32_t tagged=active?asc_ldcg(savedIds+q*2048+i):UINT32_MAX,slot=tagged&0x3fffffffU;
  MarkTerminalSlot(bits,tagged!=UINT32_MAX && (tagged&0x80000000U)?slot:UINT32_MAX,resident);}
 asc_syncthreads();
 // Each Q publishes only its own protected hits. Merge four compact masks
 // rather than scanning all four 2048-entry lists on every worker.
 for(uint32_t i=t;i<words;i+=512)asc_stcg(protection+q*words+i,bits[i]);
 asc_syncthreads();
 if(!t){asc_threadfence();asc_atomic_add(flags+10*32,1U);asc_threadfence();}
 if(t<32){uint32_t done;do{done=asc_ballot(t!=0 || asc_ldcg(flags+10*32)==4U);}while(done!=UINT32_MAX);}
 asc_syncthreads();
 for(uint32_t i=t;i<words;i+=512){uint32_t merged=0;for(uint32_t j=0;j<4;++j)merged|=asc_ldcg(protection+j*words+i);bits[i]=merged;}
 asc_syncthreads();
 for(uint32_t i=q*512+t;i<2048;i+=2048)asc_stcg(extended+i,asc_ldcg(lru+i));
 for(uint32_t base=0;base<words;base+=512){
  uint32_t word=base+t,mask=word<words?~bits[word]:0;
  if(word+1==words && (resident%32))mask&=(1U<<(resident%32))-1U;
  uint32_t offset=prefix(__popc(mask),scan),prior=ctl[0];
  if(word<words)offsets[word]=prior+offset;
  asc_syncthreads();if(!t)ctl[0]+=scan[15];asc_syncthreads();
 }
 // One warp expands one bitmap word, distributing words across all Q cores.
 // Each active lane writes one slot; no lane serializes 32 GM stores.
 uint32_t lane=t%32;
 for(uint32_t word=4*(t/32)+q;word<words;word+=64){
  uint32_t offset=offsets[word];
  if(2048+offset>=requested)break;
  uint32_t mask=~bits[word];
  if(word+1==words && resident%32)mask&=(1U<<(resident%32))-1U;
  uint32_t ordinal=2048+offset+__popc(mask&((1U<<lane)-1U));
  if((mask&(1U<<lane)) && ordinal<requested){
   uint32_t slot=word*32+lane;
   asc_stcg(extended+ordinal,slot);asc_stcg(inverse+slot,int32_t(ordinal));
  }
 }
 asc_syncthreads();
 if(!t){ctl[0]=2048+ctl[0]<requested?2048+ctl[0]:requested;asc_threadfence();asc_atomic_add(flags+8*32,1U);asc_threadfence();}
 if(t<32){uint32_t done;do{done=asc_ballot(t!=0 || asc_ldcg(flags+8*32)==4U);}while(done!=UINT32_MAX);}
 asc_syncthreads();
 // Every worker computes the same actual capacity. Retire unused extension
 // markers too: the next graph replay must see the original inverse LRU.
 if(!q && !t)asc_stcg(flags+9*32,ctl[0]);
 if(!t)work[0]=ctl[0];
}
