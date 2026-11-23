#pragma once
// The original window's exact UB bitmap, partitioned by Q ownership.
// Same terminal slot/record operations, with request-local pointers resolved once.
__simt_callee__ inline void PrepareDirect(
 uint32_t resident,__ubuf__ uint32_t* ids,__ubuf__ uint32_t* slots,
 __gm__ uint32_t* map,uint32_t count,uint32_t slotBase,
 __gm__ uint32_t* lru,__gm__ uint32_t* changes,__gm__ uint32_t* changeSlots,
 __gm__ rotating::MappingJournal* journal){
 for(uint32_t i=threadIdx.x;i<count;i+=1024){
  uint32_t token=ids[i],slot=slotBase+i,ordinal=slot-resident;
  uint32_t physical=lru && ordinal<rotating::kLruSlots?asc_ldcg(lru+ordinal):slot;
  slots[i]=physical;asc_stcg(map+token,physical);
  // Change records are written by separate warp groups after the existing rendezvous.
 }
 asc_threadfence();
}

__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void PackUniqueLocal(
 __ubuf__ uint32_t* input,uint32_t width,uint32_t n0,uint32_t n1,uint32_t n2,uint32_t n3,uint32_t q,
 __gm__ uint32_t* mapping,__ubuf__ uint32_t* ids,__ubuf__ uint32_t* counter,__ubuf__ uint32_t* bits,__gm__ rotating::Row* row,__gm__ uint64_t* addresses,__ubuf__ uint32_t* lo,__ubuf__ uint32_t* hi,uint32_t resident,__ubuf__ uint32_t* slots,__gm__ uint32_t* lru,uint32_t slotCapacity,bool preclaimed,bool capacityProbe,__gm__ uint32_t* headers,uint32_t priorSlots,__gm__ uint32_t* changes,__gm__ uint32_t* changeSlots,uint64_t tailRange,uint32_t tailCapacity,__gm__ uint32_t* boundTemplate,__gm__ KvafExecuteBatchGetKernelArgs* qArgs,__ubuf__ uint32_t* fullIds,__ubuf__ uint32_t* outputSlots,__ubuf__ uint32_t* error,__gm__ uint64_t* logical,__gm__ uint32_t* flags,uint32_t generation,uint32_t history,__gm__ uint32_t* selected,uint32_t outputBase,__gm__ uint64_t* update,uint32_t batch,uint32_t query){
 uint32_t tailBegin=uint32_t(tailRange),tailEnd=uint32_t(tailRange>>32);
 uint32_t counts[4]={n0,n1,n2,n3},tid=threadIdx.x,lane=tid%32;
 if(tid>=2 && tid<9 && tid!=5)counter[tid]=0; // The original block barrier publishes all control words.
 if(tid<32){uint32_t upper=0;
  if(capacityProbe){uint32_t ready;
   do{ready=asc_ballot(lane>=4 || asc_ldcg(flags+lane*32)>=generation);}while(ready!=UINT32_MAX);
   upper=lane<4?asc_ldcg(headers+lane*32+6):0;
   upper+=asc_shfl_down(upper,2);upper+=asc_shfl_down(upper,1);
   upper+=priorSlots;
  }
  if(!lane)counter[14]=capacityProbe && upper>2048;
 }

 asc_syncthreads();
 if(counter[14])return;
 // High-bit owner tags sort above existing prior-window mappings.
 const uint32_t claim=0x80000000U+q;
 if(tid<96 && !preclaimed){
  if(!preclaimed)for(uint32_t i=tid;i<counts[q];i+=96)asc_atomic_min(mapping+input[i],claim);
  asc_threadfence();
  if(tid>=32){asc_threadfence_block();if(!lane)*reinterpret_cast<__ubuf__ volatile uint32_t*>(counter+5+tid/32)=1;}
 }
 if(tid<32){
  if(!preclaimed){
  uint32_t claimed;do{bool done=lane>=2 || *reinterpret_cast<__ubuf__ volatile uint32_t*>(counter+6+lane)==1;claimed=asc_ballot(done);}while(claimed!=UINT32_MAX);
  asc_threadfence_block();
  if(!lane){asc_stcg(flags+20*32+q,generation);asc_threadfence();}
  uint32_t ready;
  do{bool done=lane>=q+1 || asc_ldcg(flags+20*32+lane)>=generation;ready=asc_ballot(done);}while(ready!=UINT32_MAX);
  }
  uint32_t count=0;
  if(preclaimed){count=counts[q];for(uint32_t i=lane;i<count;i+=32)ids[i]=input[i];}
  else if(counts[q]>64 && counts[q]<=96){
   uint32_t t0=input[lane],t1=input[lane+32],t2=input[lane+64<counts[q]?lane+64:counts[q]-1];
   uint32_t o0=asc_ldcg(mapping+t0),o1=asc_ldcg(mapping+t1),o2=asc_ldcg(mapping+t2);
   bool f0=o0==claim,f1=o1==claim,f2=lane+64<counts[q] && o2==claim;
   uint32_t m0=asc_ballot(f0),m1=asc_ballot(f1),m2=asc_ballot(f2),lower=(1U<<lane)-1;
   uint32_t c0=__popc(m0),c1=__popc(m1);
   if(f0)ids[__popc(m0&lower)]=t0;
   if(f1)ids[c0+__popc(m1&lower)]=t1;
   if(f2)ids[c0+c1+__popc(m2&lower)]=t2;
   count=c0+c1+__popc(m2);
  }else{
  for(uint32_t base=0;base<counts[q];base+=32){uint32_t i=base+lane,token=i<counts[q]?input[i]:0;
   bool fresh=i<counts[q] && asc_ldcg(mapping+token)==claim;uint32_t mask=asc_ballot(fresh);
   if(fresh)ids[count+__popc(mask&((1U<<lane)-1))]=token;
   count+=__popc(mask);
  }
  }
  asc_threadfence_block();
  uint32_t slotBase=0;if(!lane && count)slotBase=asc_atomic_add(&row->slots,count);slotBase=asc_shfl(slotBase,0);
  if(slotBase>slotCapacity || count>slotCapacity-slotBase){
   if(!lane)asc_atomic_or(error,8U);count=0;
  }
  if(!lane){counter[0]=count;counter[1]=slotBase;}
  asc_threadfence_block();if(!lane)*reinterpret_cast<__ubuf__ volatile uint32_t*>(counter+2)=1;
  for(uint32_t i=lane;i<count;i+=96){uint32_t token=ids[i],ordinal=slotBase+i;
   uint32_t physical=asc_ldcg(lru+ordinal);uint64_t remote=asc_ldcg(addresses+token);
   slots[i]=outputBase+physical;lo[i]=uint32_t(remote);hi[i]=uint32_t(remote>>32);
   asc_stcg(mapping+token,physical);asc_stcg(changes+ordinal,token);asc_stcg(changeSlots+ordinal,physical);
  }
  asc_threadfence();
  uint32_t finished;do{bool done=lane>=2 || *reinterpret_cast<__ubuf__ volatile uint32_t*>(counter+3+lane)==1;finished=asc_ballot(done);}while(finished!=UINT32_MAX);
  if(!lane){asc_stcg(flags+24*32+q,generation);asc_threadfence();}
 }else if(tid<96){
  uint32_t ready;do{bool done=*reinterpret_cast<__ubuf__ volatile uint32_t*>(counter+2)==1;ready=asc_ballot(done);}while(ready!=UINT32_MAX);
  asc_threadfence_block();
  uint32_t count=counter[0],slotBase=counter[1];
  for(uint32_t i=tid;i<count;i+=96){uint32_t token=ids[i],ordinal=slotBase+i;
   uint32_t physical=asc_ldcg(lru+ordinal);uint64_t remote=asc_ldcg(addresses+token);
   slots[i]=outputBase+physical;lo[i]=uint32_t(remote);hi[i]=uint32_t(remote>>32);
   asc_stcg(mapping+token,physical);asc_stcg(changes+ordinal,token);asc_stcg(changeSlots+ordinal,physical);
  }
  asc_threadfence();asc_threadfence_block();if(!lane)*reinterpret_cast<__ubuf__ volatile uint32_t*>(counter+2+tid/32)=1;
 }else{
  if(tid>=96 && tid<128){
   uint32_t f=tid-96;auto cfg=lo+6144;
   auto unpack=reinterpret_cast<__gm__ int64_t*>(asc_ldcg(update+20));
   auto valid=reinterpret_cast<__gm__ int64_t*>(asc_ldcg(update+21));
   for(uint32_t w=0;w<2;++w){
    uint32_t row=w*32+f;
    bool active=row<batch*4 && asc_ldcg(valid+row)!=0;
    uint64_t source=active?uint64_t(asc_ldcg(unpack+row)):UINT64_MAX;
    uint32_t mask=asc_ballot(active && source==query);
    uint32_t pad=asc_ballot(row<batch*4 && !active && query%4==0 && row/4==query/4);
    if(!f){counter[9+w]=mask;counter[11+w]=pad;}
   }
   if(f!=16 && f!=18 && f!=19)cfg[f]=asc_ldcg(boundTemplate+f);
   if(f==0){auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(asc_ldcg(&qArgs->queue_state));uint64_t head=asc_ldcg(&queue->sq_producer_index);cfg[16]=uint32_t(head);cfg[18]=uint32_t(head);cfg[19]=uint32_t(head>>32);}
  }
  if(tid>=128)for(uint32_t i=tid-128;i<2048;i+=384){uint32_t tagged=fullIds[i],token=tagged&0x3fffffffU,slot=UINT32_MAX;
   outputSlots[i]=UINT32_MAX;
   asc_stcg(selected+i,tagged==UINT32_MAX?UINT32_MAX:token);
   if(tagged==UINT32_MAX)continue;
   if(token>=history){asc_atomic_or(error,1U);fullIds[i]=UINT32_MAX;continue;}
   if((tagged&0x80000000U)&&token<resident)slot=token;
   else slot=rotating::DirectTailSlot(token,resident,tailBegin,tailEnd,tailCapacity);
   if(slot!=UINT32_MAX){outputSlots[i]=slot;fullIds[i]=asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(logical)+2*token);}
  }
 }
 asc_syncthreads();
 // Emit begins with the producer-warp wait and its own block rendezvous.
 Emit(fullIds,outputSlots,error,mapping,lru,logical,history,resident,history-512,256,512,nullptr,nullptr,flags,generation,q+1);
 if(tid<32){uint32_t ready;
  do{ready=asc_ballot(lane>=4 || asc_ldcg(flags+24*32+lane)>=generation);}while(ready!=UINT32_MAX);
  if(!lane)counter[13]=asc_ldcg(&row->slots);
 }



}
// Payload precedes release publication; readers bypass scalar cache.
// One full warp; no inter-warp state or payload reads. Callers publish separately.
__simt_vf__ __aicore__ LAUNCH_BOUND(32) inline void WaitFour(
 __gm__ uint32_t* flags,uint32_t base,uint32_t generation,__ubuf__ uint32_t* counts,
 uint32_t q,uint32_t own,bool publish,bool readCounts,uint32_t limit){
 // Monotonic arrival count: every producer completes output before arrival.
 if(!threadIdx.x){asc_threadfence();asc_atomic_add(flags+base*32,1U);asc_threadfence();}
 uint32_t readyMask;
 do {bool ready=threadIdx.x!=0 || asc_ldcg(flags+base*32)==generation*limit;readyMask=asc_ballot(ready);}while(readyMask!=UINT32_MAX);
}
