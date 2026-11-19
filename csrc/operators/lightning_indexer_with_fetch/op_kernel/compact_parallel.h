// SPDX-License-Identifier: Apache-2.0
#pragma once
// Q1 already owns its slots. The earliest (query, position) in Q2–Q4
// owns each new token, preserving the sequential LRU allocation order.
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void ResolveCompactParallel(
 __ubuf__ Args* a,StagedContextPointer ctx,uint32_t b,uint32_t resident,uint32_t used,
 uint32_t workers,uint32_t stride,uint32_t meta,__ubuf__ uint32_t* keys,__ubuf__ uint32_t* ranks,
 __ubuf__ uint32_t* slots,__ubuf__ uint32_t* lru,__ubuf__ uint32_t* ctl){
 uint32_t t=threadIdx.x,q=t/64,l=t%64,warp=t/32;
 auto n=ctl+64,warpPrefix=n+4,groupOffset=warpPrefix+8;
 auto hdr=reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)+b*128;
 if(t<4)n[t]=t?asc_ldcg(hdr+32*t+1)-1:0;
 if(!t)ctl[16]=0;
 asc_syncthreads();
 if(n[0]>64 || n[1]>64 || n[2]>64 || n[3]>64)return;
 uint32_t old=ctl[0];
 uint32_t upper=n[0]+n[1]+n[2]+n[3];
 if(used+old+upper>kLruSlots)return;
 if(t<upper){
  auto slot=lru[used+old+t];
  auto inverse=reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+b*resident;
  if(slot>=resident || asc_ldcg(inverse+slot)!=int32_t(used+old+t))asc_atomic_or(ctl+16,1U);
 }
 asc_syncthreads();
 if(ctl[16])return;
 uint32_t entry=0,h=0,token=0;
 auto srcBlocks=reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks);
 if(l<n[q]){
  entry=uint32_t(asc_ldcg(srcBlocks+(b*4+q)*kTop+l));
  token=entry>>kPositionBits;h=SparseHash(token)&(ctl[7]-1);
  while(true){
   auto previous=asc_atomic_cas(keys+h,UINT32_MAX,token);
   if(previous==UINT32_MAX || previous==token)break;
   h=(h+1)&(ctl[7]-1);
  }
  if(ranks[h]==UINT32_MAX || !(ranks[h]&kStagedRankTag))asc_atomic_min(ranks+h,t);
 }
 asc_syncthreads();
 bool fresh=l<n[q] && ranks[h]==t;
 uint32_t bits=asc_ballot(fresh);
 if(!(t%32))warpPrefix[warp]=__popc(bits);
 asc_syncthreads();
 if(t<8){
  uint32_t sum=warpPrefix[t];
  for(uint32_t d=1;d<=4;d*=2){auto earlier=asc_shfl_up(sum,d);if(t>=d)sum+=earlier;}
  warpPrefix[t]=sum;
 }
 asc_syncthreads();
 uint32_t before=q?warpPrefix[2*q-1]:0,after=warpPrefix[2*q+1],added=after-before;
 uint32_t w=q*workers+b%workers;
 auto base=reinterpret_cast<__gm__ int32_t*>(a->packedFetch);
 auto direct=base+4*workers*(3*stride+meta)+w*(6*stride+64);
 auto counters=reinterpret_cast<__gm__ uint32_t*>(direct+6*stride);
 if(q && !l)groupOffset[q]=added?asc_atomic_add(counters,added):0;
 asc_syncthreads();
 if(fresh){
  uint32_t rank=(warp?warpPrefix[warp-1]:0)+__popc(bits&((1U<<(t%32))-1));
  uint32_t inQ=rank-before;rank+=old;uint32_t ordinal=used+rank,slot=lru[ordinal];
  ranks[h]=kStagedRankTag|rank;slots[rank]=slot;
  auto changes=reinterpret_cast<__gm__ uint32_t*>(ctx->changes)+b*kChangeCapacity;
  asc_stcg(changes+kLruSlots+rank,UINT32_MAX);asc_stcg(changes+ordinal,token);
  asc_stcg(reinterpret_cast<__gm__ uint32_t*>(ctx->changeSlots)+b*kChangeCapacity+ordinal,slot);
  asc_stcg(reinterpret_cast<__gm__ uint32_t*>(a->eventsSrc)+b*a->maxEvents+kPrefetch+rank,token);
  asc_stcg(reinterpret_cast<__gm__ uint32_t*>(a->eventsDst)+b*a->maxEvents+kPrefetch+rank,slot);
  uint32_t pos=groupOffset[q]+inQ,out=(b*4+q)*kTop+inQ;
  uint32_t dstSlot=OwnerRow(ctx,b)*OutputStride(resident,ctx)+slot;
  auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
  auto addr=asc_ldcg(reinterpret_cast<__gm__ uint64_t*>(row->addresses)+token);
  StoreFallbackSource(ctx->shape.history,ctx->shape.tokenBytes,ctx,b,out,token,addr,srcBlocks,reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets));
  asc_stcg(reinterpret_cast<__gm__ int32_t*>(ctx->destinations)+out,int32_t(dstSlot));
  asc_stcg(reinterpret_cast<__gm__ uint64_t*>(base+w*(3*stride+meta))+pos,addr);
  asc_stcg(direct+pos,int32_t(pos));asc_stcg(direct+stride+pos,0);asc_stcg(direct+2*stride+pos,int32_t(dstSlot));
 }
 asc_syncthreads();
 if(l<n[q])asc_stcg(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+(b*4+q)*kTop+(entry&kPositionMask),slots[ranks[h]&~kStagedRankTag]);
 auto post=reinterpret_cast<__gm__ FinalPost*>(ctx->owners+ctx->shape.finalPostOffset)+b*4+q;
 if(q && l<8){
  uint32_t value=l==0 || l==5?added:l==3?uint32_t(added!=0):l==4?(b*4+q)*kTop:0;
  asc_stcg(reinterpret_cast<__gm__ uint32_t*>(post)+l,value);
 }
 auto f=reinterpret_cast<__gm__ BatchFlag*>(a->flags)+b*kFlagCount+kRawQueryFlag+q;
 if(q && !l){
  auto status=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*32;
  asc_stcg(status+4+q,added);asc_stcg(status+8+q,old+after);
  asc_stcg(&f->reserved[0],1U);asc_stcg(&f->count,added);asc_stcg(&f->error,0U);
  ctl[2+q]=added;
 }
 if(!t){
  uint32_t total=old+warpPrefix[7];
  ctl[0]=total;ctl[20]=4;
  asc_stcg(reinterpret_cast<__gm__ uint32_t*>(ctx->changeCount)+b,kLruSlots+total);
  if(total>old)asc_atomic_add(reinterpret_cast<__gm__ uint32_t*>(a->liveCount?a->liveCount:ctx->count),total-old);
 }
 asc_threadfence();asc_syncthreads();
 if(q && !l){asc_atomic_add(counters+kGroupReadyWord,1U);asc_stcg(&f->ready,1U);}
 asc_threadfence();
}
