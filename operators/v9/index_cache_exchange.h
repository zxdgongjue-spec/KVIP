#pragma once
namespace rotating {
// Move disjoint rows. Each phase joins before any source range is reused.
__simt_callee__ inline void MoveIndexRow(__gm__ Contract* c, uint32_t b, bool sh, uint32_t src,
                                        bool dh, uint32_t dst, uint32_t word) {
  using namespace contract_simt;
  asc_stcg(IndexRowKey(c,b,dh,dst)+word,asc_ldcg(IndexRowKey(c,b,sh,src)+word));
  if(!word) asc_stcg(IndexRowScale(c,b,dh,dst),asc_ldcg(IndexRowScale(c,b,sh,src)));
}

// Only reservations and final misses can change H0 at the tail. A reserved
// slot may also be in the final plan: deduplicate slots before reading old
// owners. No dependence on other final workers' touched bitmap or CQ state.
struct IndexChanges { uint32_t demote, promote; bool valid; };
__simt_callee__ inline IndexChanges LoadIndexPlanRecords(__gm__ Context* ctx,uint32_t b,
                                                        __ubuf__ uint32_t* scratch) {
  using namespace contract_simt;
  auto c=ContractOf(ctx);
  auto state=ContractPtr<int32_t>(IndexBase(c,b));
  int32_t records=asc_ldcg(state+28);
  bool ready=asc_ldcg(state+27)==1 && records>=0 && uint32_t(records)<=ContractBudget(c,b) &&
      2*uint32_t(records)<=c->resident_capacity+c->candidate_capacity &&
      asc_ldcg(state+29)==ContractPtr<int32_t>(c->plan_count)[b] &&
      asc_ldcg(state+30)==int32_t(ContractPtr<Row>(ctx->rows)[b].slots);
  if(!ready)return {0,0,false};
  uint32_t n=uint32_t(records),tid=threadIdx.x,count=0;
  auto jobs=IndexPositions(c,b)+c->candidate_capacity;
  auto incoming=scratch,outgoing=scratch+kIndexDeltaCapacity,scan=scratch+2*kIndexDeltaCapacity;
  for(uint32_t i=tid;i<n;i+=256) {
    int32_t old=asc_ldcg(jobs+2*i),next=asc_ldcg(jobs+2*i+1);
    outgoing[i]=uint32_t(old);incoming[i]=uint32_t(next);
    count+=uint32_t(old>=0)+(uint32_t(next>=0)<<16);
  }
  auto prefix=ContractPrefixUB(count,scan);
  if((prefix.total&0xffffU)==n && (prefix.total>>16)==n) {
    if(!tid)asc_stcg(state+27,0);
    asc_threadfence();asc_syncthreads();return {n,n,true};
  }
  uint32_t rank=prefix.begin&0xffffU,inRank=prefix.begin>>16;
  for(uint32_t i=tid;i<n;i+=256) {
    int32_t old=asc_ldcg(jobs+2*i);
    if(old>=0)outgoing[rank++]=uint32_t(old);
    int32_t next=asc_ldcg(jobs+2*i+1);
    if(next>=0)incoming[inRank++]=uint32_t(next);
  }
  asc_syncthreads();
  if(!tid)asc_stcg(state+27,0);
  asc_threadfence();return {prefix.total&0xffffU,prefix.total>>16,true};
}

__simt_callee__ inline IndexChanges BuildIndexChanges(__gm__ Context* ctx,uint32_t b,
                                                      __ubuf__ uint32_t* scratch) {
  using namespace contract_simt;
  auto c=ContractOf(ctx);
  uint32_t tid=threadIdx.x,pool=ContractPool(c,b),length=ContractLength(c,b);
  uint32_t reserved=ContractPtr<Row>(ctx->rows)[b].slots;
  uint32_t planned=ContractPtr<int32_t>(c->plan_count)[b],n=reserved+planned;
  auto reverse=ContractPtr<int32_t>(c->reverse)+uint64_t(pool)*c->resident_capacity;
  auto forward=ContractPtr<int32_t>(c->forward)+uint64_t(pool)*c->candidate_capacity;
  auto pos=IndexPositions(c,b),jobs=pos+c->candidate_capacity;
  auto target=ContractPtr<int32_t>(c->next_reverse)+b*c->resident_capacity;
  auto incoming=scratch,outgoing=scratch+kIndexDeltaCapacity,scan=scratch+2*kIndexDeltaCapacity;
  for(uint32_t i=tid;i<512;i+=256)scratch[i]=0;
  asc_syncthreads();
  for(uint32_t i=tid;i<n;i+=256) {
    int32_t slot=i<reserved?ContractPtr<int32_t>(c->reserved_slots)[b*2048+i]:
        ContractPtr<int32_t>(c->planned_slots)[b*kSlots+i-reserved];
    uint32_t mask=1U<<(slot%32);
    bool first=!(asc_atomic_or(scratch+slot/32,mask)&mask);
    asc_stcg(jobs+i,first?slot:-1);
    if(first)asc_stcg(target+slot,-1);
  }
  asc_threadfence();asc_syncthreads();
  for(uint32_t i=tid;i<planned;i+=256)
    asc_stcg(target+ContractPtr<int32_t>(c->planned_slots)[b*kSlots+i],
             ContractPtr<int32_t>(c->planned_tokens)[b*kSlots+i]);
  for(uint32_t i=tid;i<reserved;i+=256) {
    int32_t slot=ContractPtr<int32_t>(c->extra_prefetch_slots)[b*2048+i];
    if(slot>=0)asc_stcg(target+slot,ContractPtr<int32_t>(c->reserved_tokens)[b*2048+i]);
  }
  asc_threadfence();asc_syncthreads();
  // Cache each dependent metadata lookup once. incoming/outgoing are sparse
  // by candidate here, then compacted through jobs after its slot list dies.
  // This avoids repeatedly chasing reverse -> forward -> positions in GM.
  if(!tid)scan[8]=0;
  asc_syncthreads();
  uint32_t invalid=0,demoteCount=0,promoteCount=0;
  for(uint32_t i=tid;i<n;i+=256) {
    int32_t slot=asc_ldcg(jobs+i),oldPosition=-1,newPosition=-1;
    if(slot>=0) {
      int32_t t=asc_ldcg(target+slot);
      if(t>=0) {
        if(uint32_t(t)>=length)++invalid;
        else { newPosition=asc_ldcg(pos+t);invalid+=newPosition<0; }
      }
      int32_t old=asc_ldcg(reverse+slot);
      if(old>=0 && uint32_t(old)<length && asc_ldcg(forward+old)==slot) {
        int32_t at=asc_ldcg(pos+old);
        if(at>-2)++invalid;
        else oldPosition=int32_t(-int64_t(at)-2);
      }
    }
    outgoing[i]=uint32_t(oldPosition);incoming[i]=uint32_t(newPosition);
    demoteCount+=oldPosition>=0;promoteCount+=newPosition>=0;
  }
  if(invalid)asc_atomic_or(scan+8,1U);
  asc_syncthreads();
  if(scan[8])return {0,0,false};
  // Each sum <= n <= 10240, so the low-half sum cannot carry into high.
  auto prefix=ContractPrefixUB(demoteCount+(promoteCount<<16),scan);
  uint32_t demote=prefix.total&0xffffU,promote=prefix.total>>16;
  uint32_t dr=prefix.begin&0xffffU,pr=prefix.begin>>16;
  // At most S demotions plus C promotions fit the existing S+C jobs array.
  // A separate compact destination avoids overwriting another lane's sparse
  // UB input. All jobs[] readers above joined before these writes.
  for(uint32_t i=tid;i<n;i+=256) {
    if(outgoing[i]!=UINT32_MAX)asc_stcg(jobs+dr++,int32_t(outgoing[i]));
    if(incoming[i]!=UINT32_MAX)asc_stcg(jobs+demote+pr++,int32_t(incoming[i]));
  }
  asc_threadfence();asc_syncthreads();
  for(uint32_t i=tid;i<demote;i+=256)outgoing[i]=uint32_t(asc_ldcg(jobs+i));
  for(uint32_t i=tid;i<promote;i+=256)incoming[i]=uint32_t(asc_ldcg(jobs+demote+i));
  asc_syncthreads();
  return {demote,promote,true};
}

// The final plan is frozen on worker 0. Do not read the touched bitmap here:
// other final workers can still be publishing their descriptor slot bits.
__simt_callee__ inline bool ExchangeIndexTail(__gm__ Context* ctx, uint32_t b, __ubuf__ uint32_t* scratch, bool planDelta=false) {
  using namespace contract_simt;
  auto c=ContractOf(ctx);
  auto state=ContractPtr<int32_t>(IndexBase(c,b));
  uint32_t tid=threadIdx.x, length=ContractLength(c,b), budget=ContractBudget(c,b), pool=ContractPool(c,b);
  if(!asc_ldcg(state) || asc_ldcg(state+1)!=int32_t(length) || asc_ldcg(state+4)!=int32_t(pool) ||
     asc_ldcg(state+5)!=int32_t(c->layer_id) || asc_ldcg(state+6)!=int32_t(budget)) return false;
  uint32_t oldh=uint32_t(asc_ldcg(state+2)), oldc=uint32_t(asc_ldcg(state+3));
  auto hi=IndexIds(c,b,true), ci=IndexIds(c,b,false), pos=IndexPositions(c,b);
  auto forward=ContractPtr<int32_t>(c->forward)+uint64_t(pool)*c->candidate_capacity;
  auto reverse=ContractPtr<int32_t>(c->reverse)+uint64_t(pool)*c->resident_capacity;
  auto target=ContractPtr<int32_t>(c->next_reverse)+b*c->resident_capacity;
  auto incoming=scratch, outgoing=scratch+kIndexDeltaCapacity, scan=scratch+2*kIndexDeltaCapacity;
  IndexChanges changes{0,0,false};
  bool direct=false;
  if(planDelta && IndexOwned(c,b)) {
    changes=LoadIndexPlanRecords(ctx,b,scratch);direct=changes.valid;
    if(!changes.valid)changes=BuildIndexChanges(ctx,b,scratch);
  }
  uint32_t demote=changes.demote,promote=changes.promote,count=0,rank=0;
  ContractPrefix prefix{0,0};
  if(!changes.valid) {
  for(uint32_t s=tid;s<budget;s+=256) {
    int32_t t=asc_ldcg(reverse+s);
    asc_stcg(target+s,t>=0 && uint32_t(t)<length && asc_ldcg(forward+t)==int32_t(s)?t:-1);
  }
  asc_threadfence(); asc_syncthreads();
  uint32_t reserved=ContractPtr<Row>(ctx->rows)[b].slots;
  for(uint32_t i=tid;i<reserved;i+=256)
    asc_stcg(target+ContractPtr<int32_t>(c->reserved_slots)[b*2048+i],-1);
  asc_threadfence(); asc_syncthreads();
  uint32_t planned=ContractPtr<int32_t>(c->plan_count)[b];
  for(uint32_t i=tid;i<planned;i+=256)
    asc_stcg(target+ContractPtr<int32_t>(c->planned_slots)[b*kSlots+i],
             ContractPtr<int32_t>(c->planned_tokens)[b*kSlots+i]);
  for(uint32_t i=tid;i<reserved;i+=256) {
    int32_t s=ContractPtr<int32_t>(c->extra_prefetch_slots)[b*2048+i];
    if(s>=0) asc_stcg(target+s,ContractPtr<int32_t>(c->reserved_tokens)[b*2048+i]);
  }
  asc_threadfence(); asc_syncthreads();
  // Membership is token-based, not old KV-slot-based. This also covers an
  // external resident restore that moves an existing hot token to a new slot.
  auto retainedHot=scratch; // Reuse incoming list storage until demotions are frozen.
  for(uint32_t i=tid;i<oldh;i+=256) retainedHot[i]=0;
  asc_threadfence(); asc_syncthreads();
  for(uint32_t s=tid;s<budget;s+=256) {
    int32_t t=asc_ldcg(target+s);
    if(t>=0) {
      int32_t at=asc_ldcg(pos+t);
      if(at<=-2) retainedHot[-at-2]=1;
      asc_stcg(target+s,at>=0?at:-1);
    }
  }
  asc_threadfence(); asc_syncthreads();
  count=0;
  for(uint32_t i=tid;i<oldh;i+=256) count+=!retainedHot[i];
  prefix=ContractPrefixUB(count,scan);
  demote=prefix.total; rank=prefix.begin;
  if(demote>kIndexDeltaCapacity) return false;
  for(uint32_t i=tid;i<oldh;i+=256) if(!retainedHot[i]) outgoing[rank++]=i;
  asc_syncthreads(); // All retained flags consumed before incoming aliases them.
  count=0;
  for(uint32_t s=tid;s<budget;s+=256) {
    int32_t t=asc_ldcg(target+s);
    count+=t>=0;
  }
  prefix=ContractPrefixUB(count,scan);
  promote=prefix.total; rank=prefix.begin;
  if(promote>kIndexDeltaCapacity) return false;
  for(uint32_t s=tid;s<budget;s+=256) {
    int32_t t=asc_ldcg(target+s);
    if(t>=0) incoming[rank++]=uint32_t(t);
  }
  asc_syncthreads();
  }
  uint32_t nh=oldh-demote+promote, nc=length-nh, copies=0;
  if(demote==promote) {
    // Independent pairs: no sorting, no source cache, no payload workspace.
    for(uint32_t j=tid;j<promote*32;j+=256) {
      uint32_t h=outgoing[j/32], d=incoming[j/32], w=j%32;
      auto hp=IndexRowKey(c,b,true,h)+w, cp=IndexRowKey(c,b,false,d)+w;
      uint32_t hv=asc_ldcg(hp), cv=asc_ldcg(cp);
      asc_stcg(hp,cv); asc_stcg(cp,hv);
      if(!w) {
        hp=IndexRowScale(c,b,true,h); cp=IndexRowScale(c,b,false,d);
        hv=asc_ldcg(hp); cv=asc_ldcg(cp); asc_stcg(hp,cv); asc_stcg(cp,hv);
      }
    }
    asc_threadfence(); asc_syncthreads();
    for(uint32_t i=tid;i<promote;i+=256) {
      uint32_t h=outgoing[i], d=incoming[i];
      int32_t ht=asc_ldcg(hi+h), ct=asc_ldcg(ci+d);
      asc_stcg(hi+h,ct); asc_stcg(ci+d,ht);
      asc_stcg(pos+ct,-int32_t(h)-2); asc_stcg(pos+ht,int32_t(d));
    }
    copies=2*promote;
  } else {
    // 1. Preserve demotions at the cold end (oldc + demote <= length).
    for(uint32_t j=tid;j<demote*32;j+=256)
      MoveIndexRow(c,b,true,outgoing[j/32],false,oldc+j/32,j%32);
    asc_threadfence(); asc_syncthreads();
    for(uint32_t i=tid;i<demote;i+=256) {
      int32_t t=asc_ldcg(hi+outgoing[i]);
      asc_stcg(ci+oldc+i,t); asc_stcg(pos+t,int32_t(oldc+i));
    }
    asc_threadfence(); asc_syncthreads();
    // 2. Compact hot survivors. jobs has S+C entries; these two lists together
    // have at most S entries, so no capacity proportional payload scratch.
    auto jobs=pos+c->candidate_capacity;
    uint32_t retained=oldh-demote;
    count=0;
    for(uint32_t i=tid;i<demote;i+=256) count+=outgoing[i]<retained;
    prefix=ContractPrefixUB(count,scan); rank=prefix.begin;
    uint32_t hotMoves=prefix.total;
    for(uint32_t i=tid;i<demote;i+=256) if(outgoing[i]<retained) asc_stcg(jobs+rank++,int32_t(outgoing[i]));
    count=0;
    for(uint32_t i=retained+tid;i<oldh;i+=256) count+=asc_ldcg(pos+asc_ldcg(hi+i))<=-2;
    prefix=ContractPrefixUB(count,scan); rank=prefix.begin;
    for(uint32_t i=retained+tid;i<oldh;i+=256)
      if(asc_ldcg(pos+asc_ldcg(hi+i))<=-2) asc_stcg(jobs+hotMoves+rank++,int32_t(i));
    asc_threadfence(); asc_syncthreads();
    for(uint32_t j=tid;j<hotMoves*32;j+=256)
      MoveIndexRow(c,b,true,uint32_t(asc_ldcg(jobs+hotMoves+j/32)),true,uint32_t(asc_ldcg(jobs+j/32)),j%32);
    asc_threadfence(); asc_syncthreads();
    for(uint32_t i=tid;i<hotMoves;i+=256) {
      int32_t d=asc_ldcg(jobs+i), t=asc_ldcg(hi+asc_ldcg(jobs+hotMoves+i));
      asc_stcg(hi+d,t); asc_stcg(pos+t,-d-2);
    }
    asc_threadfence(); asc_syncthreads();
    // 3. Append promoted keys only after the old hot tail has been consumed.
    for(uint32_t j=tid;j<promote*32;j+=256)
      MoveIndexRow(c,b,false,incoming[j/32],true,retained+j/32,j%32);
    asc_threadfence(); asc_syncthreads();
    for(uint32_t i=tid;i<promote;i+=256) {
      int32_t t=asc_ldcg(ci+incoming[i]);
      asc_stcg(hi+retained+i,t); asc_stcg(pos+t,-int32_t(retained+i)-2);
    }
    asc_threadfence(); asc_syncthreads();
    // 4. Fill cold holes from the now-invalid cold suffix. Sources >= nc;
    // destinations < nc. A promoted suffix row is excluded via pos[].
    count=0;
    for(uint32_t i=tid;i<promote;i+=256) count+=incoming[i]<nc;
    prefix=ContractPrefixUB(count,scan); rank=prefix.begin;
    uint32_t coldMoves=prefix.total;
    for(uint32_t i=tid;i<promote;i+=256) if(incoming[i]<nc) outgoing[rank++]=incoming[i];
    asc_syncthreads();
    count=0;
    for(uint32_t i=nc+tid;i<oldc+demote;i+=256) count+=asc_ldcg(pos+asc_ldcg(ci+i))>=0;
    prefix=ContractPrefixUB(count,scan); rank=prefix.begin;
    for(uint32_t i=nc+tid;i<oldc+demote;i+=256)
      if(asc_ldcg(pos+asc_ldcg(ci+i))>=0) incoming[rank++]=i;
    asc_syncthreads();
    for(uint32_t j=tid;j<coldMoves*32;j+=256)
      MoveIndexRow(c,b,false,incoming[j/32],false,outgoing[j/32],j%32);
    asc_threadfence(); asc_syncthreads();
    for(uint32_t i=tid;i<coldMoves;i+=256) {
      int32_t t=asc_ldcg(ci+incoming[i]);
      asc_stcg(ci+outgoing[i],t); asc_stcg(pos+t,int32_t(outgoing[i]));
    }
    copies=demote+hotMoves+promote+coldMoves;
  }
  asc_threadfence(); asc_syncthreads();
  if(!tid) {
    asc_stcg(state+2,int32_t(nh)); asc_stcg(state+3,int32_t(nc));
    asc_stcg(state+7,int32_t(copies)); asc_stcg(state+17,direct?5:(changes.valid?4:3));
    asc_stcg(state,0); // CQ publication is the only commit point.
  }
  asc_threadfence();
  return true;
}
} // namespace rotating
