#pragma once
namespace rotating {
// Owned mode does not use the derived cache's dirty-row arrays. Reuse them
// for an exact normalized resident snapshot (S words) and H0 (ceil(C/32)
// words). No extra allocation or payload mirror. Header 20 is the commit bit;
// 21..26 bind the snapshot to length, budget, pool, layer, format and hot count.
// All cache-layout writers must invalidate 20. Payload-only writes need not.
__simt_callee__ inline bool ReuseOwnedH0(__gm__ Context* ctx, uint32_t b,
                                        __ubuf__ uint32_t* scratch) {
  using namespace contract_simt;
  auto c=ContractOf(ctx);
  auto state=ContractPtr<int32_t>(IndexBase(c,b));
  uint32_t length=ContractLength(c,b),budget=ContractBudget(c,b),pool=ContractPool(c,b);
  bool ready=asc_ldcg(state)==1 && asc_ldcg(state+20)==1 &&
      asc_ldcg(state+21)==int32_t(length) && asc_ldcg(state+22)==int32_t(budget) &&
      asc_ldcg(state+23)==int32_t(pool) && asc_ldcg(state+24)==int32_t(c->layer_id) &&
      asc_ldcg(state+25)==1 && asc_ldcg(state+26)==asc_ldcg(state+2) &&
      asc_ldcg(state+1)==int32_t(length) && asc_ldcg(state+4)==int32_t(pool) &&
      asc_ldcg(state+5)==int32_t(c->layer_id) && asc_ldcg(state+6)==int32_t(budget);
  if(!ready) {
    if(!threadIdx.x)asc_stcg(state+20,0);
    asc_threadfence();asc_syncthreads();return false;
  }
  auto reverse=ContractPtr<int32_t>(c->reverse)+uint64_t(pool)*c->resident_capacity;
  auto forward=ContractPtr<int32_t>(c->forward)+uint64_t(pool)*c->candidate_capacity;
  auto snapshot=IndexDirty(c,b,true);
  // An explicit positive writer epoch certifies exclusive ownership. Default
  // zero retains exact validation for unversioned/framework external writers.
  int32_t epoch=asc_ldcg(ContractPtr<int32_t>(c->mapping_epoch)+b);
  bool versioned=epoch>0 && asc_ldcg(state+31)==epoch;
  uint32_t changed=0;
  if(!versioned)for(uint32_t s=threadIdx.x;s<budget;s+=256) {
    int32_t t=asc_ldcg(reverse+s);
    int32_t normalized=t>=0 && uint32_t(t)<length && asc_ldcg(forward+t)==int32_t(s)?t:-1;
    changed+=normalized!=asc_ldcg(snapshot+s);
  }
  auto sum=ContractPrefixUB(changed,scratch);
  if(sum.total) {
    if(!threadIdx.x)asc_stcg(state+20,0);
    asc_threadfence();asc_syncthreads();return false;
  }
  uint32_t words=(c->candidate_capacity+31)/32;
  auto saved=IndexDirty(c,b,false);
  auto hot=ContractPtr<uint32_t>(c->hot_bitmap)+b*words;
  for(uint32_t i=threadIdx.x;i<words;i+=256)asc_stcg(hot+i,uint32_t(asc_ldcg(saved+i)));
  int32_t count=asc_ldcg(state+26);
  if(!count)for(uint32_t i=threadIdx.x;i<2048;i+=256)
    asc_stcg(ContractPtr<int32_t>(c->eligible_victims)+b*2048+i,ContractPtr<int32_t>(c->victims)[b*2048+i]);
  asc_threadfence();asc_syncthreads();
  if(!threadIdx.x) {
    asc_stcg(state+31,epoch);
    asc_stcg(state+19,1);
    asc_stcg(ContractPtr<int32_t>(c->hot_counts)+b,count);
    asc_stcg(ContractPtr<uint32_t>(ctx->rowStatus)+b*kRowStatusWords+29,versioned?2U:1U);
    if(!count) {
      asc_stcg(ContractPtr<int32_t>(c->eligible_counts)+b,2048);
      asc_stcg(ContractPtr<int32_t>(c->hot_ready)+b*32+1,1);
    }
  }
  asc_threadfence();return true;
}

// Called only after successful init reconciliation, with the complete H0.
__simt_callee__ inline void SnapshotOwnedH0(__gm__ Context* ctx,uint32_t b) {
  using namespace contract_simt;
  auto c=ContractOf(ctx);
  auto state=ContractPtr<int32_t>(IndexBase(c,b));
  if(asc_ldcg(state+20))return;
  uint32_t length=ContractLength(c,b),budget=ContractBudget(c,b),pool=ContractPool(c,b);
  auto reverse=ContractPtr<int32_t>(c->reverse)+uint64_t(pool)*c->resident_capacity;
  auto forward=ContractPtr<int32_t>(c->forward)+uint64_t(pool)*c->candidate_capacity;
  auto snapshot=IndexDirty(c,b,true);
  for(uint32_t s=threadIdx.x;s<budget;s+=256) {
    int32_t t=asc_ldcg(reverse+s);
    asc_stcg(snapshot+s,t>=0 && uint32_t(t)<length && asc_ldcg(forward+t)==int32_t(s)?t:-1);
  }
  uint32_t words=(c->candidate_capacity+31)/32;
  auto hot=ContractPtr<uint32_t>(c->hot_bitmap)+b*words;
  auto saved=IndexDirty(c,b,false);
  for(uint32_t i=threadIdx.x;i<words;i+=256)asc_stcg(saved+i,int32_t(asc_ldcg(hot+i)));
  asc_threadfence();asc_syncthreads();
  if(!threadIdx.x) {
    asc_stcg(state+21,int32_t(length));asc_stcg(state+22,int32_t(budget));
    asc_stcg(state+23,int32_t(pool));asc_stcg(state+24,int32_t(c->layer_id));
    asc_stcg(state+31,asc_ldcg(ContractPtr<int32_t>(c->mapping_epoch)+b));
    asc_stcg(state+25,1);asc_stcg(state+26,asc_ldcg(state+2));
    asc_threadfence();asc_stcg(state+20,1);
  }
  asc_threadfence();
}
} // namespace rotating
