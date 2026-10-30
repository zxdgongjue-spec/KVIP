#pragma once
namespace rotating {
__simt_callee__ inline void StageResidentDestination(__gm__ Contract* c,uint32_t b,uint32_t w,
    __ubuf__ uint32_t* selected,__ubuf__ uint32_t* locations,__ubuf__ uint32_t* scan) {
  using namespace contract_simt;
  uint32_t tid=threadIdx.x;
  for(uint32_t q=w;q<4;q+=kFinalWorkers) {
    auto output=ContractPtr<int32_t>(c->staged_destination)+(b*4+q)*2048;
    uint32_t validMask=0,coldMask=0,valid=0,cold=0;
    for(uint32_t j=0;j<8;++j) {
      uint32_t i=q*2048+tid*8+j;
      asc_stcg(output+tid*8+j,-1);
      if(locations[i]!=UINT32_MAX) {
        validMask|=1U<<j;++valid;
        // The hot pass erased selected[i]; cold entries retain their token.
        if(selected[i]!=UINT32_MAX){coldMask|=1U<<j;++cold;}
      }
    }
    auto prefix=ContractPrefixUB(cold+(valid<<16),scan);
    uint32_t coldRank=prefix.begin&0xffffU;
    uint32_t hotRank=(prefix.total&0xffffU)+(prefix.begin>>16)-coldRank;
    asc_threadfence();asc_syncthreads();
    for(uint32_t j=0;j<8;++j)if(validMask&(1U<<j)) {
      uint32_t dst=(coldMask&(1U<<j))?coldRank++:hotRank++;
      asc_stcg(output+dst,int32_t(locations[q*2048+tid*8+j]));
    }
    asc_threadfence();asc_syncthreads();
    if(!tid)asc_atomic_or(ContractPtr<uint32_t>(c->publish_meta)+b*4,1U<<q);
  }
  asc_threadfence();asc_syncthreads();
}
} // namespace rotating
