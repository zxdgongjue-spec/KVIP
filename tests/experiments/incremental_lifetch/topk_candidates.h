// Included inside rotating. Reuse service scratch: 16 KiB union + 8 KiB Q IDs.
// Filter local/already-posted tokens before UB atomics; no GM atomic operations.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void BuildHotEligibility(__ubuf__ uint32_t* valid,__ubuf__ uint32_t* reserved,
    __ubuf__ uint32_t* eligible,uint32_t first) {
    uint32_t tid=threadIdx.x;
    // Eight lanes form one 32-token word; each lane reads four contiguous slots.
    for(uint32_t base=tid*4;base<2048;base+=1024) {
        uint32_t bits=0;
        for(uint32_t i=0;i<4;++i)if(!valid[base+i] || int32_t(reserved[base+i])>=0)bits|=1U<<i;
        bits<<=(tid%8)*4;
        for(uint32_t delta=4;delta;delta/=2)bits|=asc_shfl_down(bits,delta);
        if(tid%8==0)eligible[(first+base)/32]=bits;
    }
    asc_syncthreads();
}
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void UnionTopkCandidates(__ubuf__ uint32_t* selected,__ubuf__ uint32_t* bits,
    __gm__ Context* ctx,__ubuf__ uint32_t* eligible,uint32_t b,uint32_t q,uint32_t history,uint32_t resident) {
    auto map=reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+uint64_t(b)*history;
    uint32_t valid=QueryEnd(history,ctx,b*4+q);
    for(uint32_t i=threadIdx.x;i<2048;i+=256) {
        uint32_t token=selected[i];
        bool take=token<valid && token<history;
        if(take && token<resident)take=(eligible[token/32]&(1U<<(token%32)))!=0;
        if(take)take=asc_ldcg(map+token)==UINT32_MAX;
        uint32_t word=token/32,bit=take?(1U<<(token%32)):0;
        // Dense adjacent IDs otherwise serialize 32 CAS loops on one UB word.
        // Collapse that common case within the warp before the single atomic.
        uint32_t firstWord=asc_shfl(word,0);
        bool oneWord=asc_ballot(word==firstWord)==UINT32_MAX;
        if(oneWord) {
            for(uint32_t delta=16;delta;delta/=2)bit|=asc_shfl_xor(bit,delta);
            if(threadIdx.x%32==0 && bit)asc_atomic_or(bits+firstWord,bit);
        } else if(take) {
            asc_atomic_or(bits+word,bit);
        }
    }
    asc_syncthreads();
}
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void ExpandTopkUnion(__ubuf__ uint32_t* bits,__ubuf__ uint32_t* ids,
    __ubuf__ uint32_t* cfg,uint32_t words) {
    uint32_t tid=threadIdx.x,lane=tid%warpSize,warp=tid/warpSize;
    // Stride words across lanes: clustered candidates must not leave only
    // a few lanes expanding hundreds of tokens while the rest are idle.
    uint32_t each=(words+255)/256,count=0;
    for(uint32_t i=0;i<each;++i)if(tid+i*256<words)count+=__popc(bits[tid+i*256]);
    uint32_t inclusive=count;
    for(uint32_t delta=1;delta<warpSize;delta*=2) {
        uint32_t prior=asc_shfl_up(inclusive,delta);if(lane>=delta)inclusive+=prior;
    }
    if(lane==warpSize-1)cfg[64+warp]=inclusive;
    asc_syncthreads();
    if(!tid) {
        uint32_t total=0;
        for(uint32_t w=0;w<256/warpSize;++w){uint32_t n=cfg[64+w];cfg[64+w]=total;total+=n;}
        cfg[40]=total<kCandidateCapacity?total:kCandidateCapacity;
    }
    asc_syncthreads();
    uint32_t rank=cfg[64+warp]+inclusive-count;
    for(uint32_t i=0;i<each;++i) {
        uint32_t word=tid+i*256,value=word<words?bits[word]:0;
        while(value && rank<kCandidateCapacity){ids[rank++]=word*32+uint32_t(__ffs(int32_t(value))-1);value&=value-1;}
    }
    asc_syncthreads();
}
