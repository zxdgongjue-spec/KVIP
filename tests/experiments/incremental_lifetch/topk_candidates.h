// Included inside rotating. The hot window keeps its original bitmap path.
// Middle windows consume only compact cold IDs exported by incremental TopK.
__simd_vf__ inline void CompactColdTopk(__ubuf__ uint32_t* input,
    __ubuf__ uint32_t* output,__ubuf__ uint32_t* count,uint32_t resident,uint32_t valid) {
    using namespace MicroAPI;
    auto all=CreateMask<uint32_t,MaskPattern::ALL>();
    auto one=CreateMask<uint32_t,MaskPattern::VL1>();
    RegTensor<uint32_t> ids,packed,ones,total,sum;
    MaskReg cold,bounded,take;UnalignRegForStore store;
    Duplicate(sum,uint32_t(0));ClearSpr<SpecialPurposeReg::AR>();
    for(uint16_t i=0;i<2048;i+=64) {
        LoadAlign(ids,input+i);
        CompareScalar<uint32_t,CMPMODE::GE>(cold,ids,resident,all);
        CompareScalar<uint32_t,CMPMODE::LT>(bounded,ids,valid,all);
        MaskAnd(take,cold,bounded,all);
        Duplicate<uint32_t,MaskMergeMode::ZEROING>(ones,uint32_t(1),take);
        Add(sum,sum,ones,all);
        Squeeze<uint32_t,MicroAPI::GatherMaskMode::STORE_REG>(packed,ids,take);
        StoreUnAlign<uint32_t,PostLiteral::POST_MODE_UPDATE>(output,packed,store);
    }
    StoreUnAlignPost(output,store);ReduceSum(total,sum,all);StoreAlign(count,total,one);
}

// One UB bit per token gives exact deduplication. Expand only newly inserted
// compact IDs, never scan the history-sized set. Each warp reserves output once.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void UnionColdTopk(__ubuf__ uint32_t* input,__ubuf__ uint32_t* bits,
    __ubuf__ uint32_t* ids,__ubuf__ uint32_t* cfg,uint32_t stride,
    uint32_t n0,uint32_t n1,uint32_t n2,uint32_t n3) {
    uint32_t tid=threadIdx.x,lane=tid%32;
    for(uint32_t i=tid;i<4*stride;i+=256) {
        uint32_t q=i/stride,j=i%stride,n=q==0?n0:q==1?n1:q==2?n2:n3;
        bool fresh=false;uint32_t token=0;
        if(j<n) {
            token=input[i];uint32_t bit=1U<<(token%32);
            fresh=(asc_atomic_or(bits+token/32,bit)&bit)==0;
        }
        uint32_t mask=asc_ballot(fresh),base=0;
        if(!lane && mask)base=asc_atomic_add(cfg+40,uint32_t(__popc(mask)));
        base=asc_shfl(base,0);
        uint32_t rank=base+__popc(mask&((1U<<lane)-1));
        if(fresh && rank<kCandidateCapacity)ids[rank]=token;
    }
    asc_syncthreads();
}
