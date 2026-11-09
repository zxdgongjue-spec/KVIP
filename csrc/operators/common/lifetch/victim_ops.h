// Shared LIFetch victim handling.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void GatherVictimScores(__ubuf__ uint32_t* ids,__ubuf__ uint32_t* output,
    const __gm__ uint32_t* scores) {
    auto input=const_cast<__gm__ uint32_t*>(scores);
    for(uint32_t i=threadIdx.x;i<1024;i+=256) {
        uint32_t a=ids[2*i],b=ids[2*i+1];
        uint32_t low=(asc_ldcg(input+a/2)>>((a%2)*16))&0xffffU;
        uint32_t high=(asc_ldcg(input+b/2)>>((b%2)*16))&0xffffU;
        output[i]=low|(high<<16);
    }
}

// Shared LIFetch victim handling.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void RestoreVictimTokens(__ubuf__ uint32_t* ids,__gm__ uint32_t* victims,uint32_t count) {
    for(uint32_t i=threadIdx.x;i<count;i+=256) {
        uint32_t token=ids[i];
        if(token<kLruSlots)ids[i]=asc_ldcg(victims+token);
    }
}
