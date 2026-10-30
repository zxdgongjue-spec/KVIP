#pragma once
namespace rotating {
struct ContractPrefix { uint32_t begin, total; };
// All 256 lanes participate. Eight shared words hold warp totals; each caller
// owns its scratch until the following block barrier.
#define V9_PREFIX_BODY \
    uint32_t tid=threadIdx.x,lane=tid%warpSize,warp=tid/warpSize,value=count; \
    for(uint32_t d=1;d<warpSize;d*=2) { uint32_t x=asc_shfl_up(value,d); if(lane>=d)value+=x; } \
    if(lane==warpSize-1)asc_stcg(scratch+warp,int32_t(value)); \
    asc_threadfence(); asc_syncthreads(); \
    uint32_t before=0,total=0; \
    for(uint32_t w=0;w<256/warpSize;++w) { uint32_t x=uint32_t(asc_ldcg(scratch+w)); total+=x; if(w<warp)before+=x; } \
    asc_syncthreads(); \
    return {before+value-count,total};
__simt_callee__ inline ContractPrefix ContractPrefixGM(uint32_t count,__gm__ int32_t* scratch) {
    V9_PREFIX_BODY
}
// UB has no cache policy: ordinary loads/stores are used for its warp totals.
__simt_callee__ inline ContractPrefix ContractPrefixUB(uint32_t count,__ubuf__ uint32_t* scratch) {
    uint32_t tid=threadIdx.x,lane=tid%warpSize,warp=tid/warpSize,value=count;
    for(uint32_t d=1;d<warpSize;d*=2) { uint32_t x=asc_shfl_up(value,d); if(lane>=d)value+=x; }
    if(lane==warpSize-1)scratch[warp]=value;
    asc_syncthreads();
    uint32_t before=0,total=0;
    for(uint32_t w=0;w<256/warpSize;++w) { uint32_t x=scratch[w]; total+=x; if(w<warp)before+=x; }
    asc_syncthreads();
    return {before+value-count,total};
}
#undef V9_PREFIX_BODY
}
