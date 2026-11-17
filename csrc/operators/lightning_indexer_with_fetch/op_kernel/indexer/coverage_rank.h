#pragma once
#include <cstdint>
#if defined(__CCE_AICORE__)
#define KVIP_COVERAGE_DEVICE __aicore__
#else
#define KVIP_COVERAGE_DEVICE
#endif
namespace kvip_coverage {
// Count one producer's alternating cold half-windows below an absolute cursor.
KVIP_COVERAGE_DEVICE inline uint32_t Shard(uint32_t x,uint32_t half,uint32_t shard) {
    uint32_t pairs=x/(2*half), remainder=x%(2*half), offset=shard*half;
    return pairs*half+(remainder>offset?(remainder-offset<half?remainder-offset:half):0);
}
KVIP_COVERAGE_DEVICE inline uint32_t Covered(uint32_t end0,uint32_t end1,
    uint32_t resident,uint32_t valid,uint32_t half,uint32_t tailBegin,uint32_t tailEnd) {
    uint32_t hot=resident<valid?resident:valid;
    if(tailBegin>valid)tailBegin=valid;
    if(tailBegin<hot)tailBegin=hot;
    if(tailEnd>valid)tailEnd=valid;
    if(tailEnd<tailBegin)tailEnd=tailBegin;
    uint32_t total=hot+tailEnd-tailBegin;
    for(uint32_t shard=0;shard<(half?2U:1U);++shard) {
        uint32_t end=shard?end1:end0;if(end>valid)end=valid;if(end<hot)end=hot;
        uint32_t n=end-hot,lo=tailBegin<end?tailBegin:end,hi=tailEnd<end?tailEnd:end;
        if(half)total+=Shard(n,half,shard)-(Shard(hi-hot,half,shard)-Shard(lo-hot,half,shard));
        else total+=n-(hi-lo);
    }
    return total;
}
KVIP_COVERAGE_DEVICE inline uint32_t Rank(uint32_t covered,uint32_t initial,uint32_t valid) {
    if(valid<=initial)return 1024;
    uint32_t done=covered>initial?covered-initial:0, remaining=valid-initial;
    if(done>remaining)done=remaining;
    return 1024+(uint64_t(done)*1024+remaining/2)/remaining;
}
}
#undef KVIP_COVERAGE_DEVICE
