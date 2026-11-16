#pragma once
#include <stdint.h>
// Internal opt-in sidecar. Context/RuntimeShape and their serialized ABI stay fixed.
namespace kvip_staged {
struct ScoreTask { uint32_t owner, queryBegin, begin, end; };
struct ScoreWave {
    uint64_t bindings, scores, tasks;
    uint32_t rows, stride, queryGroup, readyEnd;
};
struct Plan {
    uint64_t canonical;
    const ScoreWave* hot;
    const ScoreWave* cold;
    uint32_t hotCount, coldCount, aivBudget, fetchWorkers;
};
inline uint32_t OwnerCapacity(uint32_t v, uint32_t w) {
    if (v>64 || (w!=1 && w!=2 && w!=4) || v<=4*w) return 0;
    uint32_t n=(v-4*w)/3;
    return n<16?n:16;
}
// The opt-in staged path separates logical owners from resident workers.
// PrepareShape's batch<=C condition belongs to the original fused LI path.
template<class Shape> inline uint32_t PrepareShape(Shape* s,uint32_t v,uint32_t w) {
    if(!s || !s->batch || s->batch>32 || !s->aicCount || s->aicCount>32 ||
       !OwnerCapacity(v,w) || v<s->batch || v<2 || (s->indexHeads!=32 && s->indexHeads!=64) ||
       s->resident>131072 || s->history>210944 || s->history%256 || s->resident%256 || s->resident<2048 ||
       s->history<=s->resident || (s->tokenBytes!=656 && s->tokenBytes!=2624) ||
       !s->rounds || s->rounds>32 || (s->history-s->resident)/128<s->rounds-1) return 100000;
    s->splitQ=0; s->liCores=s->aicCount<v/2?s->aicCount:v/2;
    // Standalone Service uses a stable independent lane pool.
    s->serviceAics=1; s->serviceLanes=2;
    s->finalWorkers=4;s->finalCores=s->batch*4;
    s->finalPostOffset=s->batch*128+s->finalCores*288*4;
    s->finalOwnerBytes=s->finalPostOffset+s->finalCores*128;
    s->postGroups=(s->batch+1)/2;
    s->postThreads=s->postGroups<=1?512:(s->postGroups<=2?256:(s->postGroups<=4?128:64));
    s->postBank=s->postGroups<=1?448:(s->postGroups<=2?240:(s->postGroups<=4?120:56));
    s->batchedRows=s->batch<=16;s->scanTokens=s->batchedRows?23808:28672;
    return 0;
}
}
