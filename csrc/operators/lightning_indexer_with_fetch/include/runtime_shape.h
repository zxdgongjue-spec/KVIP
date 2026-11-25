#pragma once
#include <stdint.h>
// Immutable per-launch geometry, prepared before graph capture. No device discovery here.
namespace rotating {
// BSND decode carries four query positions; the three lookahead positions
// share the model-length row, padded to the streaming 256-token alignment.
constexpr uint32_t kMaxSequenceTokens = 512U * 1024;
constexpr uint32_t kMaxHistoryTokens = (kMaxSequenceTokens + 3U + 255U) / 256U * 256U;
struct RuntimeShape {
    uint32_t batch;
    uint32_t history;
    uint32_t resident;
    uint32_t indexHeads;
    uint32_t aicCount;
    uint32_t splitQ;
    uint32_t liCores;
    uint32_t rounds;
    uint32_t serviceAics;
    uint32_t serviceLanes;
    uint32_t finalWorkers;
    uint32_t finalCores;
    uint32_t finalPostOffset;
    uint32_t finalOwnerBytes;
    uint32_t tokenBytes;
    uint32_t postGroups;
    uint32_t postThreads;
    uint32_t postBank;
    uint32_t batchedRows;
    uint32_t scanTokens;
};
static_assert(sizeof(RuntimeShape)==80,"Runtime shape ABI");
inline bool NativeLargeBatch(uint32_t batch, uint32_t aicCount) {
    return aicCount == 32 && (batch == 20 || batch == 24 || batch == 28 || batch == 32);
}
// Same geometry formulas formerly evaluated by CMake. Called on the host once.
inline uint32_t PrepareShape(RuntimeShape* s) {
    if (!s || !s->batch || s->batch>32 || !s->aicCount || s->aicCount>32 ||
        s->batch>s->aicCount || (s->indexHeads!=32 && s->indexHeads!=64) ||
        (s->tokenBytes!=656 && s->tokenBytes!=2624) || s->history%256 ||
        s->resident%256 || s->resident<2048 || s->resident>131072 || s->history<=s->resident ||
        s->history>kMaxHistoryTokens || !s->rounds || s->rounds>32 || s->splitQ>1) return 100000;
    // The large Q4 terminal runs complete four-lane teams in two waves.
    // Keep the two queue sets per row plus three private final Q queues.
    if (NativeLargeBatch(s->batch, s->aicCount) && s->indexHeads == 32 &&
        (s->finalWorkers == 4 || (!s->finalWorkers && s->splitQ))) {
        if (s->rounds < 2 || (s->history - s->resident) / 128 <= s->rounds) return 100000;
        s->liCores = s->aicCount;
        s->splitQ = 0;
        s->serviceAics = s->batch / 2;
        s->serviceLanes = s->batch;
        s->finalWorkers = 4;
    } else {
        s->splitQ = s->splitQ && 2*s->batch<=s->aicCount;
        s->liCores = s->batch*(1+s->splitQ);
        if(s->rounds>s->liCores)s->rounds=s->liCores;
        if((s->history-s->resident)/128<=s->rounds)return 100000;
        if(!s->serviceAics)s->serviceAics=s->liCores/s->rounds;
        if(s->serviceAics>s->liCores/s->rounds)return 100000;
        s->serviceLanes=2*s->serviceAics;
        if(!s->finalWorkers)s->finalWorkers=s->splitQ?4:2;
        if((s->finalWorkers!=1 && s->finalWorkers!=2 && s->finalWorkers!=4) ||
            s->finalWorkers*s->batch>2*s->liCores)return 100000;
    }
    s->finalCores=s->batch*s->finalWorkers;
    s->finalPostOffset=s->batch*128+s->finalCores*288*4;
    s->finalOwnerBytes=s->finalPostOffset+s->finalCores*128;
    s->postGroups=(s->batch+s->serviceLanes-1)/s->serviceLanes;
    s->postThreads=s->postGroups<=1?512:(s->postGroups<=2?256:(s->postGroups<=4?128:64));
    s->postBank=s->postGroups<=1?448:(s->postGroups<=2?240:(s->postGroups<=4?120:56));
    s->batchedRows=s->batch<=16 || (s->batch>16 && s->finalWorkers==4);
    s->scanTokens=s->batchedRows?23808:28672;
    if(s->batchedRows && (s->postGroups*1024>8448 ||
        ((s->postGroups-1)*s->postBank+((s->postBank+63)/64)*64)*64>32000))return 100000;
    return 0;
}
}
