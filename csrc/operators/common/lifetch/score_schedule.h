#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace kvip_score {
struct Task { uint32_t owner, queryBegin, queryCount, begin, end; };
struct Plan {
    uint32_t batch, history, heads, producers, queryGroup, capacity, rowStride;
    std::vector<Task> tasks;
};
inline Plan MakePlan(uint32_t batch,uint32_t history,uint32_t heads,
                     uint32_t cube,uint32_t vector,uint32_t producers,uint32_t queryGroup) {
    if (!batch || batch>32 || !history || history%256 || history>210944 ||
        (heads!=32 && heads!=64) || !cube || cube>32 || vector<2 || vector>64 ||
        !producers || producers>32 || history/256<producers ||
        (queryGroup!=2 && queryGroup!=4) || (producers>1 && queryGroup!=4))
        throw std::invalid_argument("Unsupported score-only geometry");
    Plan p{batch,history,heads,producers,queryGroup,std::min(cube,vector/2),0,{}};
    for(uint32_t b=0;b<batch;++b)for(uint32_t q=0;q<4;q+=queryGroup)
        for(uint32_t s=0;s<producers;++s) {
            uint32_t lo=history/256*s/producers*256,hi=history/256*(s+1)/producers*256;
            p.tasks.push_back({b,q,queryGroup,lo,hi});
            p.rowStride=std::max(p.rowStride,hi-lo);
        }
    return p;
}
// Occupancy recommendation only. Measured crossover selection is separate.
inline uint32_t OccupancyProducers(uint32_t batch,uint32_t history,uint32_t cube,uint32_t vector,
                                   uint32_t minTokens=4096) {
    if(!batch || !minTokens)throw std::invalid_argument("Positive occupancy parameters required");
    return std::max(1U,std::min({32U,std::min(cube,vector/2)/batch,history/minTokens}));
}
}
