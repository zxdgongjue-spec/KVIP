#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace li_qk {
struct Task { uint32_t queryBegin, queryCount, begin, end; };
struct Plan {
    uint32_t history, heads, historyParts, queryParts, queryGroup, physicalGroup, capacity, stride;
    std::vector<Task> tasks;
};
inline Plan MakePlan(uint32_t history,uint32_t heads,uint32_t aics,uint32_t aivs,
                     uint32_t historyParts,uint32_t queryParts) {
    if (!history || history%256 || history>210944 || history<2048 ||
        heads!=32 || !aics || aics>32 || aivs<2 || aivs>64 ||
        !(historyParts==1 || historyParts==2 || historyParts==4 || historyParts==8 || historyParts==16) ||
        !(queryParts==1 || queryParts==2 || queryParts==4) || history/256<historyParts)
        throw std::invalid_argument("LI QK: B1/Q4, G32, aligned H, T=1/2/4/8/16, Qparts=1/2/4 required");
    uint32_t group=4/queryParts;
    Plan p{history,heads,historyParts,queryParts,group,std::max(2U,group),std::min(aics,aivs/2),0,{}};
    for(uint32_t q=0;q<4;q+=group)for(uint32_t t=0;t<historyParts;++t) {
        uint32_t lo=history/256*t/historyParts*256,hi=history/256*(t+1)/historyParts*256;
        p.tasks.push_back({q,group,lo,hi});p.stride=std::max(p.stride,hi-lo);
    }
    return p;
}
}
