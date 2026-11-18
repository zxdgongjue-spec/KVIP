#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace li_qk {
struct Task { uint32_t queryBegin, queryCount, begin, end, owner=0; };
struct Plan {
    uint32_t history, heads, historyParts, queryParts, queryGroup, physicalGroup, capacity, stride;
    std::vector<Task> tasks;
    uint32_t batch=1;
    bool balanced=false;
    uint32_t workGrain=2048;
    bool ownerBalanced=false;
};
inline Plan MakePlan(uint32_t history,uint32_t heads,uint32_t aics,uint32_t aivs,
                     uint32_t historyParts,uint32_t queryParts,uint32_t batch=1,bool balanced=false,
                     uint32_t workGrain=2048,bool shortScore=false,bool ownerBalanced=false) {
    if (!history || history%256 || history>210944 || (history<2048 && !(shortScore && balanced)) ||
        heads!=32 || !aics || aics>32 || aivs<2 || aivs>64 ||
        !(historyParts==1 || historyParts==2 || historyParts==4 || historyParts==8 || historyParts==16 || historyParts==32 || historyParts==30) ||
        !(queryParts==1 || queryParts==2 || queryParts==4) || (!balanced && history/256<historyParts) ||
        !(batch==1 || batch==16) || (batch==16 && (queryParts!=1 || historyParts>2)))
        throw std::invalid_argument("LI QK: B1 T<=32 or B16 T1/T2 Q4; G32 and aligned H required");
    uint32_t group=4/queryParts;
    Plan p{history,heads,historyParts,queryParts,group,std::max(2U,group),std::min(aics,aivs/2),0,{}};
    p.batch=batch;p.balanced=balanced;p.workGrain=workGrain;p.ownerBalanced=ownerBalanced;
    if(ownerBalanced && !balanced)throw std::invalid_argument("owner boundaries require balanced scheduling");
    if(!balanced && workGrain!=2048)throw std::invalid_argument("work grain applies to balanced score only");
    // Non-power-of-two T30 balances complete native H128 tiles. Preserve
    // the existing 256-token boundaries for the original controls.
    const uint32_t quantum=historyParts==30?128:256;
    for(uint32_t b=0;b<batch;++b)for(uint32_t q=0;q<4;q+=group)for(uint32_t t=0;t<historyParts;++t) {
        uint32_t lo=history/quantum*t/historyParts*quantum,hi=history/quantum*(t+1)/historyParts*quantum;
        p.tasks.push_back({q,group,lo,hi,b});p.stride=std::max(p.stride,hi-lo);
    }
    if(balanced) {
        if(batch!=16 || historyParts!=2 || queryParts!=1 || (p.capacity!=30 && p.capacity!=32) ||
           workGrain<128 || workGrain>65536 || (workGrain&(workGrain-1)) || history%(2*workGrain))
            throw std::invalid_argument("balanced requires B16/T2/Q4,30 or32 groups,power-of-two grain128..65536,H aligned to2*grain");
        // Equal budgets; the owner-only variant removes artificial half
        // boundaries while preserving real owner input/output ownership.
        const uint32_t domainsPerOwner=ownerBalanced?1:2;
        const uint32_t domainBlocks=history/(domainsPerOwner*workGrain),totalBlocks=batch*history/workGrain;
        std::vector<std::vector<Task>> cores(p.capacity);
        uint32_t slots=0;p.stride=0;
        for(uint32_t c=0;c<p.capacity;++c) {
            uint32_t lo=totalBlocks*c/p.capacity,hi=totalBlocks*(c+1)/p.capacity;
            while(lo<hi) {
                uint32_t logical=lo/domainBlocks,owner=logical/domainsPerOwner,domain=logical%domainsPerOwner;
                uint32_t end=std::min(hi,(logical+1)*domainBlocks);
                uint32_t beginToken=(domain*domainBlocks+lo%domainBlocks)*workGrain;
                uint32_t endToken=beginToken+(end-lo)*workGrain;
                cores[c].push_back({0,4,beginToken,endToken,owner});
                p.stride=std::max(p.stride,endToken-beginToken);lo=end;
            }
            slots=std::max(slots,uint32_t(cores[c].size()));
        }
        p.tasks.clear();
        for(uint32_t s=0;s<slots;++s)for(uint32_t c=0;c<p.capacity;++c)
            p.tasks.push_back(s<cores[c].size()?cores[c][s]:Task{0,0,0,0,0});
    }
    return p;
}
}
