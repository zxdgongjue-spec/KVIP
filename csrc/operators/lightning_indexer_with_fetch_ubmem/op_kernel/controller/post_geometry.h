#pragma once
#if defined(__CCE_AICORE__)
#define KVIP_GEOMETRY_FN __aicore__
#else
#define KVIP_GEOMETRY_FN
#endif
namespace rotating {
KVIP_GEOMETRY_FN constexpr uint32_t ServiceChunkFirst(uint32_t lane,uint32_t groupBase,uint32_t lanes) {
    return lane+groupBase*lanes;
}
KVIP_GEOMETRY_FN constexpr uint32_t ServiceChunkCount(uint32_t batch,uint32_t first,uint32_t lanes,uint32_t limit) {
    if(first>=batch)return 0;
    uint32_t count=(batch-first+lanes-1)/lanes;
    return count<limit?count:limit;
}
constexpr uint32_t kTotalPostGroups=(SIM_BATCH+kServiceLanes-1)/kServiceLanes;
constexpr uint32_t kPostGroups=kTotalPostGroups<8?kTotalPostGroups:8;
constexpr uint32_t kPostThreads=kPostGroups<=1?512:(kPostGroups<=2?256:(kPostGroups<=4?128:64));
constexpr uint32_t kPostBank=kPostGroups<=1?448:(kPostGroups<=2?240:(kPostGroups<=4?120:56));
}
#undef KVIP_GEOMETRY_FN
