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
struct RuntimePostGeometry { uint32_t groups, threads, bank, localIdGroups, localIdOffsetWords; };
KVIP_GEOMETRY_FN constexpr RuntimePostGeometry MakePostGeometry(uint32_t batch,uint32_t lanes,bool twoLocal=true) {
    uint32_t all=(batch+lanes-1)/lanes,groups=all<8?all:8;
    uint32_t threads=groups<=1?512:(groups<=2?256:(groups<=4?128:64));
    uint32_t bank=groups<=1?448:(groups<=2?240:(groups<=4?120:56));
    return {groups,threads,bank,groups==1?1U:(twoLocal && groups==2?2U:0U),groups*256};
}
}
#undef KVIP_GEOMETRY_FN
