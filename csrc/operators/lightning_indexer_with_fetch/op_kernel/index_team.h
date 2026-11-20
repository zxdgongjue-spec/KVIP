// SPDX-License-Identifier: Apache-2.0
#pragma once

// The existing row planner computes the same chunk closures once. Every
// lane replays them in order for its own four uint64 key words. Only this
// lane ever writes these words, even if the same destination recurs.
namespace index_team {
constexpr uint32_t kWidth=4, kMaxMoves=2*CHUNK;
constexpr uint32_t kTraceMagic=0x49445834U;
static_assert(4*kIndexTeamMinHistory>=kIndexEdgesOffset+4*(kPrefetch+kMaxFinal),
              "the two dense maps must not overlap the ordered move list");

__aicore__ inline __gm__ BatchFlag* Control(__gm__ Args* a,uint32_t b) {
    return reinterpret_cast<__gm__ BatchFlag*>(a->indexTeam)+b*(kWidth+1);
}
__aicore__ inline __gm__ uint32_t* Moves(__gm__ Args* a,uint32_t b) {
    auto dense=reinterpret_cast<__gm__ uint32_t*>(a->denseShadow)+uint64_t(b)*a->history*6;
    return dense+2*a->history;
}
__aicore__ inline void Publish(__gm__ BatchFlag* flag,uint32_t count,uint32_t error) {
    flag->count=count; flag->error=error; flag->ready=1;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flag),sizeof(*flag));
}

#include "index_team_fields.h"
__aicore__ __attribute__((noinline)) inline void Copy(__gm__ Args* a,uint32_t block,uint32_t batch) {
    uint32_t row,lane;
    if(block<3*batch){row=block/3;lane=block%3;}
    else {row=block-3*batch;lane=3;}
    auto control=Control(a,row);
    auto ctx=reinterpret_cast<__gm__ Context*>(a->context);
    bool profile=(a->prefetchCount&0x80000000U)!=0;
    auto trace=reinterpret_cast<__gm__ uint64_t*>(a->times)+batch*256+4*batch*32+block*8;
    if(profile){trace[0]=kTraceMagic; trace[1]=row; trace[2]=lane; trace[4]=GetSystemCycle();}
    uint32_t error=AwaitPublication(control),chunks=control->count;
    // Addresses and logical IDs stay immutable until Q3's descriptors and
    // service outputs have captured their original values.
    auto flags=reinterpret_cast<__gm__ BatchFlag*>(a->flags)+row*kFlagCount;
    error|=AwaitPublication(flags+4);
    uint32_t moves=0;
    if(profile)trace[6]=GetSystemCycle();
    if(!error && chunks){
        auto edges=Moves(a,row);
        uint32_t owner=ctx->rowOwners?reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[row]:row;
        auto u=reinterpret_cast<__gm__ uint64_t*>(a->updateBindings);
        TPipe pipe;TBuf<TPosition::VECCALC> buffer;
        pipe.InitBuffer(buffer,kMaxMoves*(5*sizeof(uint64_t)+2*sizeof(uint32_t)));
        auto local=reinterpret_cast<__ubuf__ uint64_t*>(buffer.Get<uint64_t>().GetPhyAddr());
        Simt::VF_CALL<CopyFields>(Simt::Dim3{256,1,1},edges,chunks,lane,
           reinterpret_cast<__gm__ uint64_t*>(a->keys)+uint64_t(owner)*a->history*16,
           reinterpret_cast<__gm__ uint32_t*>(a->scales)+uint64_t(owner)*a->history,
           reinterpret_cast<__gm__ uint64_t*>(a->addresses)+uint64_t(owner)*u[7],
           reinterpret_cast<__gm__ uint64_t*>(a->logical)+uint64_t(owner)*a->history,
           reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+uint64_t(owner)*a->resident,local);
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
        pipe.Destroy();
        if(profile){
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(edges),kIndexEdgesOffset*4);
            moves=edges[chunks-1];
        }
    }
    if(profile){trace[3]=moves; trace[5]=GetSystemCycle();trace[7]=error;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(trace),8*sizeof(uint64_t));}
    Publish(control+1+lane,0,error);
}
} // namespace index_team
