// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace index_team {
constexpr uint32_t kWidth=4, kMaxMoves=2*CHUNK;
constexpr uint32_t kTraceMagic=0x49445836U;
// The claim occupies its own cache line beyond every legal plan. Only the
// planner initializes it, before publishing this replay's chunk directory.
constexpr uint32_t kClaimOffset=kIndexEdgesOffset+4*(kPrefetch+kMaxFinal);
static_assert(kClaimOffset%32==0 && 4*kIndexTeamMinHistory>=kClaimOffset+32,
              "the ordered moves and fourth-field claim must fit after the dense maps");

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
__aicore__ inline void ResetClaim(__gm__ Args* a,uint32_t row) {
    auto claim=Moves(a,row)+kClaimOffset;
    *claim=0;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(claim),sizeof(*claim));
}
__aicore__ inline void PublishPrefix(__gm__ Args* a,uint32_t row) {
    ResetClaim(a,row);
    auto control=Control(a,row);
    control->reserved[0]=1;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(control),sizeof(*control));
}
__aicore__ inline void PublishPlan(__gm__ Args* a,uint32_t row,uint32_t chunks,uint32_t error,
                                  bool prefixPublished=false) {
    if(!prefixPublished)ResetClaim(a,row);
    Publish(Control(a,row),chunks,error);
}

#include "index_team_fields.h"

__aicore__ inline void TraceField(__gm__ uint64_t* base,uint32_t batch,uint32_t row,uint32_t field,
        uint32_t block,uint32_t moves,uint64_t arrival,uint64_t start,uint64_t end,uint32_t error) {
    uint32_t slot=field==3?3*batch+row:3*row+field;
    auto trace=base+8*slot;
    trace[0]=kTraceMagic; trace[1]=row; trace[2]=(uint64_t(block)<<32)|field; trace[3]=moves;
    trace[4]=arrival; trace[5]=end; trace[6]=start; trace[7]=error;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(trace),8*sizeof(uint64_t));
}

__aicore__ inline uint32_t TryFourth(__gm__ uint32_t* claim,uint32_t block,LocalTensor<uint32_t> result) {
    Simt::VF_CALL<ClaimFourth>(Simt::Dim3{32,1,1},claim,block,reinterpret_cast<__ubuf__ uint32_t*>(result.GetPhyAddr()));
    SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
    return result.GetValue(0);
}

template<bool Claim>
__aicore__ inline void RunField(__gm__ Args* a,uint32_t row,uint32_t chunks,uint32_t field,
                                __ubuf__ uint64_t* local,__gm__ uint32_t* claim,uint32_t block) {
    auto ctx=reinterpret_cast<__gm__ Context*>(a->context);
    uint32_t owner=ctx->rowOwners?reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[row]:row;
    auto u=reinterpret_cast<__gm__ uint64_t*>(a->updateBindings);
    Simt::VF_CALL<CopyFields<Claim>>(Simt::Dim3{256,1,1},Moves(a,row),chunks,field,
        reinterpret_cast<__gm__ uint64_t*>(a->keys)+uint64_t(owner)*a->history*16,
        reinterpret_cast<__gm__ uint32_t*>(a->scales)+uint64_t(owner)*a->history,
        reinterpret_cast<__gm__ uint64_t*>(a->addresses)+uint64_t(owner)*u[7],
        reinterpret_cast<__gm__ uint64_t*>(a->logical)+uint64_t(owner)*a->history,
        reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+uint64_t(owner)*a->resident,local,claim,block);
    SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
}

#include "index_team_help.h"
#include "index_team_prefix.h"

__aicore__ __attribute__((noinline)) inline void Copy(__gm__ Args* a,uint32_t block,uint32_t batch) {
    uint32_t row,lane;
    if(block<3*batch){row=block/3;lane=block%3;}
    else {row=block-3*batch;lane=3;}
    auto control=Control(a,row);
    auto edges=Moves(a,row);
    auto claim=edges+kClaimOffset;
    bool profile=(a->prefetchCount&0x80000000U)!=0;
    uint64_t arrival=profile?GetSystemCycle():0;
    auto trace=reinterpret_cast<__gm__ uint64_t*>(a->times)+batch*256+4*batch*32;
    uint32_t error=AwaitPublication(control);
    bool planReady=control->ready!=0;
    uint32_t chunks=planReady?control->count:0;
    TPipe pipe;TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer,kMaxMoves*(5*sizeof(uint64_t)+2*sizeof(uint32_t)));
    auto local=reinterpret_cast<__ubuf__ uint64_t*>(buffer.Get<uint64_t>().GetPhyAddr());
    uint64_t attempt=profile?GetSystemCycle():0;
    uint32_t won=0;
    if(lane==3){
        // The Get worker has already published its completion. A losing CAS
        // returns even while the winner is still copying.
        won=planReady?TryFourth(claim,block,buffer.Get<uint32_t>()):0;
        if(!won){
            pipe.Destroy();
            HelpAfterLocalWork(a,block,batch);
            return;
        }
    }
    // All changes to the four fields still follow the Q4 descriptor snapshot.
    auto flags=reinterpret_cast<__gm__ BatchFlag*>(a->flags)+row*kFlagCount;
    error|=AwaitPublication(flags+4);
    uint32_t moves=0;
    if(profile && !error && chunks){
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(edges),kIndexEdgesOffset*4);
        moves=edges[chunks-1];
    }
    uint64_t start=profile?GetSystemCycle():0;
    if(lane<3){
        if(!error && chunks){
            RunField<true>(a,row,chunks,lane,local,claim,block);
            won=buffer.Get<uint32_t>().GetValue(0);
        }else if(planReady){
            won=TryFourth(claim,block,buffer.Get<uint32_t>());
        }
        attempt=profile?GetSystemCycle():0;
        if(profile)TraceField(trace,batch,row,lane,block,moves,arrival,start,attempt,error);
        Publish(control+1+lane,0,error);
    }
    if(won){
        start=profile?GetSystemCycle():0;
        if(!error && chunks){
            if(won==1)RunField<false>(a,row,chunks,3,local,nullptr,block);
            else CopyOwnedFourth(a,row,won-1,chunks);
        }
        if(profile)TraceField(trace,batch,row,3,block,moves,attempt,start,GetSystemCycle(),error);
        Publish(control+4,0,error);
    }
    pipe.Destroy();
    if(lane==3)HelpAfterLocalWork(a,block,batch);
}
} // namespace index_team
