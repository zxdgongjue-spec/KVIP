// SPDX-License-Identifier: Apache-2.0
#include "row_kernel.h"
// A real SQ producer + CQ consumer. Only the physical CQ entries are supplied by the host.
#if defined(__CCE_AICORE__)
__simt_vf__ __aicore__ LAUNCH_BOUND(32) inline void TestReservation(
    __gm__ uint32_t* plan,__gm__ uint32_t* shared,__gm__ uint32_t* observed,
    __ubuf__ uint32_t* ids,uint32_t q,uint32_t generation) {
    const uint32_t lane=threadIdx.x,count=asc_ldcg(plan);
    if(lane<count) ids[lane]=500+q*10+lane;
    // Q1 (q=0) in the last round has a genuinely different set: preserve its live IDs.
    if(!lane && q==0 && generation==3)ids[0]=777;
    if(lane==1 && q==2 && generation==2) ids[1]=520;
    asc_threadfence_block();
    const bool matched=MatchFinalOrder(plan,ids,count,lane,ids+12);
    const uint32_t at=ReserveCapturedOrder(plan,shared,count,shared+32,generation,q,lane,matched,(q==1 && generation==3)?8:65536);
    if(!lane){asc_stcg(observed,at);asc_stcg(observed+8,asc_ldcg(plan+2));}
    if(lane<count)asc_stcg(observed+1+lane,ids[lane]);
    asc_threadfence();
}
#endif
extern "C" __global__ __aicore__ void gate_camodel(GM_ADDR argbuf, GM_ADDR results, uint32_t enabled,uint32_t offset,GM_ADDR plans,GM_ADDR scratch) {
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 if ASCEND_IS_AIV {
    const uint32_t idx=offset+GetBlockIdx(), q=idx%4, own=idx==0?0:(q==1?2:31+q);
    auto args=reinterpret_cast<__gm__ KvafExecuteBatchGetKernelArgs*>(argbuf)+idx;
    auto out=reinterpret_cast<__gm__ uint64_t*>(results)+idx*64;
    auto post=reinterpret_cast<__gm__ rotating::FinalPost*>(out+32);
    const auto cuts=rotating::PlanTerminalBatches(own,idx<4 || idx>=8?320:52,0);
    const uint32_t group=enabled&&cuts.count==4?(idx/4+q)&3U:rotating::kGateDisabled;
    const uint64_t head=93;
    TPipe pipe;TBuf<TPosition::VECCALC> buffer;pipe.InitBuffer(buffer,128);
    Simt::VF_CALL<TestReservation>(Simt::Dim3{32,1,1},reinterpret_cast<__gm__ uint32_t*>(plans)+idx*kPrivateFinalWords,
         reinterpret_cast<__gm__ uint32_t*>(scratch),reinterpret_cast<__gm__ uint32_t*>(out+48),
         reinterpret_cast<__ubuf__ uint32_t*>(buffer.Get<uint32_t>().GetPhyAddr()),q,1+offset/4);
    SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
    out[0]=GetSystemCycle();
    rotating::CommitTerminalPost(args,head,cuts.ends[0],post,group);
    const auto err=rotating::CompleteTerminalPost(args,post,cuts,head,group);
    out[1]=GetSystemCycle();out[2]=err;out[3]=cuts.count;out[4]=group;
    for(uint32_t i=0;i<cuts.count;++i)out[5+i]=cuts.ends[i];
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(out),512);
 }
#endif
}
