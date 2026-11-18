// Reuse exact native histogram and SIMD compact primitives, in one per-Q launch.
#include "radix.cpp"
#include "producer_plan.h"
#if defined(__CCE_AICORE__)
namespace radix_device {
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void RemapCandidates(
    __ubuf__ const uint16_t* selected,__gm__ const uint32_t* candidates,__gm__ uint32_t* output){
    for(uint32_t i=threadIdx.x;i<2048;i+=blockDim.x)output[i]=candidates[selected[i]];
    asc_threadfence();
}
}
#endif
extern "C" __global__ __aicore__ void li_producer_high_tail(GM_ADDR binding){
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if ASCEND_IS_AIV {
    using namespace AscendC;auto a=reinterpret_cast<__gm__ const li_producer::Bindings*>(binding);uint32_t q=GetBlockIdx();if(q>=4)return;
    TPipe pipe;TBuf<TPosition::VECCALC> input,high,low,tmp,idx,rank,localIds,output;
    pipe.InitBuffer(input,32768);pipe.InitBuffer(high,1024);pipe.InitBuffer(low,1024);pipe.InitBuffer(tmp,1024);pipe.InitBuffer(idx,1024);pipe.InitBuffer(rank,1024);pipe.InitBuffer(localIds,32768);pipe.InitBuffer(output,8192);
    auto in=input.Get<uint16_t>(),ids=localIds.Get<uint16_t>();auto hi=high.Get<uint32_t>(),lo=low.Get<uint32_t>(),work=tmp.Get<uint32_t>(),bin=idx.Get<uint32_t>(),nk=rank.Get<uint32_t>();
    GlobalTensor<uint32_t> partial;partial.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(a->high)+q*li_producer::Parts*256);
    Duplicate(hi,uint32_t(0),256);PipeBarrier<PIPE_V>();
    // Load all partials once into the input UB, before that UB holds scores.
    // Reusing one 1KiB scratch with only MTE2_V lets the next MTE2 overwrite
    // it before the prior vector Add has consumed it (V_MTE2 is also needed).
    auto partials=in.template ReinterpretCast<uint32_t>();
    DataCopy(partials,partial,li_producer::Parts*256);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    for(uint32_t p=0;p<li_producer::Parts;++p){Add(hi,hi,partials[p*256],256);PipeBarrier<PIPE_V>();}
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    topkb16gather::FindHighTargetBinVFImpl(reinterpret_cast<__ubuf__ uint32_t*>(bin.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(nk.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(hi.GetPhyAddr()),a->history-2048+1);PipeBarrier<PIPE_ALL>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);uint32_t highBin=bin.GetValue(0);
    GlobalTensor<uint16_t> scores;scores.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(a->canonical)+q*a->history);
    Duplicate(lo,uint32_t(0),256);PipeBarrier<PIPE_V>();
    for(uint32_t x=0;x<a->history;x+=16384){DataCopy(in,scores[x],16384);SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        topkb16gather::HistogramsLowVFImpl<uint16_t>(reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr()),reinterpret_cast<__ubuf__ uint16_t*>(in.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(bin.GetPhyAddr()),64,true);PipeBarrier<PIPE_V>();Add(lo,lo,work,256);PipeBarrier<PIPE_V>();SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);}
    topkb16gather::FindKthVFImpl(reinterpret_cast<__ubuf__ uint32_t*>(nk.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(lo.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(bin.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr()));PipeBarrier<PIPE_ALL>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);uint32_t tau=rank.Get<uint16_t>().GetValue(0),totalGT=hi.GetValue(255)-hi.GetValue(highBin)+lo.GetValue(255)-lo.GetValue(tau&255),quota=2048-totalGT,gp=0,ep=0;
    if(totalGT>=2048){pipe.Destroy();return;} // Invalid tau must not overwrite another Q/metadata.
    GlobalTensor<int32_t> dst;dst.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a->selected)+q*2048);auto out=output.Get<uint32_t>();
    for(uint32_t x=0;x<a->history;x+=16384){DataCopy(in,scores[x],16384);SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        __VEC_SCOPE__ {radix_device::Filter(reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),reinterpret_cast<__ubuf__ uint16_t*>(in.GetPhyAddr()),16384,uint16_t(tau),false);}
        PipeBarrier<PIPE_ALL>();uint32_t gc=GetSpr<SpecialPurposeReg::AR>()/2;
        if(gc){__VEC_SCOPE__ {radix_device::Widen(reinterpret_cast<__ubuf__ uint32_t*>(out.GetPhyAddr()),reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),gc,x);}PipeBarrier<PIPE_ALL>();DataCopyExtParams cp{1,gc*4,0,0,0};DataCopyPad(dst[gp],out.template ReinterpretCast<int32_t>(),cp);SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);}gp+=gc;
        __VEC_SCOPE__ {radix_device::Filter(reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),reinterpret_cast<__ubuf__ uint16_t*>(in.GetPhyAddr()),16384,uint16_t(tau),true);}
        PipeBarrier<PIPE_ALL>();uint32_t ec=GetSpr<SpecialPurposeReg::AR>()/2,take=ep<quota?quota-ep:0;if(take>ec)take=ec;PipeBarrier<PIPE_ALL>();
        if(take){__VEC_SCOPE__ {radix_device::Widen(reinterpret_cast<__ubuf__ uint32_t*>(out.GetPhyAddr()),reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),take,x);}PipeBarrier<PIPE_ALL>();DataCopyExtParams cp{1,take*4,0,0,0};DataCopyPad(dst[totalGT+ep],out.template ReinterpretCast<int32_t>(),cp);SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);}ep+=take;
    }pipe.Destroy();
 }
#endif
}
extern "C" __global__ __aicore__ void li_producer_candidate_tail(GM_ADDR binding){
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if ASCEND_IS_AIV {
    using namespace AscendC;auto a=reinterpret_cast<__gm__ const li_producer::Bindings*>(binding);uint32_t q=GetBlockIdx();if(q>=4)return;
    constexpr uint32_t n=li_producer::Parts*li_producer::K;
    TPipe pipe;TBuf<TPosition::VECCALC> input,indices,values,hist,hi,lo,rank;
    pipe.InitBuffer(input,n*2);pipe.InitBuffer(indices,4096+256);pipe.InitBuffer(values,4096);pipe.InitBuffer(hist,1024);pipe.InitBuffer(hi,1024);pipe.InitBuffer(lo,1024);pipe.InitBuffer(rank,256);
    auto in=input.Get<uint16_t>(),out=indices.Get<uint16_t>(),val=values.Get<uint16_t>();auto h=hist.Get<uint32_t>(),bhi=hi.Get<uint32_t>(),blo=lo.Get<uint32_t>(),nk=rank.Get<uint32_t>();
    GlobalTensor<uint16_t> scores;scores.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(a->candidateScores)+q*n);
    DataCopy(in,scores,n);SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);PipeBarrier<PIPE_ALL>();
    topkb16gather::LiTopKVF<false>(out,val,in,h,bhi,blo,nk,2048,n);PipeBarrier<PIPE_ALL>();
    // The selected candidate index fits uint16; fetch only 2048 real token IDs.
    Simt::VF_CALL<radix_device::RemapCandidates>(Simt::Dim3{256,1,1},reinterpret_cast<__ubuf__ const uint16_t*>(out.GetPhyAddr()),reinterpret_cast<__gm__ const uint32_t*>(a->candidateIds)+q*n,reinterpret_cast<__gm__ uint32_t*>(a->selected)+q*2048);
    pipe.Destroy();
 }
#endif
}
