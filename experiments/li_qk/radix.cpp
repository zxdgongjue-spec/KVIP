// Exact distributed uint16 radix selection. GM score is immutable; no QK/Fetch.
#include "kernel_operator.h"
#include "radix_plan.h"
#if defined(__CCE_AICORE__)
using namespace AscendC;
#include "indexer/li_types.h"
#include "simt_api/asc_simt.h"
#include "indexer/topk_common.h"
namespace radix_device {
// Keep the native cumulative representation throughout both reductions.
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void ChooseHigh(
    __ubuf__ uint32_t* idx,__ubuf__ uint32_t* rank,__gm__ uint32_t* param) {
    if(!threadIdx.x){param[0]=idx[0];param[1]=rank[0];}
    asc_threadfence();
}
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void FinishLow(
    __ubuf__ uint16_t* kth,__ubuf__ uint32_t* state,__gm__ uint32_t* param,
    __ubuf__ const uint32_t* high,__ubuf__ const uint32_t* low,__gm__ uint32_t* offsets,uint32_t parts) {
    uint32_t t=threadIdx.x;
    uint32_t tau=kth[0],hi=tau>>8,lo=tau&255;
    if(t<parts){state[32+t*2]=high[t*256+255]-high[t*256+hi]+low[t*256+255]-low[t*256+lo];
        state[33+t*2]=low[t*256+lo]-(lo?low[t*256+lo-1]:0);}asc_syncthreads();
    if(t==0){uint32_t total=0;for(uint32_t p=0;p<parts;++p)total+=state[32+p*2];
        param[2]=tau;param[3]=total;
        uint32_t gp=0,ep=0;
        for(uint32_t p=0;p<parts;++p){uint32_t eq=state[33+p*2],quota=li_radix::K-total-ep;if(quota>eq)quota=eq;
            offsets[p*8]=gp;offsets[p*8+1]=ep;offsets[p*8+2]=quota;offsets[p*8+3]=state[32+p*2];
            gp+=state[32+p*2];ep+=quota;}
    }
    asc_threadfence();
}
// Native SIMD compaction; only selected indices leave UB. Mask excludes padding.
__simd_vf__ void Filter(__ubuf__ uint16_t* output,__ubuf__ uint16_t* input,
    uint32_t n,uint16_t tau,bool equal) {
    using namespace MicroAPI;uint32_t remaining=n;ClearSpr<SpecialPurposeReg::AR>();UnalignRegForStore store;
    for(uint32_t i=0;i<(n+127)/128;++i){RegTensor<uint16_t> raw,compact;RegTensor<int16_t> ids;MaskReg selected;
        MaskReg valid=UpdateMask<uint16_t>(remaining);LoadAlign<uint16_t,LoadDist::DIST_NORM>(raw,input+i*128);
        Arange(ids,int16_t(i*128));
        if(equal)Compares<uint16_t,CMPMODE::EQ>(selected,raw,tau,valid);
        else Compares<uint16_t,CMPMODE::GT>(selected,raw,tau,valid);
        Squeeze<uint16_t,MicroAPI::GatherMaskMode::STORE_REG>(compact,reinterpret_cast<RegTensor<uint16_t>&>(ids),selected);StoreUnAlign<uint16_t,PostLiteral::POST_MODE_UPDATE>(output,compact,store);
    }StoreUnAlignPost(output,store);
}
__simd_vf__ void Widen(__ubuf__ uint32_t* output,__ubuf__ uint16_t* input,uint32_t n,uint32_t begin){
    using namespace MicroAPI;uint32_t remaining=n;
    for(uint32_t i=0;i<(n+63)/64;++i){RegTensor<uint16_t> raw;RegTensor<uint32_t> ids;
        MaskReg valid=UpdateMask<uint32_t>(remaining);LoadAlign<uint16_t,LoadDist::DIST_UNPACK_B16>(raw,input+i*64);
        Adds(ids,reinterpret_cast<RegTensor<uint32_t>&>(raw),begin,valid);StoreAlign<uint32_t,StoreDist::DIST_NORM>(output+i*64,ids,valid);}
}
__aicore__ inline void Histogram(GM_ADDR bindings,bool lowPass) {
    auto a=reinterpret_cast<__gm__ const li_radix::Bindings*>(bindings);uint32_t jobs=a->batch*4*a->parts;
    TPipe pipe;TBuf<TPosition::VECCALC> input,total,tmp,idx,padCounts;
    pipe.InitBuffer(input,32768);pipe.InitBuffer(total,1024);pipe.InitBuffer(tmp,1024);pipe.InitBuffer(idx,1024);pipe.InitBuffer(padCounts,1024);
    auto in=input.Get<uint16_t>();auto sum=total.Get<uint32_t>(),scratch=tmp.Get<uint32_t>(),highBin=idx.Get<uint32_t>();
    GlobalTensor<uint16_t> src;src.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(a->scores));
    GlobalTensor<uint32_t> dst;dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(lowPass?a->low:a->high));
    for(uint32_t job=GetBlockIdx();job<jobs;job+=GetBlockNum()){
        uint32_t q=job%(a->batch*4),p=job/(a->batch*4),begin=li_radix::Begin(a->history,a->parts,p),end=li_radix::Begin(a->history,a->parts,p+1);
        uint32_t valid=reinterpret_cast<__gm__ const uint32_t*>(a->ends)[q];if(end>valid)end=valid;
        Duplicate(sum,uint32_t(0),256);uint32_t bin=0;
        if(lowPass){bin=reinterpret_cast<__gm__ uint32_t*>(a->params)[q*32];Duplicate(highBin,bin,256);}
        for(uint32_t x=begin;x<end;x+=16384){uint32_t n=end-x;if(n>16384)n=16384;uint32_t aligned=(n+255)/256*256;
            if(aligned>n)Duplicate(in[n/256*256],uint16_t(0),256);PipeBarrier<PIPE_ALL>();
            DataCopyExtParams cp{1,n*2,0,0,0};DataCopyPadExtParams<uint16_t> pad{true,0,uint8_t((n+15)/16*16-n),0};DataCopyPad(in,src[uint64_t(q)*a->history+x],cp,pad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if(lowPass)topkb16gather::HistogramsLowVFImpl<uint16_t>(reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr()),reinterpret_cast<__ubuf__ uint16_t*>(in.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(highBin.GetPhyAddr()),aligned/256,true);
            else topkb16gather::HistogramsHighVFImpl<uint16_t>(reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr()),reinterpret_cast<__ubuf__ uint16_t*>(in.GetPhyAddr()),aligned/256,true);
            PipeBarrier<PIPE_V>();
            if(aligned>n && (!lowPass || bin==0)){Duplicate(padCounts.Get<uint32_t>(),aligned-n,256);PipeBarrier<PIPE_V>();Sub(scratch,scratch,padCounts.Get<uint32_t>(),256);PipeBarrier<PIPE_V>();}
            Add(sum,sum,scratch,256);PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_ALL>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);DataCopy(dst[uint64_t(q*a->parts+p)*256],sum,256);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }pipe.Destroy();
}
__aicore__ inline void Reduce(GM_ADDR bindings,bool lowPass){
    auto a=reinterpret_cast<__gm__ const li_radix::Bindings*>(bindings);uint32_t q=GetBlockIdx();if(q>=a->batch*4)return;
    TPipe pipe;TBuf<TPosition::VECCALC> total,tmp,state,idxHigh,idxLow,rank,allHigh,allLow;pipe.InitBuffer(total,1024);pipe.InitBuffer(tmp,1024);pipe.InitBuffer(state,2048);
    pipe.InitBuffer(idxHigh,1024);pipe.InitBuffer(idxLow,1024);pipe.InitBuffer(rank,1024);
    pipe.InitBuffer(allHigh,32768);pipe.InitBuffer(allLow,32768);
    auto sum=total.Get<uint32_t>(),scratch=tmp.Get<uint32_t>();GlobalTensor<uint32_t> src;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(lowPass?a->low:a->high)+uint64_t(q)*a->parts*256);Duplicate(sum,uint32_t(0),256);
    auto partials=lowPass?allLow.Get<uint32_t>():allHigh.Get<uint32_t>();
    DataCopy(partials,src,a->parts*256);SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);PipeBarrier<PIPE_ALL>();
    for(uint32_t p=0;p<a->parts;++p){Add(sum,sum,partials[p*256],256);PipeBarrier<PIPE_V>();}
    auto param=reinterpret_cast<__gm__ uint32_t*>(a->params)+q*32;PipeBarrier<PIPE_ALL>();
    if(lowPass){GlobalTensor<uint32_t> hg;hg.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(a->high)+uint64_t(q)*a->parts*256);
        Duplicate(idxHigh.Get<uint32_t>(),uint32_t(param[0]),256);Duplicate(rank.Get<uint32_t>(),uint32_t(param[1]),256);PipeBarrier<PIPE_ALL>();
        topkb16gather::FindKthVFImpl(reinterpret_cast<__ubuf__ uint32_t*>(rank.Get<uint32_t>().GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(sum.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(idxHigh.Get<uint32_t>().GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(idxLow.Get<uint32_t>().GetPhyAddr()));PipeBarrier<PIPE_ALL>();
        DataCopy(allHigh.Get<uint32_t>(),hg,a->parts*256);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);PipeBarrier<PIPE_ALL>();
        Simt::VF_CALL<FinishLow>(Simt::Dim3{256,1,1},reinterpret_cast<__ubuf__ uint16_t*>(rank.Get<uint16_t>().GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(state.Get<uint32_t>().GetPhyAddr()),param,
        reinterpret_cast<__ubuf__ const uint32_t*>(allHigh.Get<uint32_t>().GetPhyAddr()),reinterpret_cast<__ubuf__ const uint32_t*>(allLow.Get<uint32_t>().GetPhyAddr()),
        reinterpret_cast<__gm__ uint32_t*>(a->offsets)+uint64_t(q)*a->parts*8,a->parts);}
    else {uint32_t valid=reinterpret_cast<__gm__ uint32_t*>(a->ends)[q];
        topkb16gather::FindHighTargetBinVFImpl(reinterpret_cast<__ubuf__ uint32_t*>(idxHigh.Get<uint32_t>().GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(rank.Get<uint32_t>().GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(sum.GetPhyAddr()),valid-2048+1);PipeBarrier<PIPE_ALL>();
        Simt::VF_CALL<ChooseHigh>(Simt::Dim3{256,1,1},reinterpret_cast<__ubuf__ uint32_t*>(idxHigh.Get<uint32_t>().GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(rank.Get<uint32_t>().GetPhyAddr()),param);}
    pipe.Destroy();
}
}
#endif
extern "C" __global__ __aicore__ void li_radix_high(GM_ADDR bindings){
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if ASCEND_IS_AIV {radix_device::Histogram(bindings,false);}
#endif
}
extern "C" __global__ __aicore__ void li_radix_high_reduce(GM_ADDR bindings){
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if ASCEND_IS_AIV {radix_device::Reduce(bindings,false);}
#endif
}
extern "C" __global__ __aicore__ void li_radix_low(GM_ADDR bindings){
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if ASCEND_IS_AIV {radix_device::Histogram(bindings,true);}
#endif
}
extern "C" __global__ __aicore__ void li_radix_low_reduce(GM_ADDR bindings){
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if ASCEND_IS_AIV {radix_device::Reduce(bindings,true);}
#endif
}
extern "C" __global__ __aicore__ void li_radix_emit(GM_ADDR bindings){
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if ASCEND_IS_AIV {
    using namespace AscendC;auto a=reinterpret_cast<__gm__ const li_radix::Bindings*>(bindings);uint32_t jobs=a->batch*4*a->parts;
    TPipe pipe;TBuf<TPosition::VECCALC> input,indices,expanded;pipe.InitBuffer(input,32768);pipe.InitBuffer(indices,32768);pipe.InitBuffer(expanded,8192);
    auto in=input.Get<uint16_t>();auto ids=indices.Get<uint16_t>();auto out=expanded.Get<int32_t>();GlobalTensor<uint16_t> src;src.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(a->scores));
    GlobalTensor<int32_t> dst;dst.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a->selected));
    for(uint32_t job=GetBlockIdx();job<jobs;job+=GetBlockNum()){
        uint32_t q=job%(a->batch*4),p=job/(a->batch*4),begin=li_radix::Begin(a->history,a->parts,p),end=li_radix::Begin(a->history,a->parts,p+1);
        uint32_t valid=reinterpret_cast<__gm__ const uint32_t*>(a->ends)[q];if(end>valid)end=valid;
        auto params=reinterpret_cast<__gm__ uint32_t*>(a->params)+q*32;auto off=reinterpret_cast<__gm__ uint32_t*>(a->offsets)+(q*a->parts+p)*8;
        uint32_t gp=0,ep=0,tau=params[2],totalGT=params[3],gtBase=off[0],eqBase=off[1],quota=off[2];
        for(uint32_t x=begin;x<end;x+=16384){uint32_t n=end-x;if(n>16384)n=16384;
            DataCopyExtParams cp{1,n*2,0,0,0};DataCopyPadExtParams<uint16_t> pad{false,0,0,0};DataCopyPad(in,src[uint64_t(q)*a->history+x],cp,pad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            __VEC_SCOPE__ {radix_device::Filter(reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),reinterpret_cast<__ubuf__ uint16_t*>(in.GetPhyAddr()),n,uint16_t(tau),false);}
            uint32_t gc=uint32_t(GetSpr<SpecialPurposeReg::AR>()/2);PipeBarrier<PIPE_ALL>();
            if(gc){__VEC_SCOPE__ {radix_device::Widen(reinterpret_cast<__ubuf__ uint32_t*>(out.GetPhyAddr()),reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),gc,x);}PipeBarrier<PIPE_ALL>();DataCopyExtParams cpOut{1,gc*4,0,0,0};DataCopyPad(dst[uint64_t(q)*2048+gtBase+gp],out,cpOut);SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);}gp+=gc;
            __VEC_SCOPE__ {radix_device::Filter(reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),reinterpret_cast<__ubuf__ uint16_t*>(in.GetPhyAddr()),n,uint16_t(tau),true);}
            uint32_t ec=uint32_t(GetSpr<SpecialPurposeReg::AR>()/2),take=ep<quota?quota-ep:0;if(take>ec)take=ec;PipeBarrier<PIPE_ALL>();
            if(take){__VEC_SCOPE__ {radix_device::Widen(reinterpret_cast<__ubuf__ uint32_t*>(out.GetPhyAddr()),reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),take,x);}PipeBarrier<PIPE_ALL>();DataCopyExtParams cpOut{1,take*4,0,0,0};DataCopyPad(dst[uint64_t(q)*2048+totalGT+eqBase+ep],out,cpOut);SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);}ep+=ec;
        }
    }pipe.Destroy();
 }
#endif
}
