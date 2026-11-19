// Keep AIV-only entry points separate from MIX_AIC producer translation units.
#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "context.h"
struct QLITilingData {
    uint32_t bSize,tSize,gSize,s1Size,s2Size,sparseCount,usedCoreNum,blockSize,maxBlockNumPerBatch,sparseMode;
    uint32_t keyStride0=16384,keyDequantScaleStride0=128;
    uint32_t resident,splitQ,aicCount;
};
#include "indexer/indexer_vector.h"
namespace {
template<uint32_t Mode> __aicore__ inline void WindowService(
    GM_ADDR bindings,uint32_t epoch,uint32_t end0,uint32_t end1) {
    if ASCEND_IS_AIV {
        using namespace AscendC; using namespace QLICommon;
        auto a=reinterpret_cast<__gm__ uint64_t*>(bindings);
        auto ctx=reinterpret_cast<__gm__ rotating::Context*>(a[14]);
        auto s=rotating::LoadRuntimeShape(ctx);
        const uint32_t group=GetBlockIdx()/2, sub=GetBlockIdx()%2;
        const uint32_t core=rotating::ServiceCore(s.batch,s.splitQ,s.serviceAics,epoch,group);
        const uint32_t lane=rotating::ServiceLane(s.batch,s.splitQ,s.serviceAics,core,sub);
        // Same-stream producer completion joins all P shards and all four Q.
        // Only the lane consuming a request publishes its two MR23 cursors.
        if constexpr (Mode==1) {
            auto ready=reinterpret_cast<__gm__ uint32_t*>(ctx->progress)+GetBlockIdx()*32;
            ready[0]=end0;ready[16]=end1;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ready),128);
        } else if constexpr (Mode==0 || Mode==3) {
            for(uint32_t b=lane;b<s.batch;b+=s.serviceLanes) for(uint32_t q=0;q<4;++q) {
                auto ready=reinterpret_cast<__gm__ uint32_t*>(ctx->progress)+(b*4+q)*32;
                ready[0]=end0;ready[16]=end1;
                kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ready),128);
            }
        }
        TPipe pipe;
        using Type=QLIType<fp8_e4m3fn_t,fp8_e4m3fn_t,int32_t,true,LI_LAYOUT::BSND,LI_LAYOUT::PA_BSND,bfloat16_t,float,float,uint16_t>;
        QLIKernel::QLIVector<Type> op;
        ConstInfo c{};c.batchSize=s.batch;c.tSize=s.batch*4;c.qSeqSize=c.s1BaseSize=4;
        c.s2BaseSize=128;c.qHeadNum=c.gSize=s.indexHeads;c.kHeadNum=1;c.kSeqSize=s.history;
        c.sparseCount=2048;c.kCacheBlockSize=128;c.maxBlockNumPerBatch=s.history/128;
        c.headDim=128;c.mBaseSize=4*s.indexHeads;
        QLITilingData tiling{};
        op.InitParams(c,&tiling,false);op.SetServiceBlock(Mode==1?GetBlockIdx():core*2+sub);
        op.SetPrefixStorage(reinterpret_cast<GM_ADDR>(a[13]));
        op.SetRotationContext(reinterpret_cast<GM_ADDR>(a[14]));
        op.SetStreamingWorkspace(reinterpret_cast<GM_ADDR>(a[15]));
        op.InitBuffers(&pipe,false);
        // These standalone AIV kernels never run ProcessVec1/ProcessTopK.
        // Their window helpers balance every event locally, including 6/7
        // for next-Q DMA and candidate output. AllocEventID would seed score
        // bank credits on those same IDs, causing duplicate SetFlag signals.
#if defined(KVIP_WINDOW_TOPK_PROBE) && KVIP_WINDOW_TOPK_PROBE
        auto clocks=reinterpret_cast<__gm__ uint64_t*>(a[11])+GetBlockIdx()*16;
        if constexpr(Mode==1 || Mode==3)clocks[0]=GetSystemCycle();
#endif
        if constexpr (Mode==1)op.AdvanceQueryWindow(GetBlockIdx(),epoch,rotating::SequenceHalfWindow(s));
        else if constexpr (Mode==2)op.PostWindow(epoch,rotating::SequenceHalfWindow(s));
        else if constexpr (Mode==3)op.AdvanceWindow(epoch,rotating::SequenceHalfWindow(s));
        else op.RotateFetch(epoch,rotating::SequenceHalfWindow(s));
#if defined(KVIP_WINDOW_TOPK_PROBE) && KVIP_WINDOW_TOPK_PROBE
        if constexpr(Mode==1 || Mode==3){clocks[1]=GetSystemCycle();kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(clocks),16);}
#endif
        pipe.Destroy();
    }
}
}
#endif
extern "C" __global__ __aicore__ void v9_li_window_service(
    GM_ADDR bindings,uint32_t epoch,uint32_t end0,uint32_t end1) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    WindowService<0>(bindings,epoch,end0,end1);
#endif
}
extern "C" __global__ __aicore__ void v9_li_window_topk(
    GM_ADDR bindings,uint32_t epoch,uint32_t end0,uint32_t end1) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    WindowService<1>(bindings,epoch,end0,end1);
#endif
}
extern "C" __global__ __aicore__ void v9_li_window_post(
    GM_ADDR bindings,uint32_t epoch,uint32_t end0,uint32_t end1) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    WindowService<2>(bindings,epoch,end0,end1);
#endif
}
#if defined(KVIP_WINDOW_TOPK_PROBE) && KVIP_WINDOW_TOPK_PROBE
// Contract/Model control: the real original AdvanceWindow, without URMA posting.
extern "C" __global__ __aicore__ void v9_li_window_topk_serial_probe(
    GM_ADDR bindings,uint32_t epoch,uint32_t end0,uint32_t end1) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    WindowService<3>(bindings,epoch,end0,end1);
#endif
}
#endif
