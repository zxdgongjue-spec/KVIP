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
#endif
extern "C" __global__ __aicore__ void v9_li_window_service(
    GM_ADDR bindings, uint32_t epoch, uint32_t end0, uint32_t end1) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
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
        for(uint32_t b=lane;b<s.batch;b+=s.serviceLanes) for(uint32_t q=0;q<4;++q) {
            auto ready=reinterpret_cast<__gm__ uint32_t*>(ctx->progress)+(b*4+q)*32;
            ready[0]=end0;ready[16]=end1;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ready),128);
        }
        TPipe pipe;
        using Type=QLIType<fp8_e4m3fn_t,fp8_e4m3fn_t,int32_t,true,LI_LAYOUT::BSND,LI_LAYOUT::PA_BSND,bfloat16_t,float,float,uint16_t>;
        QLIKernel::QLIVector<Type> op;
        ConstInfo c{};c.batchSize=s.batch;c.tSize=s.batch*4;c.qSeqSize=c.s1BaseSize=4;
        c.s2BaseSize=128;c.qHeadNum=c.gSize=s.indexHeads;c.kHeadNum=1;c.kSeqSize=s.history;
        c.sparseCount=2048;c.kCacheBlockSize=128;c.maxBlockNumPerBatch=s.history/128;
        c.headDim=128;c.mBaseSize=4*s.indexHeads;
        QLITilingData tiling{};
        op.InitParams(c,&tiling,false);op.SetServiceBlock(core*2+sub);
        op.SetPrefixStorage(reinterpret_cast<GM_ADDR>(a[13]));
        op.SetRotationContext(reinterpret_cast<GM_ADDR>(a[14]));
        op.SetStreamingWorkspace(reinterpret_cast<GM_ADDR>(a[15]));
        op.InitBuffers(&pipe,false);op.AllocEventID();
        op.RotateFetch(epoch,rotating::SequenceHalfWindow(s));
        op.FreeEventID();pipe.Destroy();
    }
#endif
}
