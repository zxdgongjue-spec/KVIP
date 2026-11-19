// Exercise the real precomputed Cold consumer without Service or transport.
#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "context.h"
struct QLITilingData {
    uint32_t bSize,tSize,gSize,s1Size,s2Size,sparseCount,usedCoreNum,blockSize,maxBlockNumPerBatch,sparseMode;
    uint32_t keyStride0=16384,keyDequantScaleStride0=128;
    uint32_t resident,splitQ,aicCount;
};
#include "indexer.h"
#endif
extern "C" __global__ __aicore__ void cold_contract(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIV {
        using namespace AscendC; using namespace QLICommon;
        auto a=reinterpret_cast<__gm__ uint64_t*>(bindings);
#define P(i) reinterpret_cast<GM_ADDR>(a[i])
        auto shape=rotating::LoadRuntimeShape(reinterpret_cast<__gm__ rotating::Context*>(a[14]));
        TPipe pipe;
        using Type=QLIType<fp8_e4m3fn_t,fp8_e4m3fn_t,int32_t,true,LI_LAYOUT::BSND,LI_LAYOUT::PA_BSND,bfloat16_t,float,float,uint16_t>;
        QLIKernel::QuantLightningIndexerKernel<Type> op;
        op.runtimeShape_=shape;op.sequenceSplit_=true;op.precomputedScore_=true;
        op.rotationContext_=P(14);
        QLITilingData tiling{shape.batch,shape.batch*4,shape.indexHeads,4,shape.history,2048,
            shape.liCores,128,shape.history/128,0,16384,128,shape.resident,0,shape.aicCount};
        op.Init(P(0),P(1),P(2),P(3),P(4),P(5),P(6),P(7),P(9),P(12),&tiling,&pipe);
        op.vectorService.SetPrefixStorage(P(13));op.vectorService.SetRotationContext(P(14));
        op.vectorService.AllocEventID();
        uint32_t core=GetBlockIdx()/2,batch=core/2,shard=core%2;
        op.vectorService.LoadSequenceSplitThresholds(batch);
        op.vectorService.PrepareRotationProgress();
        uint32_t half=(shape.history-shape.resident)/256;
        uint32_t begin=shape.resident/128+shard*half,end=begin+half;
        for(uint32_t tile=begin;tile<end;++tile) {
            RunInfo info{};info.loop=tile-begin;info.bIdx=batch;info.s2Idx=tile;
            op.vectorService.ProcessPrecomputedVec1(info,shard,tile+1==end);
        }
        op.vectorService.FreeEventID();pipe.Destroy();
#undef P
    }
#endif
}
