// Separate numeric LI consumers. No resident threshold, classification or Fetch.
#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "context.h"
#include "plan.h"
struct QLITilingData {
    uint32_t bSize,tSize,gSize,s1Size,s2Size,sparseCount,usedCoreNum,blockSize,maxBlockNumPerBatch,sparseMode;
    uint32_t keyStride0=16384,keyDequantScaleStride0=128;
    uint32_t resident,splitQ,aicCount;
};
#include "indexer.h"
#endif
extern "C" __global__ __aicore__ void li_qk_scatter(
    GM_ADDR input,GM_ADDR canonical,GM_ADDR tasks,uint32_t rows,uint32_t stride,
    uint32_t history,uint32_t physicalGroup) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        using namespace AscendC;uint32_t row=GetBlockIdx();if(row>=rows)return;
        auto t=reinterpret_cast<__gm__ const li_qk::Task*>(tasks)+row;
        TPipe pipe;TBuf<TPosition::VECCALC> buffer;pipe.InitBuffer(buffer,8192);
        auto local=buffer.Get<uint16_t>();GlobalTensor<uint16_t> src,dst;
        src.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(input));
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(canonical));
        for(uint32_t q=0;q<t->queryCount;++q)for(uint32_t x=0;x<t->end-t->begin;x+=4096) {
            uint32_t n=t->end-t->begin-x;if(n>4096)n=4096;
            DataCopy(local,src[(uint64_t(row)*physicalGroup+q)*stride+x],n);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            DataCopy(dst[(uint64_t(t->queryBegin)+q)*history+t->begin+x],local,n);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        pipe.Destroy();
    }
#endif
}
extern "C" __global__ __aicore__ void li_qk_topk(
    GM_ADDR canonical,GM_ADDR indices,GM_ADDR context,uint32_t history,uint32_t heads) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        using namespace AscendC;using namespace QLICommon;uint32_t q=GetBlockIdx();if(q>=4)return;
        using Type=QLIType<fp8_e4m3fn_t,fp8_e4m3fn_t,int32_t,true,LI_LAYOUT::BSND,LI_LAYOUT::PA_BSND,
                          bfloat16_t,float,float,uint16_t>;
        TPipe pipe;QLIKernel::QLIVector<Type> op;ConstInfo c{};
        c.batchSize=1;c.tSize=4;c.qSeqSize=c.s1BaseSize=4;c.s2BaseSize=128;
        c.qHeadNum=c.gSize=heads;c.kHeadNum=1;c.kSeqSize=history;c.sparseCount=2048;
        c.kCacheBlockSize=128;c.maxBlockNumPerBatch=history/128;c.headDim=128;c.mBaseSize=4*heads;
        QLITilingData tiling{};op.InitParams(c,&tiling,true);
        GlobalTensor<uint16_t> scores;GlobalTensor<int32_t> output,pages;
        GlobalTensor<bfloat16_t> weights;GlobalTensor<float> qs,ks;
        scores.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(canonical));
        output.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));
        op.InitVecInputTensor(weights,qs,ks,output,pages);op.InitVecWorkspaceTensor(scores);
        // Only the shape and optional queryEnds are read; no queues/Service.
        op.SetRotationContext(context);op.InitBuffers(&pipe,true);
        RunInfo info{};info.actS1Size=4;info.actS2Size=info.actS2SizeOrig=info.validS2Len=history;
        op.AllocEventID();op.ProcessTopK(info,q);op.FreeEventID();pipe.Destroy();
    }
#endif
}
