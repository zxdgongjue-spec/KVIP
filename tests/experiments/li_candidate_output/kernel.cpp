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
extern "C" __global__ __aicore__ void li_candidate_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
 using namespace AscendC; using namespace QLICommon;
 auto a=reinterpret_cast<__gm__ uint64_t*>(bindings);
 #define P(i) reinterpret_cast<GM_ADDR>(a[i])
 TPipe pipe;
 using Type=QLIType<fp8_e4m3fn_t,fp8_e4m3fn_t,int32_t,true,LI_LAYOUT::BSND,LI_LAYOUT::PA_BSND,bfloat16_t,float,float,uint16_t>;
 QLIKernel::QuantLightningIndexerKernel<Type> op;
 op.runtimeShape_=rotating::LoadRuntimeShape(reinterpret_cast<__gm__ rotating::Context*>(a[12]));
 QLITilingData tiling{1,4,32,4,8192,2048,1,128,64,0,16384,128,4096,0,1};
 op.Init(P(0),P(1),P(2),P(3),P(4),P(5),P(6),P(7),P(8),P(9),&tiling,&pipe);
 if ASCEND_IS_AIV {op.vectorService.SetPrefixStorage(P(10));op.vectorService.SetRotationContext(P(12));}
 op.Process();pipe.Destroy();
 #undef P
#endif
}
