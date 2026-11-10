#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "context.h"
struct QLITilingData {uint32_t bSize,tSize,gSize,s1Size,s2Size,sparseCount,usedCoreNum,blockSize,maxBlockNumPerBatch,sparseMode,keyStride0=16384,keyDequantScaleStride0=128,resident,splitQ,aicCount;};
#include "indexer_vector.h"
using namespace AscendC;
using namespace QLICommon;
#endif
extern "C" __global__ __aicore__ void test_real_topk(GM_ADDR scores,GM_ADDR bindings,GM_ADDR context,GM_ADDR workspace) {
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 if ASCEND_IS_AIV {
  uint32_t q=GetBlockIdx(); if(q>=4)return;
  TPipe pipe;
  using Type=QLIType<fp8_e4m3fn_t,fp8_e4m3fn_t,int32_t,true,LI_LAYOUT::BSND,LI_LAYOUT::PA_BSND,bfloat16_t,float,float,uint16_t>;
  QLIKernel::QLIVector<Type> op;
  ConstInfo ci{}; ci.gSize=32;ci.qHeadNum=32;ci.kHeadNum=1;ci.headDim=128;ci.mBaseSize=128;ci.s1BaseSize=4;ci.s2BaseSize=128;ci.sparseCount=2048;ci.kSeqSize=131072;ci.kCacheBlockSize=128;ci.maxBlockNumPerBatch=1024;
  QLITilingData tiling{};
  op.SetStreamingWorkspace(workspace);op.InitParams(ci,&tiling,true);op.InitBuffers(&pipe,true);op.SetRotationContext(context);
  GlobalTensor<uint16_t> sg;sg.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(scores));op.InitVecWorkspaceTensor(sg);
  GlobalTensor<bfloat16_t> weights;GlobalTensor<float> qscale,kscale;GlobalTensor<int32_t> out,blocks;
  out.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(bindings));op.InitVecInputTensor(weights,qscale,kscale,out,blocks);
  RunInfo info{};info.actS1Size=4;info.actS2Size=131072;info.actS2SizeOrig=131072;info.indiceOutOffset=0;
  op.AllocEventID();op.ProcessTopK(info,q);op.FreeEventID();pipe.Destroy();
 }
#endif
}
