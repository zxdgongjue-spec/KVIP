// No algorithm patches: invoke the production intermediate-window entry point.
#include "window_types.h"
extern "C" __global__ __aicore__ void test_service(GM_ADDR arg,uint32_t mode){
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 auto a=reinterpret_cast<__gm__ pipeline_ab::Args*>(arg);
 auto ctx=reinterpret_cast<__gm__ rotating::Context*>(a->context);
 auto s=rotating::LoadRuntimeShape(ctx);
 TPipe pipe;
 using Type=QLICommon::QLIType<fp8_e4m3fn_t,fp8_e4m3fn_t,int32_t,true,QLICommon::LI_LAYOUT::BSND,QLICommon::LI_LAYOUT::PA_BSND,bfloat16_t,float,float,uint16_t>;
 QLIKernel::QLIVector<Type> op;
 QLICommon::ConstInfo ci{};ci.gSize=32;ci.qHeadNum=32;ci.kHeadNum=1;ci.headDim=128;ci.mBaseSize=64;ci.s1BaseSize=2;ci.s2BaseSize=128;ci.sparseCount=2048;ci.kSeqSize=s.history;ci.kCacheBlockSize=128;ci.maxBlockNumPerBatch=s.history/128;
 QLITilingData tiling{};op.SetStreamingWorkspace(arg);op.InitParams(ci,&tiling,false);op.InitBuffers(&pipe,false);op.SetRotationContext(reinterpret_cast<GM_ADDR>(ctx));
 GlobalTensor<uint16_t> scores;scores.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(reinterpret_cast<__gm__ uint64_t*>(ctx->incremental)[0]));op.InitVecWorkspaceTensor(scores);
 GlobalTensor<bfloat16_t> weights;GlobalTensor<float> qs,ks;GlobalTensor<int32_t> out,blocks;
 out.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ctx->selected));op.InitVecInputTensor(weights,qs,ks,out,blocks);
 op.AllocEventID();
 op.RotateFetch(s.rounds-1,rotating::SequenceHalfWindow(s));
 op.FreeEventID();pipe.Destroy();
#endif
}
