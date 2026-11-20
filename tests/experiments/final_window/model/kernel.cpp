#define KVIP_FINAL_WINDOW_INCLUDE
#include "indexer.cpp"
#if defined(__CCE_AICORE__)
#include "pipeline_types.h"
#include "terminal_window.h"
#endif
extern "C" __global__ __aicore__ void test_service(GM_ADDR arg,uint32_t mode){
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 uint32_t q=GetBlockIdx();
 auto a=reinterpret_cast<__gm__ pipeline_ab::Args*>(arg);
 auto ctx=reinterpret_cast<__gm__ rotating::Context*>(a->context);
 auto s=rotating::LoadRuntimeShape(ctx);
 {
 TPipe pipe;
 using Type=QLICommon::QLIType<fp8_e4m3fn_t,fp8_e4m3fn_t,int32_t,true,QLICommon::LI_LAYOUT::BSND,QLICommon::LI_LAYOUT::PA_BSND,bfloat16_t,float,float,uint16_t>;
 QLIKernel::QLIVector<Type> op;
 QLICommon::ConstInfo ci{};ci.gSize=32;ci.qHeadNum=32;ci.kHeadNum=1;ci.headDim=128;ci.mBaseSize=64;ci.s1BaseSize=2;ci.s2BaseSize=128;ci.sparseCount=2048;ci.kSeqSize=s.history;ci.kCacheBlockSize=128;ci.maxBlockNumPerBatch=s.history/128;
 QLITilingData tiling{};op.SetStreamingWorkspace(mode?arg:nullptr);op.InitParams(ci,&tiling,mode!=0);op.InitBuffers(&pipe,mode!=0);op.SetRotationContext(reinterpret_cast<GM_ADDR>(ctx));
 auto header=reinterpret_cast<__gm__ uint64_t*>(ctx->incremental);
 GlobalTensor<uint16_t> scores;scores.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(header[0]));op.InitVecWorkspaceTensor(scores);
 GlobalTensor<bfloat16_t> weights;GlobalTensor<float> qs,ks;GlobalTensor<int32_t> out,blocks;
 out.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ctx->selected));op.InitVecInputTensor(weights,qs,ks,out,blocks);
 QLICommon::RunInfo info{};info.actS1Size=4;info.actS2Size=s.history;info.actS2SizeOrig=s.history;info.indiceOutOffset=0;
 op.AllocEventID();if(!mode)op.AdvanceWindow(3,rotating::SequenceHalfWindow(s));else op.ProcessTopK(info,q);op.FreeEventID();pipe.Destroy();
 }
 AscendC::SyncAll<true>();
 if(q==0){
  if(mode)RunTerminalService(a,0);
  else {TPipe pipe;TBuf<TPosition::VECCALC> scratch,flags,ids;
   pipe.InitBuffer(scratch,65536);pipe.InitBuffer(flags,32768);pipe.InitBuffer(ids,8192);
   rotating::FastService<true,true,false>(1,s.history,s.resident,4,1,656,256,240,1,23808,
    reinterpret_cast<GM_ADDR>(ctx),3,0,0,scratch.Get<uint32_t>(),flags.Get<uint16_t>(),ids.Get<uint32_t>(),rotating::SequenceHalfWindow(s),a->queryErrors);
   pipe.Destroy();}
 }
#endif
}
