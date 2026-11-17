// Post-score controls. All inputs are final sortable keys; no QK or transport.
#include "kernel_operator.h"
#include "tail_plan.h"
#if defined(__CCE_AICORE__)
#include "context.h"
struct QLITilingData {
 uint32_t bSize,tSize,gSize,s1Size,s2Size,sparseCount,usedCoreNum,blockSize,maxBlockNumPerBatch,sparseMode;
 uint32_t keyStride0=16384,keyDequantScaleStride0=128,resident,splitQ,aicCount;
};
#include "indexer.h"
#include "staged_topk_classify.h"
namespace li_tail_device {
using namespace AscendC;using namespace QLICommon;
using Type=QLIType<fp8_e4m3fn_t,fp8_e4m3fn_t,int32_t,true,LI_LAYOUT::BSND,LI_LAYOUT::PA_BSND,bfloat16_t,float,float,uint16_t>;
__aicore__ inline void Select(const __gm__ li_tail::Bindings* a,uint32_t job,bool resident,bool merge) {
 uint32_t parts=resident?a->residentParts:a->parts,query=merge?job:job/parts,part=merge?0:job%parts;
 uint32_t owner=query/4,q=query%4,h=a->history,span=resident?a->resident:h;
 uint32_t length=merge?parts*2048:span/parts,start=merge?0:part*length,stride=merge?length:h;
 auto ctx=reinterpret_cast<__gm__ rotating::Context*>(a->context);
 if(!merge && ctx->queryEnds){uint32_t end=reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds)[query];
  length=end<=start?0:Min(length,end-start);}
 uint64_t destination=merge?a->mergeIndices:(parts>1?a->localIndices:(resident?a->residentSelected:a->selected));
 TPipe pipe;QLIKernel::QLIVector<Type> op;ConstInfo c{};
 c.batchSize=a->batch;c.tSize=a->batch*4;c.qSeqSize=c.s1BaseSize=4;c.s2BaseSize=128;
 c.qHeadNum=c.gSize=32;c.kHeadNum=1;c.kSeqSize=stride;c.sparseCount=2048;c.kCacheBlockSize=128;
 c.maxBlockNumPerBatch=stride/128;c.headDim=128;c.mBaseSize=128;
 QLITilingData tiling{};op.InitParams(c,&tiling,true);
 GlobalTensor<uint16_t> scores;GlobalTensor<int32_t> output,pages;
 GlobalTensor<bfloat16_t> weights;GlobalTensor<float> qs,ks;
 scores.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(merge?a->candidateScores:a->scores)+uint64_t(owner)*4*stride+start);
 // Keep real Q score stride; only the output view is adjusted to this job.
 bool fullResident=resident && !merge && parts==1;
 output.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(destination)+(fullResident?0:uint64_t(job-q)*2048));
 op.InitVecInputTensor(weights,qs,ks,output,pages);op.InitVecWorkspaceTensor(scores);
 op.SetRotationContext(reinterpret_cast<GM_ADDR>(merge?a->plainContext:a->context));
 if(fullResident)op.SetPrefixStorage(reinterpret_cast<GM_ADDR>(a->prefix));
 op.InitBuffers(&pipe,true);RunInfo info{};info.bIdx=owner;info.actS1Size=4;
 info.actS2Size=info.actS2SizeOrig=info.validS2Len=length;
 if(fullResident)info.indiceOutOffset=uint64_t(owner)*4*2048;
 op.AllocEventID();
 if(fullResident)op.ProcessResidentTopK(info,q);else op.ProcessTopK(info,q);
 op.FreeEventID();pipe.Destroy();
}
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void GatherCandidates(
 const __gm__ li_tail::Bindings* a,uint32_t job,bool resident) {
 uint32_t parts=resident?a->residentParts:a->parts,q=job/parts,p=job%parts;
 uint32_t span=resident?a->resident:a->history,length=span/parts;
 auto local=reinterpret_cast<__gm__ int32_t*>(a->localIndices)+uint64_t(job)*2048;
 auto keys=reinterpret_cast<__gm__ uint16_t*>(a->candidateScores)+uint64_t(q)*parts*2048+p*2048;
 auto ids=reinterpret_cast<__gm__ int32_t*>(a->candidateTokens)+uint64_t(q)*parts*2048+p*2048;
 auto score=reinterpret_cast<__gm__ uint16_t*>(a->scores)+uint64_t(q)*a->history;
 for(uint32_t i=threadIdx.x;i<2048;i+=256){int32_t x=asc_ldcg(local+i);
  int32_t token=x>=0 && uint32_t(x)<length?int32_t(p*length)+x:-1;
  asc_stcg(ids+i,token);asc_stcg(keys+i,token>=0?asc_ldcg(score+token):uint16_t(0));}
 asc_threadfence();
}
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void MapCandidates(
 const __gm__ li_tail::Bindings* a,uint32_t query,bool resident) {
 uint32_t parts=resident?a->residentParts:a->parts;
 auto ids=reinterpret_cast<__gm__ int32_t*>(a->candidateTokens)+uint64_t(query)*parts*2048;
 auto chosen=reinterpret_cast<__gm__ int32_t*>(a->mergeIndices)+uint64_t(query)*2048;
 auto out=reinterpret_cast<__gm__ int32_t*>(resident?a->residentSelected:a->selected)+uint64_t(query)*2048;
 for(uint32_t i=threadIdx.x;i<2048;i+=256){int32_t x=asc_ldcg(chosen+i);
  asc_stcg(out+i,x>=0 && uint32_t(x)<parts*2048?asc_ldcg(ids+x):-1);}
 asc_threadfence();
}
// Unique theta only after the complete resident selection (or exact merge).
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void ResidentTheta(
 const __gm__ li_tail::Bindings* a,uint32_t query,__ubuf__ uint16_t* scratch) {
 auto indices=reinterpret_cast<__gm__ int32_t*>(a->residentSelected)+uint64_t(query)*2048;
 auto score=reinterpret_cast<__gm__ uint16_t*>(a->scores)+uint64_t(query)*a->history;
 uint16_t value=0xffff;
 for(uint32_t i=threadIdx.x;i<2048;i+=256){int32_t x=asc_ldcg(indices+i);uint16_t s=x>=0?asc_ldcg(score+x):0;value=value<s?value:s;}
 scratch[threadIdx.x]=value;asc_syncthreads();
 for(uint32_t step=128;step;step/=2){if(threadIdx.x<step){auto x=scratch[threadIdx.x],y=scratch[threadIdx.x+step];scratch[threadIdx.x]=x<y?x:y;}asc_syncthreads();}
 if(!threadIdx.x)asc_stcg(reinterpret_cast<__gm__ uint16_t*>(a->prefix)+kvip_staged_post::ThresholdOffset(query/4,query%4),scratch[0]);
 asc_threadfence();
}
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void HotBits(
 const __gm__ li_tail::Bindings* a,uint32_t query) {
 auto ctx=reinterpret_cast<__gm__ rotating::Context*>(a->context);
 auto lru=reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+(query/4)*2048;
 auto score=reinterpret_cast<__gm__ uint16_t*>(a->scores)+uint64_t(query)*a->history;
 auto bits=reinterpret_cast<__gm__ uint8_t*>(a->hits)+uint64_t(query)*(a->history/4);
 uint16_t theta=asc_ldcg(reinterpret_cast<__gm__ uint16_t*>(a->prefix)+kvip_staged_post::ThresholdOffset(query/4,query%4));
 uint32_t valid=ctx->queryEnds?asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds)+query):a->history;
 for(uint32_t i=threadIdx.x;i<512;i+=256){uint8_t value=0;
  for(uint32_t j=0;j<4;++j){uint32_t slot=asc_ldcg(lru+i*4+j);if(slot<valid && asc_ldcg(score+slot)>theta)value|=1U<<(2*j);}
  asc_stcg(bits+i,value);}
 asc_threadfence();
}
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void ColdBits(
 const __gm__ li_tail::Bindings* a,uint32_t owner,uint32_t firstQ,uint32_t queries,uint32_t begin,uint32_t end) {
 auto ctx=reinterpret_cast<__gm__ rotating::Context*>(a->context);
 for(uint32_t q=firstQ;q<firstQ+queries;++q){uint32_t query=owner*4+q;
  auto score=reinterpret_cast<__gm__ uint16_t*>(a->scores)+uint64_t(query)*a->history;
  auto bits=reinterpret_cast<__gm__ uint8_t*>(a->hits)+uint64_t(query)*(a->history/4);
  uint16_t theta=asc_ldcg(reinterpret_cast<__gm__ uint16_t*>(a->prefix)+kvip_staged_post::ThresholdOffset(owner,q));
  uint32_t valid=ctx->queryEnds?asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds)+query):a->history;
  for(uint32_t x=begin+threadIdx.x*4;x<end;x+=256*4){uint8_t value=0;
   for(uint32_t j=0;j<4;++j)if(x+j<valid && asc_ldcg(score+x+j)>theta)value|=1U<<(2*j);
   asc_stcg(bits+x/4,value);}}
 asc_threadfence();
}
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void UnionBits(const __gm__ li_tail::Bindings* a,uint32_t owner) {
 uint32_t words=a->resident/32;
 auto src=reinterpret_cast<__gm__ uint32_t*>(a->queryBitmaps)+uint64_t(owner)*4*words;
 auto out=reinterpret_cast<__gm__ uint32_t*>(a->unionBitmap)+uint64_t(owner)*words;
 for(uint32_t x=threadIdx.x;x<words;x+=256)asc_stcg(out+x,asc_ldcg(src+x)|asc_ldcg(src+words+x)|asc_ldcg(src+2*words+x)|asc_ldcg(src+3*words+x));
 asc_threadfence();
}
}
#endif
#define TAIL_KERNEL KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)
extern "C" __global__ __aicore__ void li_tail_select(GM_ADDR bindings,uint32_t first,uint32_t resident,uint32_t merge) {
#if defined(__CCE_AICORE__)
 TAIL_KERNEL;if ASCEND_IS_AIV {li_tail_device::Select(reinterpret_cast<__gm__ li_tail::Bindings*>(bindings),first+AscendC::GetBlockIdx(),resident,merge);}
#endif
}
extern "C" __global__ __aicore__ void li_tail_gather(GM_ADDR bindings,uint32_t first,uint32_t resident) {
#if defined(__CCE_AICORE__)
 TAIL_KERNEL;if ASCEND_IS_AIV {AscendC::Simt::VF_CALL<li_tail_device::GatherCandidates>(AscendC::Simt::Dim3{256,1,1},reinterpret_cast<__gm__ li_tail::Bindings*>(bindings),first+AscendC::GetBlockIdx(),resident!=0);}
#endif
}
extern "C" __global__ __aicore__ void li_tail_map(GM_ADDR bindings,uint32_t first,uint32_t resident) {
#if defined(__CCE_AICORE__)
 TAIL_KERNEL;if ASCEND_IS_AIV {AscendC::Simt::VF_CALL<li_tail_device::MapCandidates>(AscendC::Simt::Dim3{256,1,1},reinterpret_cast<__gm__ li_tail::Bindings*>(bindings),first+AscendC::GetBlockIdx(),resident!=0);}
#endif
}
extern "C" __global__ __aicore__ void li_tail_resident_finish(GM_ADDR bindings,uint32_t first) {
#if defined(__CCE_AICORE__)
 TAIL_KERNEL;if ASCEND_IS_AIV {using namespace AscendC;auto a=reinterpret_cast<__gm__ li_tail::Bindings*>(bindings);uint32_t q=first+GetBlockIdx();
  TPipe pipe;TBuf<TPosition::VECCALC> buffer;pipe.InitBuffer(buffer,512);auto scratch=buffer.Get<uint16_t>();
  Simt::VF_CALL<li_tail_device::ResidentTheta>(Simt::Dim3{256,1,1},a,q,reinterpret_cast<__ubuf__ uint16_t*>(scratch.GetPhyAddr()));
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
  Simt::VF_CALL<li_tail_device::HotBits>(Simt::Dim3{256,1,1},a,q);pipe.Destroy();}
#endif
}
extern "C" __global__ __aicore__ void li_tail_cold(GM_ADDR bindings,uint32_t first,uint32_t parallel) {
#if defined(__CCE_AICORE__)
 TAIL_KERNEL;if ASCEND_IS_AIV {using namespace AscendC;auto a=reinterpret_cast<__gm__ li_tail::Bindings*>(bindings);uint32_t job=first+GetBlockIdx();
  uint32_t owner=parallel?job/(a->coldParts*2):job,lane=parallel?job%2:0,part=parallel?(job/2)%a->coldParts:0;
  uint32_t span=(a->history-a->resident)/256,begin=a->resident+(parallel?span*part/a->coldParts*256:0),end=parallel?a->resident+span*(part+1)/a->coldParts*256:a->history;
  Simt::VF_CALL<li_tail_device::ColdBits>(Simt::Dim3{256,1,1},a,owner,parallel?lane*2:0,parallel?2:4,begin,end);}
#endif
}
extern "C" __global__ __aicore__ void li_tail_classify(GM_ADDR bindings,uint32_t first) {
#if defined(__CCE_AICORE__)
 TAIL_KERNEL;if ASCEND_IS_AIV {using namespace AscendC;auto a=reinterpret_cast<__gm__ li_tail::Bindings*>(bindings);uint32_t query=first+GetBlockIdx(),words=a->resident/32;
  TPipe pipe;TBuf<TPosition::VECCALC> ib,lb,bb,eb;pipe.InitBuffer(ib,8192);pipe.InitBuffer(lb,8192);pipe.InitBuffer(bb,16384);pipe.InitBuffer(eb,128);
  auto indices=ib.Get<uint32_t>(),locations=lb.Get<uint32_t>(),bitmap=bb.Get<uint32_t>(),error=eb.Get<uint32_t>();
  GlobalTensor<uint32_t> src,dst,bs,es;src.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(a->selected)+uint64_t(query)*2048);
  dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(a->locations)+uint64_t(query)*2048);
  bs.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(a->queryBitmaps)+uint64_t(query)*words);es.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)+query*32);
  DataCopy(indices,src,2048);Duplicate(bitmap,uint32_t(0),4096);Duplicate(error,uint32_t(0),32);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);PipeBarrier<PIPE_V>();
  Simt::VF_CALL<rotating::ClassifyTopkHalf>(Simt::Dim3{512,1,1},a->history,a->resident,query/4,1,0,a->resident,0,0,false,false,false,
   reinterpret_cast<__gm__ rotating::Context*>(a->context),reinterpret_cast<__ubuf__ uint32_t*>(indices.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(locations.GetPhyAddr()),
   reinterpret_cast<__ubuf__ uint32_t*>(uint64_t(0)),reinterpret_cast<__ubuf__ int32_t*>(uint64_t(0)),reinterpret_cast<__ubuf__ uint32_t*>(bitmap.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(error.GetPhyAddr()));
  Simt::VF_CALL<rotating::MergeClassifyBitmap>(Simt::Dim3{512,1,1},reinterpret_cast<__ubuf__ uint32_t*>(bitmap.GetPhyAddr()),a->resident);
  SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);DataCopy(dst,locations,2048);DataCopy(bs,bitmap,words);DataCopy(es,error,32);
  SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);pipe.Destroy();}
#endif
}
extern "C" __global__ __aicore__ void li_tail_union(GM_ADDR bindings,uint32_t first) {
#if defined(__CCE_AICORE__)
 TAIL_KERNEL;if ASCEND_IS_AIV {AscendC::Simt::VF_CALL<li_tail_device::UnionBits>(AscendC::Simt::Dim3{256,1,1},reinterpret_cast<__gm__ li_tail::Bindings*>(bindings),first+AscendC::GetBlockIdx());}
#endif
}
