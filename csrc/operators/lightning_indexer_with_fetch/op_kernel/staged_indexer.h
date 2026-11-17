#pragma once
// Keep all entry points in an AIV-only translation unit: legacy CANN CMake
// applies KERNEL_TASK_TYPE_DEFAULT to every kernel in the source file.
#ifndef KVIP_STAGED_VECTOR_ENTRY
#error "Include staged_indexer.h only from staged_vector.cpp"
#endif
#include "../../common/lifetch/staged_schedule.h"
#include "staged_post_device.h"
namespace kvip_staged_device {
using namespace AscendC;
using namespace QLICommon;
using Type=QLIType<fp8_e4m3fn_t,fp8_e4m3fn_t,int32_t,true,LI_LAYOUT::BSND,LI_LAYOUT::PA_BSND,
                  bfloat16_t,float,float,uint16_t>;
__aicore__ inline void TopK(GM_ADDR bindings,GM_ADDR canonical,uint32_t first,bool resident) {
    auto a=reinterpret_cast<__gm__ uint64_t*>(bindings);
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(a[14]);
    auto s=rotating::LoadRuntimeShape(ctx);
    uint32_t query=first+GetBlockIdx(),b=query/4,q=query%4;
    if(b>=s.batch)return;
    TPipe pipe; QLIKernel::QLIVector<Type> op;
    QLICommon::ConstInfo c{};
    c.batchSize=s.batch;c.tSize=s.batch*4;c.qSeqSize=c.s1BaseSize=4;c.s2BaseSize=128;
    c.qHeadNum=c.gSize=s.indexHeads;c.kHeadNum=1;c.kSeqSize=s.history;c.sparseCount=2048;
    c.kCacheBlockSize=128;c.maxBlockNumPerBatch=s.history/128;c.headDim=128;
    c.mBaseSize=4*s.indexHeads;
    QLITilingData tiling{};
    op.InitParams(c,&tiling,true);
    GlobalTensor<uint16_t> score;score.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(canonical)+uint64_t(b)*4*s.history);
    GlobalTensor<int32_t> output,pages;output.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a[9]));
    pages.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a[7]));
    GlobalTensor<bfloat16_t> weights;GlobalTensor<float> qs,ks;
    op.InitVecInputTensor(weights,qs,ks,output,pages);op.InitVecWorkspaceTensor(score);
    op.SetPrefixStorage(reinterpret_cast<GM_ADDR>(a[13]));op.SetRotationContext(reinterpret_cast<GM_ADDR>(a[14]));
#if KVIP_STREAMING_LI
    op.SetStreamingWorkspace(reinterpret_cast<GM_ADDR>(a[15]));
#endif
    op.InitBuffers(&pipe,true);
    QLICommon::RunInfo info{};info.bIdx=b;info.bN2Idx=b;info.actS1Size=4;
    info.actS2Size=info.actS2SizeOrig=info.validS2Len=s.history;info.indiceOutOffset=uint64_t(b)*4*2048;
    op.AllocEventID();
    if(resident)op.ProcessResidentTopK(info,q);else op.ProcessTopK(info,q);
    op.FreeEventID();pipe.Destroy();
}
}
extern "C" __global__ __aicore__ void v9_staged_scatter(GM_ADDR input,GM_ADDR canonical,
    GM_ADDR tasks,uint32_t rows,uint32_t stride,uint32_t history,uint32_t queryGroup) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        using namespace AscendC;
        uint32_t row=GetBlockIdx();if(row>=rows)return;
        auto t=reinterpret_cast<__gm__ const kvip_staged::ScoreTask*>(tasks)+row;
        TPipe pipe;TBuf<TPosition::VECCALC> buf;pipe.InitBuffer(buf,8192);
        auto local=buf.Get<uint16_t>();GlobalTensor<uint16_t> src,dst;
        src.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(input));
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(canonical));
        for(uint32_t q=0;q<queryGroup;++q)for(uint32_t x=0;x<t->end-t->begin;x+=4096){
            uint32_t n=t->end-t->begin-x;if(n>4096)n=4096;
            DataCopy(local,src[(uint64_t(row)*queryGroup+q)*stride+x],n);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            DataCopy(dst[(uint64_t(t->owner)*4+t->queryBegin+q)*history+t->begin+x],local,n);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        pipe.Destroy();
    }
}
// Match the score kernel's paired AIV query ownership: Q4 -> two Q per AIV,
// Q2 -> one Q per AIV. No task writes Progress or another task's bitmap bytes.
extern "C" __global__ __aicore__ void v9_staged_scatter_predicate(
    GM_ADDR original,GM_ADDR input,GM_ADDR canonical,GM_ADDR tasks,
    uint32_t rows,uint32_t stride,uint32_t queryGroup) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        using namespace AscendC;
        uint32_t row=GetBlockIdx()/2,lane=GetBlockIdx()%2;if(row>=rows)return;
        auto a=reinterpret_cast<__gm__ uint64_t*>(original);
        auto ctx=reinterpret_cast<__gm__ rotating::Context*>(a[14]);
        auto shape=rotating::LoadRuntimeShape(ctx);
        auto t=reinterpret_cast<__gm__ const kvip_staged::ScoreTask*>(tasks)+row;
        TPipe pipe;TBuf<TPosition::VECCALC> data,packed;
        pipe.InitBuffer(data,8192);pipe.InitBuffer(packed,1024);
        auto local=data.Get<uint16_t>();auto bits=packed.Get<uint8_t>();
        GlobalTensor<uint16_t> src,dst;GlobalTensor<uint8_t> bitmap;
        src.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(input));
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(canonical));
        bitmap.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(ctx->hits));
        uint32_t first=kvip_staged_post::FirstQuery(queryGroup,lane);
        for(uint32_t q=first;q<first+kvip_staged_post::QueryCount(queryGroup);++q) {
            uint32_t realQ=t->queryBegin+q;
            uint16_t theta=reinterpret_cast<__gm__ uint16_t*>(a[13])[
                kvip_staged_post::ThresholdOffset(t->owner,realQ)];
            uint32_t valid=ctx->queryEnds?reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds)[t->owner*4+realQ]:shape.history;
            for(uint32_t x=0;x<t->end-t->begin;x+=4096) {
                uint32_t n=t->end-t->begin-x;if(n>4096)n=4096;
                DataCopy(local,src[(uint64_t(row)*queryGroup+q)*stride+x],n);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                Simt::VF_CALL<kvip_staged_device::PackColdUB>(Simt::Dim3{256,1,1},
                    reinterpret_cast<__ubuf__ uint16_t*>(local.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint8_t*>(bits.GetPhyAddr()),n,n,n/4,1,t->begin+x,
                    theta,theta,valid,valid);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopy(dst[kvip_staged_post::CanonicalOffset(t->owner,realQ,shape.history,t->begin+x)],local,n);
                DataCopy(bitmap[kvip_staged_post::BitmapOffset(t->owner,realQ,shape.history,t->begin+x)],bits,n/4);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        }
        pipe.Destroy();
    }
}
extern "C" __global__ __aicore__ void v9_staged_topk(GM_ADDR bindings,GM_ADDR canonical,uint32_t first,uint32_t resident) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV { kvip_staged_device::TopK(bindings,canonical,first,resident!=0); }
}
// Each thread owns one packed byte (four predicates), avoiding subword races.
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void StagedColdBits(__gm__ rotating::Context* ctx,
    __gm__ uint16_t* score,__gm__ uint16_t* prefix,uint32_t b,uint32_t h,
    uint32_t begin,uint32_t end) {
    auto bits=reinterpret_cast<__gm__ uint8_t*>(ctx->hits)+uint64_t(b)*h;
    for(uint32_t q=0;q<4;++q){
        uint16_t theta=asc_ldcg(prefix+uint64_t(b)*(4*2048*3+512)+4*2048*3+q*128);
        uint32_t valid=ctx->queryEnds?asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds)+b*4+q):h;
        for(uint32_t x=begin+threadIdx.x*4;x<end;x+=256*4){
            uint8_t value=0;
            for(uint32_t i=0;i<4;++i)if(x+i<valid && asc_ldcg(score+(uint64_t(b)*4+q)*h+x+i)>theta)value|=1U<<(2*i);
            asc_stcg(bits+q*(h/4)+x/4,value);
        }
    }
    asc_threadfence();
}
extern "C" __global__ __aicore__ void v9_staged_progress(GM_ADDR bindings,GM_ADDR canonical,uint32_t begin,uint32_t end) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        using namespace AscendC;
        auto a=reinterpret_cast<__gm__ uint64_t*>(bindings);auto ctx=reinterpret_cast<__gm__ rotating::Context*>(a[14]);
        auto s=rotating::LoadRuntimeShape(ctx);uint32_t b=GetBlockIdx();if(b>=s.batch)return;
        if(begin>=s.resident && end>begin){
            Simt::VF_CALL<StagedColdBits>(Simt::Dim3{256,1,1},ctx,reinterpret_cast<__gm__ uint16_t*>(canonical),
                reinterpret_cast<__gm__ uint16_t*>(a[13]),b,s.history,begin,end);
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        }
        auto ready=reinterpret_cast<__gm__ rotating::Progress*>(ctx->progress)+b*4;
        for(uint32_t q=0;q<4;++q){ready[q].end[0]=end;kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ready+q),128);}
    }
}
namespace kvip_staged_device {
// Previous same-stream kernels completed all four Hot predicates or every
// Cold task in this epoch. Publish only the owners read by this Service lane;
// no lane reads its sibling's records and no cross-lane barrier is needed.
__aicore__ inline void PublishServiceReady(GM_ADDR bindings,uint32_t end) {
    using namespace AscendC;
    auto a=reinterpret_cast<__gm__ uint64_t*>(bindings);
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(a[14]);
    auto s=rotating::LoadRuntimeShape(ctx);
    uint32_t block=GetBlockIdx();
    uint32_t lane=rotating::ServiceLane(s.batch,s.splitQ,s.serviceAics,block/2,block%2);
    for(uint32_t b=lane;b<s.batch;b+=s.serviceLanes) {
        auto ready=reinterpret_cast<__gm__ rotating::Progress*>(ctx->progress)+b*4;
        for(uint32_t q=0;q<4;++q) {
            ready[q].end[0]=end;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ready+q),128);
        }
    }
}
__aicore__ inline void Service(GM_ADDR bindings,uint32_t epoch) {
    using namespace AscendC;
    auto a=reinterpret_cast<__gm__ uint64_t*>(bindings);auto s=rotating::LoadRuntimeShape(reinterpret_cast<__gm__ rotating::Context*>(a[14]));
    TPipe pipe;QLIKernel::QLIVector<kvip_staged_device::Type> op;
    QLICommon::ConstInfo c{};c.gSize=s.indexHeads;c.kSeqSize=s.history;c.s1BaseSize=4;c.s2BaseSize=128;c.sparseCount=2048;
    QLITilingData tiling{};op.InitParams(c,&tiling,true);
    op.SetRotationContext(reinterpret_cast<GM_ADDR>(a[14]));
#if KVIP_STREAMING_LI
    op.SetStreamingWorkspace(reinterpret_cast<GM_ADDR>(a[15]));
#endif
    op.InitBuffers(&pipe,true);
    op.RotateFetch(epoch);pipe.Destroy();
}
}
extern "C" __global__ __aicore__ void v9_staged_service(GM_ADDR bindings,uint32_t epoch) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV { kvip_staged_device::Service(bindings,epoch); }
}
extern "C" __global__ __aicore__ void v9_staged_service_ready(GM_ADDR bindings,uint32_t epoch,uint32_t end) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        kvip_staged_device::PublishServiceReady(bindings,end);
        kvip_staged_device::Service(bindings,epoch);
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void StagedPreparePages(__gm__ uint32_t* input,
    __gm__ uint32_t* output,uint32_t owner,uint32_t begin,uint32_t length,uint32_t row,uint32_t stride,uint32_t history) {
 for(uint32_t p=threadIdx.x;p<length/128;p+=256)
  asc_stcg(output+uint64_t(row)*stride/128+p,asc_ldcg(input+uint64_t(owner)*history/128+begin/128+p));
 asc_threadfence();
}
// Refresh graph-captured wave inputs from the live bindings every invocation.
// Index can permute pages between rounds; cached host copies are not valid inputs.
__aicore__ inline void PrepareRow(GM_ADDR original,GM_ADDR waveBindings,
    GM_ADDR tasks,uint32_t rows,uint32_t stride,uint32_t queryGroup,uint32_t row,AscendC::LocalTensor<uint8_t> local) {

  using namespace AscendC;if(row>=rows)return;
  auto src=reinterpret_cast<__gm__ uint64_t*>(original),dst=reinterpret_cast<__gm__ uint64_t*>(waveBindings);
  auto t=reinterpret_cast<__gm__ kvip_staged::ScoreTask*>(tasks)+row;
  auto s=rotating::LoadRuntimeShape(reinterpret_cast<__gm__ rotating::Context*>(src[14]));
  if(!row){dst[1]=src[1];dst[4]=src[4];kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(dst),64);}
  const uint32_t fields[3]={0,2,3},widths[3]={128,2,4};
  for(uint32_t f=0;f<3;++f){uint32_t field=fields[f],bytes=queryGroup*s.indexHeads*widths[f];
   GlobalTensor<uint8_t> input,output;input.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(src[field]));
   output.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(dst[field]));
   uint64_t begin=(uint64_t(t->owner)*4+t->queryBegin)*s.indexHeads*widths[f];
   for(uint32_t x=0;x<bytes;x+=8192){uint32_t n=bytes-x;if(n>8192)n=8192;
    DataCopy(local,input[begin+x],n);SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    DataCopy(output[uint64_t(row)*bytes+x],local,n);SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
   }
  }
  auto inputPages=reinterpret_cast<__gm__ uint32_t*>(src[7]),pages=reinterpret_cast<__gm__ uint32_t*>(dst[7]);
  Simt::VF_CALL<StagedPreparePages>(Simt::Dim3{256,1,1},inputPages,pages,t->owner,t->begin,t->end-t->begin,row,stride,s.history);
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
}

extern "C" __global__ __aicore__ void v9_staged_prepare(GM_ADDR original,GM_ADDR waveBindings,
    GM_ADDR tasks,uint32_t rows,uint32_t stride,uint32_t queryGroup) {
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 if ASCEND_IS_AIV {
  AscendC::TPipe pipe;AscendC::TBuf<AscendC::TPosition::VECCALC> buf;pipe.InitBuffer(buf,8192);
  PrepareRow(original,waveBindings,tasks,rows,stride,queryGroup,AscendC::GetBlockIdx(),buf.Get<uint8_t>());
  pipe.Destroy();
 }
}
extern "C" __global__ __aicore__ void v9_staged_prepare_all(GM_ADDR original,GM_ADDR descriptors,uint32_t count) {
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 if ASCEND_IS_AIV {
  AscendC::TPipe pipe;AscendC::TBuf<AscendC::TPosition::VECCALC> buf;pipe.InitBuffer(buf,8192);
  auto waves=reinterpret_cast<__gm__ const kvip_staged::PrepareWave*>(descriptors);
  for(uint32_t i=0;i<count;++i) {
   auto w=waves+i;
   PrepareRow(original,reinterpret_cast<GM_ADDR>(w->bindings),reinterpret_cast<GM_ADDR>(w->tasks),
       w->rows,w->stride,w->queryGroup,AscendC::GetBlockIdx(),buf.Get<uint8_t>());
  }
  pipe.Destroy();
 }
}
