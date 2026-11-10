// SPDX-License-Identifier: Apache-2.0
// Isolated experiment: original LITopk<uint16_t>, one Q per AIV; no Fetch.
#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "simt_api/asc_simt.h"
#include "simt_api/device_functions.h"
#include "indexer_types.h"
#include "topk.h"
using namespace AscendC;
constexpr uint32_t K=2048, CHUNK=16384, HIST=131072, HALF=8192;
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void ClassifyHalf(__ubuf__ uint32_t* indices, __ubuf__ uint32_t* output,
  __ubuf__ uint32_t* valid, __ubuf__ int32_t* reserved, __gm__ uint32_t* mapping,
  __gm__ uint32_t* lru, uint32_t base) {
  for (uint32_t i=threadIdx.x; i<K; i+=512) {
    uint32_t token=indices[i];
    if (token>=base && token<base+HALF) {
      uint32_t p=token-base;
      if (valid[p] && reserved[p]<0) output[i]=token;
      else { uint32_t slot=asc_ldcg(mapping+token);
        output[i]=(slot>=16384 && slot<18432) ? asc_ldcg(lru+slot-16384) : slot; }
    } else if (base==0 && token>=16384) {
      uint32_t slot=asc_ldcg(mapping+token);
      output[i]=(slot>=16384 && slot<18432) ? asc_ldcg(lru+slot-16384) : slot;
    }
  }
}
#endif
extern "C" __global__ __aicore__ void test_topk_classify(GM_ADDR scores, GM_ADDR validGM,
 GM_ADDR reservedGM, GM_ADDR mappingGM, GM_ADDR lruGM, GM_ADDR topkGM, GM_ADDR outGM, uint32_t classify) {
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 if ASCEND_IS_AIV {
  uint32_t q=GetBlockIdx(); if(q>=4)return;
  TPipe pipe; TBuf<TPosition::VECCALC> mrgBuf,idxBuf,scoreBuf,tmpBuf,hotBuf,resBuf,outBuf;
  pipe.InitBuffer(mrgBuf,(K+CHUNK)*2);pipe.InitBuffer(idxBuf,(K+64)*4);
  pipe.InitBuffer(scoreBuf,K*2);pipe.InitBuffer(tmpBuf,48384);
  if(classify){pipe.InitBuffer(hotBuf,HALF*4);pipe.InitBuffer(resBuf,HALF*4);pipe.InitBuffer(outBuf,K*4);}
  auto mrg=mrgBuf.Get<uint16_t>();auto idx=idxBuf.Get<uint32_t>();auto val=scoreBuf.Get<uint16_t>();
  auto tmp=tmpBuf.Get<uint32_t>();topk::LITopk<uint16_t> op;op.Init(K,CHUNK);op.InitBuffers(tmp,idx);
  GlobalTensor<uint16_t> input;input.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(scores)+q*HIST);
  GlobalTensor<uint32_t> hot,res,output; hot.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(validGM));
  res.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(reservedGM));
  for(uint32_t i=0;i<8;++i){
   if(i){SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);DataCopy(mrg,val,K);}
   DataCopy(mrg[i?K:0],input[i*CHUNK],CHUNK);
   SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
   if(classify && i==7){DataCopy(hotBuf.Get<uint32_t>(),hot,HALF);DataCopy(resBuf.Get<uint32_t>(),res,HALF);}
   op(mrg,idx,val,i?K+CHUNK:CHUNK,i,8);
  }
  PipeBarrier<PIPE_V>();
  GlobalTensor<uint32_t> top;top.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(topkGM)+q*K);
  SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);DataCopy(top,idx,K);
  if(classify){
   auto result=outBuf.Get<uint32_t>();
   SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
   // Prefetch the second half into dead TopK scratch while classifying the first.
   DataCopy(mrg.ReinterpretCast<uint32_t>(),hot[HALF],HALF);DataCopy(tmp,res[HALF],HALF);
   Simt::VF_CALL<ClassifyHalf>(Simt::Dim3{512,1,1},reinterpret_cast<__ubuf__ uint32_t*>(idx.GetPhyAddr()),
    reinterpret_cast<__ubuf__ uint32_t*>(result.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(hotBuf.Get<uint32_t>().GetPhyAddr()),
    reinterpret_cast<__ubuf__ int32_t*>(resBuf.Get<uint32_t>().GetPhyAddr()),reinterpret_cast<__gm__ uint32_t*>(mappingGM),reinterpret_cast<__gm__ uint32_t*>(lruGM),0);
   PipeBarrier<PIPE_V>();
   SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
   Simt::VF_CALL<ClassifyHalf>(Simt::Dim3{512,1,1},reinterpret_cast<__ubuf__ uint32_t*>(idx.GetPhyAddr()),
    reinterpret_cast<__ubuf__ uint32_t*>(result.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(mrg.GetPhyAddr()),
    reinterpret_cast<__ubuf__ int32_t*>(tmp.GetPhyAddr()),reinterpret_cast<__gm__ uint32_t*>(mappingGM),reinterpret_cast<__gm__ uint32_t*>(lruGM),HALF);
   output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(outGM)+q*K);
   SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);DataCopy(output,result,K);
  }
  pipe.Destroy();
 }
#endif
}
