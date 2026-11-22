#include "window_types.h"
#if defined(__CCE_AICORE__)
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void Emit(
 __ubuf__ uint32_t* ids,__ubuf__ uint32_t* loc,__ubuf__ uint32_t* error,
 __gm__ uint32_t* mapping,__gm__ uint32_t* lru,__gm__ uint64_t* logical,
 uint32_t history,uint32_t resident,uint32_t begin,uint32_t size,uint32_t capacity){
 for(uint32_t i=threadIdx.x;i<2048;i+=512){
  uint32_t tagged=ids[i],token=tagged&0x3fffffffU,slot=UINT32_MAX;
  if(tagged==UINT32_MAX){ids[i]=UINT32_MAX;loc[i]=UINT32_MAX;continue;}
  if(token>=history){asc_atomic_or(error,1U);ids[i]=UINT32_MAX;loc[i]=UINT32_MAX;continue;}
  if((tagged&0x80000000U)&&token<resident)slot=token;
  else if(token-begin<size)slot=resident+(begin%capacity+token-begin)%capacity;
  else{uint32_t mapped=asc_ldcg(mapping+token);if(mapped<resident)slot=mapped;else if(mapped-resident<2048)slot=asc_ldcg(lru+mapped-resident);}
  if(slot==UINT32_MAX)asc_atomic_or(error,2U);
  loc[i]=slot;ids[i]=uint32_t(asc_ldcg(logical+token));
 }
 asc_syncthreads();
}
#endif
extern "C" __global__ __aicore__ void test_service(GM_ADDR arg,uint32_t mode){
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 auto a=reinterpret_cast<__gm__ pipeline_ab::Args*>(arg);auto ctx=reinterpret_cast<__gm__ rotating::Context*>(a->context);auto s=rotating::LoadRuntimeShape(ctx);
 TPipe pipe;TBuf<TPosition::VECCALC> buf;pipe.InitBuffer(buf,16416);auto ids=buf.Get<uint32_t>(),loc=ids[2048],error=ids[4096];
 GlobalTensor<uint32_t> input,output,slots;input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->selected));output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(a->eventsSrc));slots.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations));
 for(uint32_t q=0;q<4;++q){
  DataCopy(ids,input[q*2048],2048);Duplicate(error,uint32_t(0),8);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  Simt::VF_CALL<Emit>(Simt::Dim3{512,1,1},reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(loc.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(error.GetPhyAddr()),reinterpret_cast<__gm__ uint32_t*>(ctx->mapping),reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots),reinterpret_cast<__gm__ uint64_t*>(a->logical),s.history,s.resident,s.history-512,256,512);
  SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
  DataCopy(output[q*2048],ids,2048);DataCopy(slots[q*2048],loc,2048);
  SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
  reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)[q*32]=error.GetValue(0);kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(a->queryErrors+q*128),128);
 }
 pipe.Destroy();
#endif
}
