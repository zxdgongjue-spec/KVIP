#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "resolver.h"
#include "pipeline_types.h"
#include "shadow_exchange_parallel.h"
using namespace AscendC;using namespace rotating;using namespace pipeline_ab;
#include "helpers.h"
#endif
extern "C" __global__ __aicore__ void test_service(GM_ADDR arg,uint32_t mode){
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);auto a=reinterpret_cast<__gm__ Args*>(arg);TPipe pipe;TBuf<TPosition::VECCALC> buf;pipe.InitBuffer(buf,SCRATCH_WORDS*4);auto local=buf.Get<uint32_t>();auto work=reinterpret_cast<__ubuf__ uint32_t*>(local.GetPhyAddr());auto ts=reinterpret_cast<__gm__ uint64_t*>(a->times);
 uint32_t counts[5]={84,11,0,94,41},begin=0;
 for(uint32_t q=0;q<(mode?5:1);++q){uint32_t count=mode?counts[q]:230;ts[q*2]=GetSystemCycle();
 if(count){Simt::VF_CALL<SwapColdIndexRows<false>>(Simt::Dim3{512,1,1},reinterpret_cast<__gm__ uint32_t*>(a->eventsSrc)+begin,reinterpret_cast<__gm__ uint32_t*>(a->eventsDst)+begin,begin,count,a->resident,a->history,reinterpret_cast<__gm__ uint64_t*>(a->keys),reinterpret_cast<__gm__ uint32_t*>(a->scales),work);SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);}
 ts[q*2+1]=GetSystemCycle();begin+=count;}
 ts[10]=GetSystemCycle();Simt::VF_CALL<SwapColdMetadata>(Simt::Dim3{512,1,1},reinterpret_cast<__gm__ uint64_t*>(a->addresses),reinterpret_cast<__gm__ uint64_t*>(a->logical),230,work);SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);ts[11]=GetSystemCycle();reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)[0]=local.GetValue(CONTROL);kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(a->queryErrors),128);kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ts),128);pipe.Destroy();
#endif
}
