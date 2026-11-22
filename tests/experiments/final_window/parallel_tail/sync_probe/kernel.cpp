#include "window_types.h"
#if defined(__CCE_AICORE__)
#include "parallel_post.h"
#endif
extern "C" __global__ __aicore__ void test_service(GM_ADDR arg,uint32_t mode){
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 auto a=reinterpret_cast<__gm__ pipeline_ab::Args*>(arg);uint32_t q=GetBlockIdx();TPipe pipe;TBuf<TPosition::VECCALC> buffer;pipe.InitBuffer(buffer,256);auto tmp=buffer.Get<uint32_t>();Duplicate(tmp,uint32_t(0),64);PipeBarrier<PIPE_V>();
 Simt::VF_CALL<WaitFour>(Simt::Dim3{32,1,1},reinterpret_cast<__gm__ uint32_t*>(a->flags),20,1,reinterpret_cast<__ubuf__ uint32_t*>(tmp.GetPhyAddr()),q,q+1,true);
 SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);GlobalTensor<uint32_t> out;out.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(a->eventsSrc)+q*32);DataCopy(out,tmp,32);SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
 pipe.Destroy();
#endif
}
