// SPDX-License-Identifier: Apache-2.0
#include "args.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/cpp/kernel_simt_utils.h"
using namespace AscendC;
#if defined(__CCE_AICORE__)
#include "pipeline_types.h"
#include "shadow_exchange_parallel.h"
using namespace pipeline_ab;
#include "terminal/overlap_update.h"
__simt_vf__ __aicore__ LAUNCH_BOUND(32) inline void Save(__gm__ uint32_t* status,uint32_t q,__ubuf__ uint32_t* work){if(!threadIdx.x)asc_stcg(status+q,work[0]);asc_threadfence();}
#endif
extern "C" __global__ __aicore__ void v9_parallel_terminal(GM_ADDR ptr){
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 if ASCEND_IS_AIV {
 auto a=reinterpret_cast<__gm__ TestArgs*>(ptr);uint32_t q=GetBlockIdx(),count=a->count,resident=a->resident,history=a->history,generation=a->generation;
 TPipe pipe;TBuf<TPosition::VECCALC> buf;pipe.InitBuffer(buf,kParallelIndexWords*4);
 auto local=buf.Get<uint32_t>();auto work=reinterpret_cast<__ubuf__ uint32_t*>(local.GetPhyAddr());
 if(count>kSmallIndexChanges){Simt::VF_CALL<ApplyParallelIndex<true>>(Simt::Dim3{512,1,1},(__gm__ uint32_t*)a->src,(__gm__ uint32_t*)a->dst,count,resident,history,q,generation,(__gm__ uint64_t*)a->keys,(__gm__ uint32_t*)a->scales,(__gm__ uint64_t*)a->addresses,(__gm__ uint64_t*)a->logical,(__gm__ uint32_t*)a->valid,(__gm__ uint32_t*)a->flags,(__gm__ int32_t*)a->inverse,work,(__gm__ uint64_t*)a->snapshot);}else{Simt::VF_CALL<ApplyParallelIndex<false>>(Simt::Dim3{512,1,1},(__gm__ uint32_t*)a->src,(__gm__ uint32_t*)a->dst,count,resident,history,q,generation,(__gm__ uint64_t*)a->keys,(__gm__ uint32_t*)a->scales,(__gm__ uint64_t*)a->addresses,(__gm__ uint64_t*)a->logical,(__gm__ uint32_t*)a->valid,(__gm__ uint32_t*)a->flags,(__gm__ int32_t*)a->inverse,work,(__gm__ uint64_t*)nullptr);}
 SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 Simt::VF_CALL<Save>(Simt::Dim3{32,1,1},reinterpret_cast<__gm__ uint32_t*>(a->status),q,work);
 SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 pipe.Destroy();
 }
#endif
}
