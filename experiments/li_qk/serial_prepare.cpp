// Query replication is part of the timed LI call, before numeric producer.
#include "kernel_operator.h"
#include "serial_plan.h"
#include "plan.h"
extern "C" __global__ __aicore__ void li_serial_prepare(GM_ADDR inputs) {
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 if ASCEND_IS_AIV {
    using namespace AscendC;
    auto a=reinterpret_cast<__gm__ const li_serial::Inputs*>(inputs);
    uint32_t row=GetBlockIdx();if(row>=a->rows)return;
    auto t=reinterpret_cast<__gm__ const li_qk::Task*>(a->tasks)+row;
    TPipe pipe;TBuf<TPosition::VECCALC> qb,wb,sb,pb;
    pipe.InitBuffer(qb,4*a->heads*128);pipe.InitBuffer(wb,4*a->heads*2);
    pipe.InitBuffer(sb,4*a->heads*4);pipe.InitBuffer(pb,a->stride/128*4);
    GlobalTensor<uint8_t> q,dq;GlobalTensor<uint16_t> w,dw;
    GlobalTensor<float> s,ds;GlobalTensor<int32_t> p,dp;
    q.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(a->query));dq.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(a->packedQuery));
    w.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(a->weights));dw.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(a->packedWeights));
    s.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a->scales));ds.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a->packedScales));
    p.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a->pages));dp.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a->packedPages));
    DataCopy(qb.Get<uint8_t>(),q[uint64_t(t->owner)*4*a->heads*128],4*a->heads*128);
    DataCopy(wb.Get<uint16_t>(),w[uint64_t(t->owner)*4*a->heads],4*a->heads);
    DataCopy(sb.Get<float>(),s[uint64_t(t->owner)*4*a->heads],4*a->heads);
    DataCopy(pb.Get<int32_t>(),p[uint64_t(t->owner)*a->history/128+t->begin/128],a->stride/128);
    SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    DataCopy(dq[uint64_t(row)*4*a->heads*128],qb.Get<uint8_t>(),4*a->heads*128);
    DataCopy(dw[uint64_t(row)*4*a->heads],wb.Get<uint16_t>(),4*a->heads);
    DataCopy(ds[uint64_t(row)*4*a->heads],sb.Get<float>(),4*a->heads);
    DataCopy(dp[uint64_t(row)*a->stride/128],pb.Get<int32_t>(),a->stride/128);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    pipe.Destroy();
 }
#endif
}
