// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "indexer_types.h"
#include "topk.h"
#include "simt_api/asc_simt.h"
using namespace AscendC;
constexpr uint32_t K = 2048;
constexpr uint32_t MaxNew = 16384;
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void MapIds(__ubuf__ uint32_t* selected, __ubuf__ uint32_t* old) {
    for (uint32_t i = threadIdx.x; i < K; i += 256) {
        uint32_t id = selected[i];
        selected[i] = id < K ? old[id] : 16384 + id - K;
    }
}
template<bool Profile>
__aicore__ inline void Run(GM_ADDR scores, GM_ADDR oldIds, GM_ADDR outScores,
    GM_ADDR outIds, GM_ADDR ticks, uint32_t rows, uint32_t added) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> inputBuf, idsBuf, scoresBuf, oldBuf, scratchBuf;
    topk::LITopk<uint16_t> select;
    select.Init(K, MaxNew);
    pipe.InitBuffer(inputBuf, (K + MaxNew) * 2);
    pipe.InitBuffer(idsBuf, K * 4);
    pipe.InitBuffer(scoresBuf, K * 2);
    pipe.InitBuffer(oldBuf, K * 4);
    pipe.InitBuffer(scratchBuf, select.GetSharedTmpBufferSize());
    auto input = inputBuf.Get<uint16_t>(); auto ids = idsBuf.Get<uint32_t>();
    auto values = scoresBuf.Get<uint16_t>(); auto old = oldBuf.Get<uint32_t>();
    auto scratch = scratchBuf.Get<uint32_t>(); select.InitBuffers(scratch, ids);
    GlobalTensor<uint16_t> source, outputValues;
    GlobalTensor<uint32_t> sourceIds, outputIds;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(scores));
    sourceIds.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(oldIds));
    outputValues.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(outScores));
    outputIds.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(outIds));
    auto times = reinterpret_cast<__gm__ uint64_t*>(ticks);
    const uint32_t width = K + added;
    for (uint32_t row = GetBlockIdx() * rows; row < (GetBlockIdx() + 1) * rows; ++row) {
        uint64_t t0=0,t1=0,t2=0,t3=0,t4=0;
        if constexpr(Profile) t0=GetSystemCycle();
        DataCopy(input, source[row * width], width);
        DataCopy(old, sourceIds[row * K], K);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        if constexpr(Profile) t1=GetSystemCycle();
        select(input, ids, values, width, 0, 1, true);
        PipeBarrier<PIPE_V>();
        if constexpr(Profile) t2=GetSystemCycle();
        Simt::VF_CALL<MapIds>(Simt::Dim3{256,1,1},
            reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(old.GetPhyAddr()));
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        if constexpr(Profile) t3=GetSystemCycle();
        DataCopy(outputValues[row * K], values, K);
        DataCopy(outputIds[row * K], ids, K);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0); WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        if constexpr(Profile) {
            t4=GetSystemCycle();
            times[row*5]=t0; times[row*5+1]=t1; times[row*5+2]=t2;
            times[row*5+3]=t3; times[row*5+4]=t4;
        }
    }
    pipe.Destroy();
}
#endif
extern "C" __global__ __aicore__ void incremental_topk_kernel(GM_ADDR scores, GM_ADDR oldIds,
    GM_ADDR outScores, GM_ADDR outIds, GM_ADDR ticks, uint32_t rows, uint32_t added, uint32_t profile) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        if (profile) Run<true>(scores,oldIds,outScores,outIds,ticks,rows,added);
        else Run<false>(scores,oldIds,outScores,outIds,ticks,rows,added);
    }
#endif
}
