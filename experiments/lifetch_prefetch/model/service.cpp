#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "prefetch.h"
#endif

extern "C" __global__ __aicore__ void prefetch_service_model(GM_ADDR binding) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        using namespace AscendC;
        const auto args = reinterpret_cast<__gm__ uint64_t*>(binding);
        auto ctx = reinterpret_cast<__gm__ rotating::Context*>(args[0]);
        const auto shape = rotating::LoadRuntimeShape(ctx);
        const uint32_t lane = GetBlockIdx();
        const uint32_t epoch = args[1];
        TPipe pipe;
        TBuf<TPosition::VECCALC> scratch, flags, rows;
        pipe.InitBuffer(scratch, 32000);
        pipe.InitBuffer(flags, 40960);
        pipe.InitBuffer(rows, 8448);
#if KVIP_SERVICE_UB_CANDIDATES
        TBuf<TPosition::VECCALC> candidates;
        pipe.InitBuffer(candidates, 512);
        auto local = candidates.Get<uint32_t>();
        GlobalTensor<uint32_t> source;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(args[2]) + lane * 128);
        DataCopy(local, source, 128);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID3);WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
#endif
        rotating::FastService<true,true>(shape.batch, shape.history, shape.resident, shape.rounds,
            shape.serviceLanes, shape.tokenBytes, shape.postThreads, shape.postBank, shape.batchedRows,
            shape.scanTokens, reinterpret_cast<GM_ADDR>(ctx), epoch, lane / 2, lane,
            scratch.Get<uint32_t>(), flags.Get<uint16_t>(), rows.Get<uint32_t>(), shape.history / 2, nullptr
#if KVIP_SERVICE_UB_CANDIDATES
            , reinterpret_cast<const __ubuf__ uint32_t*>(local.GetPhyAddr())
#endif
        );
        pipe.Destroy();
    }
#endif
}
