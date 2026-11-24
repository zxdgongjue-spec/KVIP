// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

#if defined(__CCE_AICORE__)
#include "context.h"
struct QLITilingData {
    uint32_t bSize, tSize, gSize, s1Size, s2Size, sparseCount, usedCoreNum, blockSize, maxBlockNumPerBatch, sparseMode;
    uint32_t keyStride0 = 16384, keyDequantScaleStride0 = 128;
    uint32_t resident, splitQ, aicCount;
};
#include "indexer/indexer_vector.h"
#endif

extern "C" __global__ __aicore__ void v9_li_batch_pipeline_service(
    GM_ADDR bindings, uint32_t epoch, uint32_t end0, uint32_t end1) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        using namespace AscendC;
        using namespace QLICommon;
        auto arguments = reinterpret_cast<__gm__ uint64_t*>(bindings);
        auto ctx = reinterpret_cast<__gm__ rotating::Context*>(arguments[14]);
        const auto shape = rotating::LoadRuntimeShape(ctx);
        const uint32_t lane = GetBlockIdx();
        const uint32_t sub = lane % 2;
        const uint32_t core = rotating::ServiceCore(shape.batch, shape.splitQ, shape.serviceAics, epoch, lane / 2);

        // The caller's score-ready event has joined both P2 shards and all
        // four queries. This consumer alone writes the committed cursors.
        for (uint32_t b = lane; b < shape.batch; b += shape.serviceLanes) {
            for (uint32_t q = 0; q < 4; ++q) {
                auto ready = reinterpret_cast<__gm__ uint32_t*>(ctx->progress) + (b * 4 + q) * 32;
                ready[0] = end0;
                ready[16] = end1;
                kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ready), 128);
            }
        }

        TPipe pipe;
        using Type = QLIType<fp8_e4m3fn_t, fp8_e4m3fn_t, int32_t, true,
            LI_LAYOUT::BSND, LI_LAYOUT::PA_BSND, bfloat16_t, float, float, uint16_t>;
        QLIKernel::QLIVector<Type> op;
        ConstInfo constants{};
        constants.batchSize = shape.batch;
        constants.tSize = shape.batch * 4;
        constants.qSeqSize = constants.s1BaseSize = 4;
        constants.qHeadNum = constants.gSize = shape.indexHeads;
        constants.kHeadNum = 1;
        constants.sparseCount = 2048;
        constants.s2BaseSize = constants.kCacheBlockSize = 128;
        constants.kSeqSize = shape.history;
        constants.maxBlockNumPerBatch = shape.history / 128;
        constants.headDim = 128;
        constants.mBaseSize = 4 * shape.indexHeads;
        QLITilingData tiling{};
        op.InitParams(constants, &tiling, false);
        op.SetServiceBlock(core * 2 + sub);
        op.SetPrefixStorage(reinterpret_cast<GM_ADDR>(arguments[13]));
        op.SetRotationContext(reinterpret_cast<GM_ADDR>(arguments[14]));
        op.SetStreamingWorkspace(reinterpret_cast<GM_ADDR>(arguments[15]));
        op.InitBuffers(&pipe, false);
        // Window helpers manage their own events. Score credit initialization
        // would signal the same event IDs a second time in this AIV-only kernel.
        op.RotateFetch(epoch, rotating::SequenceHalfWindow(shape));
        pipe.Destroy();
    }
#endif
}
