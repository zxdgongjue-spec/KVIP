#include "kernel_operator.h"

#if defined(__CCE_AICORE__)
#include "a5_lidu_fused_batch_get_prepare_mtp_implementation.inc"
#endif

extern "C" __global__ __aicore__ void a5_lidu_fused_batch_get_prepare_mtp(
    GM_ADDR topkIndices,
    GM_ADDR actualSeqLengthsQuery,
    GM_ADDR reqPoolEntries,
    GM_ADDR tokenToSlotPool,
    GM_ADDR slotToTokenPool,
    GM_ADDR evictionCursors,
    GM_ADDR cacheTokens,
    GM_ADDR candidateLens,
    GM_ADDR missSlots,
    GM_ADDR missSlotBitmap,
    GM_ADDR residentBlockTable,
    GM_ADDR logicalSeqLengths,
    GM_ADDR profileEnabled,
    GM_ADDR destinationSlots,
    GM_ADDR missCounts,
    GM_ADDR indices,
    GM_ADDR offsets,
    GM_ADDR outputIndices,
    GM_ADDR residentSeqLengths,
    GM_ADDR tokenToSlotAlias,
    GM_ADDR slotToTokenAlias,
    GM_ADDR evictionCursorsAlias,
    GM_ADDR profiling,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
#if defined(__CCE_AICORE__)
    (void)tokenToSlotAlias;
    (void)slotToTokenAlias;
    (void)evictionCursorsAlias;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto remote = reinterpret_cast<__gm__ const uint32_t *>(tiling);
    A5LiduFusedBatchGetPrepareMtpTilingData tilingData{};
    tilingData.usedCoreNum = remote[0];
    tilingData.coresPerRequest = remote[1];
    tilingData.batchSize = remote[2];
    tilingData.packedQueryCount = remote[3];
    tilingData.poolSize = remote[4];
    tilingData.sourceCapacity = remote[5];
    tilingData.residentCapacity = remote[6];
    tilingData.residentBlockCapacity = remote[7];
    tilingData.layerId = remote[8];
    tilingData.numLayers = remote[9];
    tilingData.maxNumReqs = remote[10];
    tilingData.qcap = remote[11];
    AscendC::TPipe pipe;
    A5LiduFusedBatchGetPrepareMtpKernel op(&pipe, &tilingData);
    op.Init(
        topkIndices, actualSeqLengthsQuery, reqPoolEntries,
        tokenToSlotPool, slotToTokenPool, evictionCursors, cacheTokens,
        candidateLens, missSlots, missSlotBitmap, residentBlockTable,
        logicalSeqLengths, profileEnabled, destinationSlots, missCounts,
        indices, offsets, outputIndices, residentSeqLengths, profiling,
        workspace);
    const bool profileActive =
        *reinterpret_cast<__gm__ const int32_t *>(profileEnabled) != 0;
    const uint64_t wrapperStart = profileActive ? AscendC::GetSystemCycle() : 0;
    op.Process();
    const uint64_t wrapperEnd = profileActive ? AscendC::GetSystemCycle() : 0;
    const uint32_t core = AscendC::GetBlockIdx();
    const uint32_t row = core / tilingData.coresPerRequest;
    const uint32_t lane = core % tilingData.coresPerRequest;
    if (profileActive && row < tilingData.batchSize && lane < 8U) {
        constexpr uint32_t profileFields = 32;
        constexpr uint32_t profileLanes = 8;
        constexpr uint32_t cacheLineBytes = 64;
        constexpr uint32_t laneBytes = profileFields * sizeof(int64_t);
        AscendC::GlobalTensor<int64_t> profileWords;
        profileWords.SetGlobalBuffer(
            reinterpret_cast<__gm__ int64_t *>(profiling));
        const uint64_t wordBase =
            (static_cast<uint64_t>(row) * profileLanes + lane) *
            profileFields;
        profileWords.SetValue(
            wordBase, static_cast<int64_t>(wrapperEnd - wrapperStart));
        AscendC::GlobalTensor<uint8_t> profileBytes;
        profileBytes.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint8_t *>(profiling));
        const uint64_t byteBase = wordBase * sizeof(int64_t);
        for (uint32_t offset = 0; offset < laneBytes;
             offset += cacheLineBytes) {
            AscendC::DataCacheCleanAndInvalid<
                uint8_t,
                AscendC::CacheLine::SINGLE_CACHE_LINE,
                AscendC::DcciDst::CACHELINE_OUT>(
                profileBytes[byteBase + offset]);
        }
    }
    pipe.Destroy();
#endif
}
