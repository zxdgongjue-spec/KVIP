// Only QK -> ReLU -> weighted head reduction -> sortable score. No LI tail.
#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "context.h"
struct QLITilingData {
    uint32_t bSize,tSize,gSize,s1Size,s2Size,sparseCount,usedCoreNum,blockSize,maxBlockNumPerBatch,sparseMode;
    uint32_t keyStride0=16384,keyDequantScaleStride0=128;
    uint32_t resident,splitQ,aicCount;
};
#include "indexer.h"
#endif
#if defined(__CCE_AICORE__)
__aicore__ inline void RunScoreRow(
    GM_ADDR bindings,uint32_t rows,uint32_t stride,uint32_t heads,uint32_t logicalGroup,bool segmentFence=false) {

    using namespace AscendC;using namespace QLICommon;
    // Native M/Fixpipe layout rounds a Q1 tail to two Q. qlen remains one;
    // the odd AIV acknowledges CV/VC and does not reduce/write dummy scores.
    uint32_t physicalGroup=logicalGroup==1?2:logicalGroup;
    TPipe pipe;auto a=reinterpret_cast<__gm__ uint64_t*>(bindings);
#define P(i) reinterpret_cast<GM_ADDR>(a[i])
    using Type=QLIType<fp8_e4m3fn_t,fp8_e4m3fn_t,int32_t,true,LI_LAYOUT::BSND,LI_LAYOUT::PA_BSND,
                      bfloat16_t,float,float,uint16_t>;
    QLIKernel::QuantLightningIndexerKernel<Type> op;
    op.scoreOnly_=true;op.scoreQueryGroup_=physicalGroup;
    op.runtimeShape_={};op.runtimeShape_.batch=rows;op.runtimeShape_.history=stride;
    op.runtimeShape_.resident=stride;op.runtimeShape_.indexHeads=heads;
    op.vectorService.runtimeShape_=op.runtimeShape_;
    QLITilingData tiling{rows,rows*physicalGroup,heads,physicalGroup,stride,2048,rows,128,
                       stride/128,0,16384,128,stride,0,rows};
    op.Init(P(0),P(1),P(2),P(3),P(4),P(5),P(6),P(7),P(9),P(12),&tiling,&pipe);
    op.Process();
    if(segmentFence) {
        // ProcessMain drains VC0/VC1 for AIV0. AIV1's two credits must also
        // be consumed before another piece reuses the same CV/VC channels.
        // Hold both AIVs at a separate boundary flag until AIC drained all
        // four credits; otherwise a fast AIV can re-prime the old channels.
        constexpr uint32_t boundary=6;
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<ConstInfo::QLI_SYNC_MODE4,PIPE_FIX>(ConstInfo::CROSS_VC_EVENT+ConstInfo::AIV0_AIV1_OFFSET);
            CrossCoreWaitFlag<ConstInfo::QLI_SYNC_MODE4,PIPE_FIX>(ConstInfo::CROSS_VC_EVENT+1+ConstInfo::AIV0_AIV1_OFFSET);
            CrossCoreSetFlag<ConstInfo::QLI_SYNC_MODE4,PIPE_FIX>(boundary);
            CrossCoreSetFlag<ConstInfo::QLI_SYNC_MODE4,PIPE_FIX>(boundary+ConstInfo::AIV0_AIV1_OFFSET);
        } else {CrossCoreWaitFlag<ConstInfo::QLI_SYNC_MODE4,PIPE_V>(boundary);}
    }
    pipe.Destroy();
#undef P
}
#endif
extern "C" __global__ __aicore__ void li_qk_score(
    GM_ADDR bindings,uint32_t rows,uint32_t stride,uint32_t heads,uint32_t logicalGroup) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    RunScoreRow(bindings,rows,stride,heads,logicalGroup);
#endif
}
// All 30 groups advance their own slot list within one device launch.
// AIC and its two AIVs read the same klen and skip empty slots together.
extern "C" __global__ __aicore__ void li_qk_score_balanced(
    GM_ADDR bindingList,uint32_t slots,uint32_t rows,uint32_t stride,uint32_t heads,uint32_t logicalGroup) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    uint32_t core=AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {core/=2;}
    if(core>=rows)return;
    auto list=reinterpret_cast<__gm__ uint64_t*>(bindingList);
    for(uint32_t slot=0;slot<slots;++slot) {
        auto binding=reinterpret_cast<__gm__ uint64_t*>(list[slot]);
        auto lengths=reinterpret_cast<__gm__ uint32_t*>(binding[6]);
        if(!lengths[core])continue;
        RunScoreRow(reinterpret_cast<GM_ADDR>(binding),rows,stride,heads,logicalGroup,true);
    }
#endif
}
