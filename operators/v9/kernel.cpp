#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "rotation_abi.h"
#include "contract_device.h"
#include "kernel_timeline.h"
struct QLITilingData {
    uint32_t bSize=SIM_BATCH,tSize=SIM_BATCH*SIM_QUERIES,gSize=64,s1Size=SIM_QUERIES,s2Size=SIM_HISTORY;
    uint32_t sparseCount=2048,usedCoreNum=rotating::kLiCores,blockSize=128,maxBlockNumPerBatch=SIM_HISTORY/128,sparseMode=V9_CONTRACT_MODE ? 3 : 0;
    uint32_t keyStride0=16384,keyDequantScaleStride0=128;
};
#define SIM_STORAGE_HISTORY SIM_HISTORY
#include INSTALLED_LI_HEADER
#endif
extern "C" __global__ __aicore__ void v9_li_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    using namespace AscendC; using namespace QLICommon;
    uint64_t kernelStarted=GetSystemCycle();
    auto a=reinterpret_cast<__gm__ uint64_t*>(bindings);
#if V9_CONTRACT_MODE
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(a[14]);
    auto contract=rotating::ContractOf(ctx);
    if(rotating::ContractDisabled(contract)) return;
#endif
    TPipe pipe;
    using Type=QLIType<fp8_e4m3fn_t,fp8_e4m3fn_t,int32_t,true,
        V9_CONTRACT_MODE ? LI_LAYOUT::TND : LI_LAYOUT::BSND,LI_LAYOUT::PA_BSND,
        bfloat16_t,float,float,uint16_t>;
    QLIKernel::QuantLightningIndexerKernel<Type> op;
    QLITilingData tiling;
#if V9_CONTRACT_MODE
    tiling.maxBlockNumPerBatch=contract->index_columns;
#endif
#define P(i) reinterpret_cast<GM_ADDR>(a[i])
    op.Init(P(0),P(1),P(2),P(3),P(4),P(5),P(6),P(7),P(9),P(12),&tiling,&pipe);
#if V9_CONTRACT_MODE
    op.SetMappedContext(P(14));
#endif
    if ASCEND_IS_AIV {
        op.vectorService.SetPrefixStorage(P(13));
        op.vectorService.SetRotationContext(P(14));
    }
    op.Process(); pipe.Destroy();
#if V9_CONTRACT_MODE
    if ASCEND_IS_AIV {
        rotating::RecordKernelSpan(ctx,32+GetBlockIdx(),kernelStarted,GetSystemCycle());
    } else {
        rotating::RecordKernelSpan(ctx,256+GetBlockIdx(),kernelStarted,GetSystemCycle());
    }
#endif
#undef P
#endif
}
