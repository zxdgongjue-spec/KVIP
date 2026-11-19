// Model-only probe of the actual Cold consumer. No Service or Fetch is run.
#include "../../csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer.cpp"
extern "C" __global__ __aicore__ void v9_predicate_probe(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIV {
        using namespace AscendC; using namespace QLICommon;
        auto args = reinterpret_cast<__gm__ uint64_t*>(bindings);
        auto shape = rotating::LoadRuntimeShape(reinterpret_cast<__gm__ rotating::Context*>(args[14]));
        TPipe pipe;
        using Type=QLIType<fp8_e4m3fn_t,fp8_e4m3fn_t,int32_t,true,LI_LAYOUT::BSND,LI_LAYOUT::PA_BSND,
            bfloat16_t,float,float,uint16_t>;
        QLIKernel::QuantLightningIndexerKernel<Type> op;
        op.runtimeShape_=shape; op.rotationContext_=reinterpret_cast<GM_ADDR>(args[14]);
        op.sequenceSplit_=true; op.precomputedScore_=true;
        QLITilingData tiling{shape.batch,shape.batch*4,shape.indexHeads,4,shape.history,2048,
            shape.liCores,128,shape.history/128,0,16384,128,shape.resident,0,shape.aicCount};
#define PROBE_P(i) reinterpret_cast<GM_ADDR>(args[i])
        op.Init(PROBE_P(0),PROBE_P(1),PROBE_P(2),PROBE_P(3),PROBE_P(4),PROBE_P(5),PROBE_P(6),
            PROBE_P(7),PROBE_P(9),PROBE_P(12),&tiling,&pipe);
        op.vectorService.SetPrefixStorage(PROBE_P(13));
        op.vectorService.SetRotationContext(PROBE_P(14));
        op.vectorService.AllocEventID();
        uint32_t owner=GetBlockIdx()/4, shard=(GetBlockIdx()/2)%2;
        op.vectorService.LoadSequenceSplitThresholds(owner);
        op.vectorService.PrepareRotationProgress();
        uint32_t half=(shape.history-shape.resident)/256;
        uint32_t begin=shape.resident/128+shard*half, end=begin+half;
        for(uint32_t tile=begin;tile<end;++tile) {
            RunInfo info{}; info.loop=tile-begin; info.bIdx=owner; info.s2Idx=tile;
            info.isLastS2InnerLoop=tile+1==end;
            op.vectorService.ProcessPrecomputedVec1(info,shard,tile+1==end);
        }
        op.vectorService.FreeEventID();
        pipe.Destroy();
#undef PROBE_P
    }
#endif
}
