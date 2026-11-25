#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "context.h"
#include "stage_trace.h"
#if !V9_UBMEM
#include "round_prepare.h"
#endif
struct QLITilingData {
    uint32_t bSize,tSize,gSize,s1Size,s2Size,sparseCount,usedCoreNum,blockSize,maxBlockNumPerBatch,sparseMode;
    uint32_t keyStride0=16384,keyDequantScaleStride0=128;
    uint32_t resident,splitQ,aicCount;
};
#include "../../lightning_indexer_with_fetch/op_kernel/indexer/indexer.h"
#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
#include "indexer/dynamic_mailbox.h"
#endif
namespace {
using QLIKernel::SequenceLiPhase;

__aicore__ inline bool HasUniformLengths(const rotating::RuntimeShape& shape, GM_ADDR q, GM_ADDR k) {
    auto query = reinterpret_cast<__gm__ const uint32_t*>(q);
    auto key = reinterpret_cast<__gm__ const uint32_t*>(k);
    for (uint32_t batch = 0; batch < shape.batch; ++batch) {
        if ((query && query[batch] != 4) || (key && key[batch] != shape.history)) return false;
    }
    return true;
}

template <SequenceLiPhase Phase, bool ExternalBody = false, bool OwnWindow = false>
__aicore__ inline void RunLi(GM_ADDR bindings, uint32_t parts = 2, uint32_t job = 0) {
    using namespace AscendC; using namespace QLICommon;
    auto a=reinterpret_cast<__gm__ uint64_t*>(bindings);
#define P(i) reinterpret_cast<GM_ADDR>(a[i])
#if KVIP_STAGE_TRACE
    constexpr bool traceHotSlot = Phase == SequenceLiPhase::Hot || Phase == SequenceLiPhase::UnifiedScore;
    constexpr bool traceColdSlot = Phase == SequenceLiPhase::Cold || Phase == SequenceLiPhase::PrecomputedCold ||
        Phase == SequenceLiPhase::DedicatedCold ||
        Phase == SequenceLiPhase::DynamicScore || Phase == SequenceLiPhase::DynamicService;
    const uint64_t stageTrace = (parts == 2 || Phase == SequenceLiPhase::Cold ||
        Phase == SequenceLiPhase::UnifiedScore)
        ? KVIP_TRACE_WORKSPACE(P(15)) : 0;
    constexpr uint32_t traceKernel = traceHotSlot ? kvip_trace::Hot :
        Phase == SequenceLiPhase::Threshold ? kvip_trace::Threshold :
        traceColdSlot ? kvip_trace::Cold : kvip_trace::Final;
#endif
#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
    if constexpr (Phase == SequenceLiPhase::DedicatedCold) {
        if ASCEND_IS_AIC {
            if (GetBlockIdx() >= kvip_dedicated_cold::kProducerGroups) return;
        }
    }
#endif
    KVIP_TRACE_RECORD(kernelScope, ExternalBody ? 0 : stageTrace, traceKernel, kvip_trace::KernelBody, 0, 0);
    KVIP_TRACE_RECORD(setupScope, stageTrace, traceKernel, kvip_trace::Setup, 0, 0);
    KVIP_TRACE_RECORD(shapeScope, stageTrace, traceKernel, kvip_trace::LiShapeSetup, 0, 0);
    auto shape = rotating::LoadRuntimeShape(reinterpret_cast<__gm__ rotating::Context*>(a[14]));
    bool splitSequence = rotating::UseSequenceSplit(shape) && HasUniformLengths(shape, P(5), P(6));
    if constexpr (Phase != SequenceLiPhase::Hot) {
        if (!splitSequence) return;
    }
    if (parts > 2 && !splitSequence) return;
    KVIP_TRACE_END(shapeScope);
    TPipe pipe;
    using Type=QLIType<fp8_e4m3fn_t,fp8_e4m3fn_t,int32_t,true,LI_LAYOUT::BSND,LI_LAYOUT::PA_BSND,
        bfloat16_t,float,float,uint16_t>;
    QLIKernel::QuantLightningIndexerKernel<Type> op;
    op.runtimeShape_ = shape;
#if KVIP_STAGE_TRACE
    op.stageTrace_ = stageTrace;
    op.traceKernel_ = traceKernel;
#endif
#if !V9_UBMEM
    op.rotationContext_ = P(14);
#endif
    op.sequenceSplit_ = splitSequence;
    op.sequenceTopKOnly_ = Phase == SequenceLiPhase::Threshold || Phase == SequenceLiPhase::Final;
    QLITilingData tiling{shape.batch,shape.batch*4,shape.indexHeads,4,shape.history,
        2048,shape.liCores,128,shape.history/128,0,16384,128,shape.resident,
        splitSequence ? 0U : shape.splitQ,shape.aicCount};
    if ASCEND_IS_AIV { op.vectorService.SetStreamingWorkspace(P(15)); }
    op.Init(P(0),P(1),P(2),P(3),P(4),P(5),P(6),P(7),P(9),P(12),&tiling,&pipe,parts);
    if ASCEND_IS_AIV {
        KVIP_TRACE_RECORD(bindingScope, stageTrace, traceKernel, kvip_trace::LiWorkspaceBind, 0, 0);
        op.vectorService.SetPrefixStorage(P(13));
        op.vectorService.SetRotationContext(P(14));
#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
        if constexpr (Phase == SequenceLiPhase::Cold) {
            if (parts > 2) op.vectorService.BindPartitionedCold(P(12), parts);
        } else if constexpr (Phase == SequenceLiPhase::DedicatedCold || Phase == SequenceLiPhase::DynamicScore ||
            Phase == SequenceLiPhase::DynamicService) {
            op.vectorService.BindDedicatedCold(P(12));
        }
#endif
    }
    KVIP_TRACE_END(setupScope);
#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
    if constexpr (Phase == SequenceLiPhase::DedicatedCold) {
        op.template ProcessDedicatedCold<OwnWindow>(P(12));
    } else if constexpr (Phase == SequenceLiPhase::DynamicScore) {
        op.ProcessDynamicScore(P(12), dynamic_li::Row(job), dynamic_li::Window(job), dynamic_li::Fragment(job));
    } else if constexpr (Phase == SequenceLiPhase::DynamicService) {
        if ASCEND_IS_AIV {
            op.vectorService.RunDynamicService(dynamic_li::Row(job), dynamic_li::Window(job),
                                               rotating::SequenceHalfWindow(shape));
        }
    } else
#endif
    if constexpr (Phase == SequenceLiPhase::Hot) {
        if (splitSequence) op.template ProcessSequenceSplit<Phase>(parts);
        else op.Process();
    } else {
        op.template ProcessSequenceSplit<Phase>(parts);
    }
    KVIP_TRACE_RECORD(teardownScope, stageTrace, traceKernel, kvip_trace::LiPipeDestroy, 0, 0);
    pipe.Destroy();
    KVIP_TRACE_END(teardownScope);
#undef P
}

#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
__aicore__ inline void ResetPartitionedColdProgress(GM_ADDR bindings, uint32_t parts) {
    using namespace AscendC;
    auto a = reinterpret_cast<__gm__ uint64_t*>(bindings);
    const auto shape = rotating::LoadRuntimeShape(reinterpret_cast<__gm__ rotating::Context*>(a[14]));
    auto compute = reinterpret_cast<GM_ADDR>(a[12]);
    auto raw = compute + kvip_cold_score::ScoreBytes(shape.batch, shape.history);
    const uint32_t words = parts * kvip_cold_score::kProgressWords;
    const uint32_t query = GetBlockIdx();
    if (query >= shape.batch * kvip_cold_score::kQueries) return;
    KVIP_TRACE_BEGIN(resetScope, KVIP_TRACE_WORKSPACE(reinterpret_cast<GM_ADDR>(a[15])),
        Threshold, ColdProgressReset, 0, query);
    GlobalTensor<uint32_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(raw) +
        kvip_cold_score::ProgressOffset(query, 0, parts));
    // Reset within the threshold dispatch before the score-only scratch is
    // reused on the next graph replay. Destroy this temporary UB before TopK.
    TPipe pipe;
    TBuf<TPosition::VECCALC> zeros;
    pipe.InitBuffer(zeros, words * sizeof(uint32_t));
    auto values = zeros.Get<uint32_t>();
    Duplicate(values, 0U, words);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(output, values, words);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    pipe.Destroy();
}

__aicore__ inline void ResetDedicatedColdProgress(GM_ADDR bindings) {
    using namespace AscendC;
    const auto a = reinterpret_cast<__gm__ uint64_t*>(bindings);
    const auto shape = rotating::LoadRuntimeShape(reinterpret_cast<__gm__ rotating::Context*>(a[14]));
    const uint32_t query = GetBlockIdx();
    if (query >= shape.batch * kvip_dedicated_cold::kQueries) return;
    KVIP_TRACE_BEGIN(resetScope, KVIP_TRACE_WORKSPACE(reinterpret_cast<GM_ADDR>(a[15])),
        Threshold, ColdProgressReset, 0, query);
    constexpr uint32_t words = (kvip_dedicated_cold::kRounds - 1) *
        kvip_dedicated_cold::kFragments * kvip_dedicated_cold::kProgressWords;
    const auto progress = reinterpret_cast<GM_ADDR>(a[12]) + kvip_cold_score::ScoreBytes(shape.batch, shape.history);
    GlobalTensor<uint32_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(progress) +
        kvip_dedicated_cold::ProgressOffset(query, 1, 0));
    TPipe pipe;
    TBuf<TPosition::VECCALC> zeros;
    pipe.InitBuffer(zeros, words * sizeof(uint32_t));
    const auto values = zeros.Get<uint32_t>();
    Duplicate(values, 0U, words);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(output, values, words);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    pipe.Destroy();
}
__aicore__ inline dynamic_li::Mailbox DynamicMailbox(GM_ADDR bindings) {
    const auto a = reinterpret_cast<__gm__ uint64_t*>(bindings);
    return {reinterpret_cast<GM_ADDR>(a[12]) +
        kvip_cold_score::ScoreBytes(dynamic_li::kRows, kvip_dedicated_cold::kHistory) +
        kvip_dedicated_cold::ProgressBytes()};
}

#endif
}
#endif

extern "C" __global__ __aicore__ void v9_li_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    RunLi<SequenceLiPhase::Hot>(bindings);
#endif
}

#if KVIP_HOT_SCORE_PARTITION
extern "C" __global__ __aicore__ void v9_li_kernel_hot_partitioned(GM_ADDR bindings, uint32_t parts) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    RunLi<SequenceLiPhase::Hot>(bindings, parts);
#endif
}

extern "C" __global__ __aicore__ void v9_li_kernel_threshold_cold_partitioned(GM_ADDR bindings, uint32_t parts) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIV {
        ResetPartitionedColdProgress(bindings, parts);
        RunLi<SequenceLiPhase::Threshold>(bindings);
    }
#endif
}

extern "C" __global__ __aicore__ void v9_li_kernel_cold_partitioned(GM_ADDR bindings, uint32_t parts) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    RunLi<SequenceLiPhase::Cold>(bindings, parts);
#endif
}

#if !V9_UBMEM
extern "C" __global__ __aicore__ void v9_li_kernel_threshold_dedicated_service(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIV {
        ResetDedicatedColdProgress(bindings);
        RunLi<SequenceLiPhase::Threshold>(bindings);
    }
#endif
}

extern "C" __global__ __aicore__ void v9_li_kernel_cold_dedicated_service(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    RunLi<SequenceLiPhase::DedicatedCold>(bindings);
#endif
}
// Reset the mailbox on the same stream for each graph invocation. Each
// physical lane clears two disjoint cache lines before normal Hot Score.
extern "C" __global__ __aicore__ void v9_li_kernel_hot_ready_service(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    auto bus = DynamicMailbox(bindings);
    const uint32_t block = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        bus.SendService(block, 0);
        bus.CompleteAiv(block, 0);
    } else {
        bus.SendScore(block, 0);
        bus.CompleteAic(block, 0);
    }
    RunLi<SequenceLiPhase::Hot>(bindings);
#endif
}

extern "C" __global__ __aicore__ void v9_li_kernel_cold_static_ready_service(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    RunLi<SequenceLiPhase::DedicatedCold, false, true>(bindings);
#endif
}

extern "C" __global__ __aicore__ void v9_li_kernel_dynamic_service(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    using namespace AscendC;
    auto bus = DynamicMailbox(bindings);
    const uint32_t block = GetBlockIdx();
#if KVIP_STAGE_TRACE
    const auto a = reinterpret_cast<__gm__ uint64_t*>(bindings);
    const uint64_t trace = KVIP_TRACE_WORKSPACE(reinterpret_cast<GM_ADDR>(a[15]));
#endif
    KVIP_TRACE_BEGIN(body, trace, Cold, KernelBody, 0, 0);
    if ASCEND_IS_AIC {
        if (block == 0) {
            dynamic_li::Dispatcher dispatcher;
            while (!dispatcher.Step(bus)) {}
            for (uint32_t group = 0; group < dynamic_li::kMix; ++group) {
                bus.SendScore(group, dynamic_li::kStop);
            }
            return;
        }
        uint32_t seen = 0;
        while (true) {
            const uint32_t job = bus.Score(block);
            if (job == dynamic_li::kStop) return;
            if (job == seen) continue;
            {
                KVIP_TRACE_BEGIN(scope, trace, Cold, DynamicScoreJob, dynamic_li::Window(job), job);
                RunLi<SequenceLiPhase::DynamicScore, true>(bindings, 2, job);
            }
            bus.CompleteAic(block, job);
            seen = job;
        }
    } else {
        ResetDedicatedColdProgress(bindings);
        RunLi<SequenceLiPhase::Threshold>(bindings);
        PipeBarrier<PIPE_ALL>();
        bus.CompleteAiv(block, dynamic_li::kResidentReady);
        uint32_t seenScore = 0, seenService = 0;
        while (true) {
            const uint32_t job = bus.Score(block / 2);
            if (job == dynamic_li::kStop) return;
            if (job != seenScore) {
                {
                    KVIP_TRACE_BEGIN(scope, trace, Cold, DynamicScoreJob, dynamic_li::Window(job), job);
                    RunLi<SequenceLiPhase::DynamicScore, true>(bindings, 2, job);
                }
                bus.CompleteAiv(block, job);
                seenScore = job;
                continue;
            }
            const uint32_t service = bus.Service(block);
            if (service != seenService) {
                {
                    KVIP_TRACE_BEGIN(scope, trace, Cold, DynamicServiceJob, dynamic_li::Window(service), service);
                    RunLi<SequenceLiPhase::DynamicService, true>(bindings, 2, service);
                    PipeBarrier<PIPE_ALL>();
                }
                bus.CompleteAiv(block, service);
                seenService = service;
            }
        }
    }
#endif
}
#endif

extern "C" __global__ __aicore__ void v9_li_kernel_score_t2(GM_ADDR bindings, uint32_t parts) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    RunLi<SequenceLiPhase::UnifiedScore>(bindings, parts);
#endif
}

extern "C" __global__ __aicore__ void v9_li_kernel_cold_t2(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIV { RunLi<SequenceLiPhase::PrecomputedCold>(bindings); }
#endif
}
#endif

extern "C" __global__ __aicore__ void v9_li_kernel_1(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIV { RunLi<SequenceLiPhase::Threshold>(bindings); }
#endif
}

extern "C" __global__ __aicore__ void v9_li_kernel_2(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    RunLi<SequenceLiPhase::Cold>(bindings);
#endif
}

extern "C" __global__ __aicore__ void v9_li_kernel_3(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIV { RunLi<SequenceLiPhase::Final>(bindings); }
#endif
}
