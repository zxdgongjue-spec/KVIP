// Native URMA large-batch entry points; the BS16 kernel stays a separate
// translation unit. Each block is a resident group of one cube and two vectors.
#define KVIP_URMA_WINDOW_LI 1
#define KVIP_LARGE_BATCH_LI 1
#include "large_batch_li.h"
#include "terminal/window_types.h"
#include "terminal/terminal_runtime_abi.h"

#if defined(__CCE_AICORE__)
namespace QLIKernel {
template <class T> __aicore__ inline void QuantLightningIndexerKernel<T>::ProcessLargeThreshold(uint32_t q) {
    CalcGS1LoopParams(assignedRow_);
    CalcS2LoopParams(assignedRow_, 0);
    QLICommon::RunInfo info;
    CalcRunInfo(0, 0, info, 0, 0);
    vectorService.AllocEventID();
    vectorService.ProcessResidentTopK(info, q);
    vectorService.FreeEventID();
    PipeBarrier<PIPE_ALL>();
}

// Fixpipe writes these two private banks at the start of each vector's UB.
// Rank and Post use different buffers reserved by the same live TPipe.
__aicore__ inline void PrimeLargeBanks() {
    using Flags = QLICommon::ConstInfo;
    CrossCoreSetFlag<Flags::QLI_SYNC_MODE4, PIPE_V>(Flags::CROSS_VC_EVENT);
    CrossCoreSetFlag<Flags::QLI_SYNC_MODE4, PIPE_V>(Flags::CROSS_VC_EVENT + 1);
}

template <class T>
__aicore__ inline void QuantLightningIndexerKernel<T>::ProcessLargeScore(uint32_t begin, uint32_t end, bool cold,
                                                                         bool loadThreshold, bool banksPrimed) {
    if (begin == end) {
        return;
    }
    using Flags = QLICommon::ConstInfo;
    constexpr uint32_t acknowledged = 4; // 0..1 return the banks; 2..3 publish the results.
    CalcGS1LoopParams(assignedRow_);
    CalcS2LoopParams(assignedRow_, 0);
    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        if (cold && loadThreshold) {
            vectorService.LoadSequenceSplitThresholds(assignedRow_);
        }
        if (!banksPrimed) PrimeLargeBanks();
    } else {
        matmulService.AllocEventID();
    }
    uint32_t loop = 0, scaleLoop = 0;
    ScanSequenceHalf(begin, end, cold ? 0 : 2, loop, scaleLoop, true);
    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
        QLICommon::ColdStamp<HardEvent::V_S>(coldDrain_, 0);
        CrossCoreWaitFlag<Flags::QLI_SYNC_MODE4, PIPE_V>(acknowledged);
        QLICommon::ColdStamp<HardEvent::V_S>(coldDrain_, 1);
    } else {
        matmulService.FreeEventID();
        // Balance both return credits of both vector lanes, including an
        // untouched ping/pong bank on a one-tile job, before the next job.
        QLICommon::ColdStamp<HardEvent::FIX_S>(coldDrain_, 0);
        for (uint32_t v = 0; v < 2; ++v) {
            CrossCoreWaitFlag<Flags::QLI_SYNC_MODE4, PIPE_FIX>(Flags::CROSS_VC_EVENT +
                                                               v * Flags::AIV0_AIV1_OFFSET);
            CrossCoreWaitFlag<Flags::QLI_SYNC_MODE4, PIPE_FIX>(Flags::CROSS_VC_EVENT +
                                                               v * Flags::AIV0_AIV1_OFFSET + 1);
        }
        QLICommon::ColdStamp<HardEvent::FIX_S>(coldDrain_, 1);
        // These flags, separate from both score buffers, acknowledge both
        // credits. A fast V cannot prime a still-set credit for the next job.
        CrossCoreSetFlag<Flags::QLI_SYNC_MODE4, PIPE_FIX>(acknowledged);
        CrossCoreSetFlag<Flags::QLI_SYNC_MODE4, PIPE_FIX>(acknowledged + Flags::AIV0_AIV1_OFFSET);
    }
    PipeBarrier<PIPE_ALL>();
    // Measure empty brackets on the same role and clock. This quantifies the
    // timer/fence floor separately from the cost of deliberately draining work.
    if (coldDrain_) {
        QLICommon::ColdStamp<HardEvent::MTE2_S>(coldDrain_, 2);
        QLICommon::ColdStamp<HardEvent::MTE2_S>(coldDrain_, 3);
        if ASCEND_IS_AIV {
            QLICommon::ColdStamp<HardEvent::V_S>(coldDrain_, 4);
            QLICommon::ColdStamp<HardEvent::V_S>(coldDrain_, 5);
            QLICommon::ColdStamp<HardEvent::MTE3_S>(coldDrain_, 6);
            QLICommon::ColdStamp<HardEvent::MTE3_S>(coldDrain_, 7);
            QLICommon::ColdStamp<HardEvent::V_S>(coldDrain_, 10);
            QLICommon::ColdStamp<HardEvent::V_S>(coldDrain_, 11);
        } else {
            QLICommon::ColdStamp<HardEvent::M_S>(coldDrain_, 4);
            QLICommon::ColdStamp<HardEvent::M_S>(coldDrain_, 5);
            QLICommon::ColdStamp<HardEvent::FIX_S>(coldDrain_, 6);
            QLICommon::ColdStamp<HardEvent::FIX_S>(coldDrain_, 7);
            QLICommon::ColdStamp<HardEvent::FIX_S>(coldDrain_, 10);
            QLICommon::ColdStamp<HardEvent::FIX_S>(coldDrain_, 11);
        }
    }
}
} // namespace QLIKernel

namespace {
using namespace AscendC;
using namespace QLICommon;
using Type = QLIType<fp8_e4m3fn_t, fp8_e4m3fn_t, int32_t, true, LI_LAYOUT::BSND, LI_LAYOUT::PA_BSND, bfloat16_t,
                     float, float, uint16_t>;
using Indexer = QLIKernel::QuantLightningIndexerKernel<Type>;

struct Inputs {
    __gm__ uint64_t* ptrs;
    __gm__ rotating::Context* ctx;
    __gm__ pipeline_ab::Args* stream;
    __gm__ uint32_t* flags;
    __gm__ uint32_t* stagedMisses;
    __gm__ uint64_t* coldProbe;
    rotating::RuntimeShape shape;
    uint32_t core, vecLane, helpers;

    __aicore__ inline Inputs(GM_ADDR bindings, GM_ADDR terminal) {
        ptrs = reinterpret_cast<__gm__ uint64_t*>(bindings);
        ctx = reinterpret_cast<__gm__ rotating::Context*>(ptrs[14]);
        auto binding = reinterpret_cast<__gm__ terminal_runtime::Binding*>(terminal);
        stream = reinterpret_cast<__gm__ pipeline_ab::Args*>(binding->pipeline);
        flags = reinterpret_cast<__gm__ uint32_t*>(binding->reserved[2]);
        coldProbe = reinterpret_cast<__gm__ uint64_t*>(binding->reserved[8]);
        stagedMisses = reinterpret_cast<__gm__ uint32_t*>(binding->reserved[4]);
        shape = rotating::LoadRuntimeShape(ctx);
        if ASCEND_IS_AIV {
            core = GetBlockIdx() / 2;
            vecLane = GetBlockIdx() % 2;
        } else {
            core = GetBlockIdx();
            vecLane = 0;
        }
        helpers = shape.liCores - shape.batch;
    }

    __aicore__ inline GM_ADDR P(uint32_t i) const {
        return reinterpret_cast<GM_ADDR>(ptrs[i]);
    }

    __aicore__ inline uint32_t End(uint32_t row) const {
        uint32_t end = shape.history;
        if (ptrs[6]) {
            end = Min(end, reinterpret_cast<__gm__ uint32_t*>(ptrs[6])[row]);
        }
        if (ctx->queryEnds) {
            uint32_t upper = 0;
            auto q = reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds) + 4 * row;
            for (uint32_t j = 0; j < 4; ++j) {
                upper = Max(upper, q[j]);
            }
            end = Min(end, upper);
        }
        return end;
    }

    __aicore__ inline __gm__ uint32_t* Line(uint32_t row, uint32_t slot) const {
        return flags + (row * 8 + slot) * 32;
    }

    __aicore__ inline void Signal(uint32_t row, uint32_t slot, uint32_t value) const {
        auto p = Line(row, slot);
        *p = value;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(p), 128);
    }

    __aicore__ inline void Wait(uint32_t row, uint32_t slot, uint32_t value) const {
        auto p = Line(row, slot);
        do {
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(p), 128);
        } while (*reinterpret_cast<__gm__ volatile uint32_t*>(p) < value);
    }

    __aicore__ inline void Init(Indexer& op, TPipe& pipe, uint32_t row, bool threshold) const {
        QLITilingData tiling{shape.batch,
                             shape.batch * 4,
                             shape.indexHeads,
                             4,
                             shape.history,
                             2048,
                             shape.liCores,
                             128,
                             shape.history / 128,
                             0,
                             16384,
                             128,
                             shape.resident,
                             0,
                             shape.aicCount};
        op.runtimeShape_ = shape;
        op.rotationContext_ = P(14);
        op.assignedRow_ = row;
        op.sequenceSplit_ = true;
        op.sequenceTopKOnly_ = threshold;
        if ASCEND_IS_AIV {
            op.vectorService.SetStreamingWorkspace(P(15));
        }
        op.Init(P(0), P(1), P(2), P(3), P(4), P(5), P(6), P(7), P(9), P(12), &tiling, &pipe);
        if ASCEND_IS_AIV {
            op.vectorService.SetPrefixStorage(P(13));
            op.vectorService.SetRotationContext(P(14));
        }
    }

    __aicore__ inline bool Trace() const {
        return stream->times && (stream->prefetchCount & 0x80000000U);
    }

    __aicore__ inline void Stamp(uint32_t row, uint32_t q, uint32_t slot, uint64_t value) const {
        auto times = reinterpret_cast<__gm__ uint64_t*>(stream->times) + shape.batch * 64 + (row * 4 + q) * 16;
        times[slot] = value;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(times), 128);
    }
};

__aicore__ inline void Score(const Inputs& input, uint32_t row, large_li::Slice slice, bool cold) {
    if (slice.begin == slice.end) {
        return;
    }
    TPipe pipe;
    Indexer op;
    input.Init(op, pipe, row, false);
    op.ProcessLargeScore(slice.begin, slice.end, cold);
    pipe.Destroy();
}

struct RowPlan {
    uint32_t row, start, end, tiles, quota;
    large_li::Slice recent;
};

template <bool Cold> __aicore__ inline RowPlan RowCost(const Inputs& input, uint32_t row) {
    uint32_t upper = input.End(row);
    RowPlan plan{};
    plan.row = row;
    if constexpr (Cold) {
        plan.start = input.shape.resident / 128;
        plan.end = Max(plan.start, large_li::CeilDiv(upper, 128));
    } else {
        plan.end = large_li::CeilDiv(Min(input.shape.resident, upper), 128);
        auto recent = kvip_recent_index::Tiles(kvip_recent_index::RecentRange(input.ctx, row, upper));
        plan.recent = {recent.begin, recent.end};
    }
    plan.tiles = plan.end - plan.start;
    return plan;
}

// Read the immutable per-request lengths at kernel entry, without waiting for
// a peer. Rank active work across the batch, then balance each helper and its
// at most seven rows. Owners and helpers derive identical physical boundaries.
template <bool Cold>
__aicore__ inline uint32_t Plan(const Inputs& input, bool helper, RowPlan* plans, uint32_t& own) {
    own = 0;
    if (!input.helpers) {
        plans[0] = RowCost<Cold>(input, input.core);
        return 1;
    }
    uint32_t costs[32], order[32];
    for (uint32_t row = 0; row < input.shape.batch; ++row) {
        auto plan = RowCost<Cold>(input, row);
        costs[row] = plan.tiles + plan.recent.end - plan.recent.begin;
    }
    uint32_t same = 1;
    while (same < input.shape.batch && costs[same] == costs[0]) ++same;
    if (same == input.shape.batch) {
        const uint32_t first = helper ? input.core - input.shape.batch : input.core % input.helpers;
        uint32_t count = 0, maxFixed = 0;
        for (uint32_t row = first; row < input.shape.batch; row += input.helpers) {
            auto& plan = plans[count];
            if constexpr (Cold) {
                plan = {};
                plan.row = row;
                plan.start = input.shape.resident / 128;
                plan.tiles = costs[0];
                plan.end = plan.start + plan.tiles;
            } else {
                plan = RowCost<false>(input, row);
                maxFixed = Max(maxFixed, plan.recent.end - plan.recent.begin);
            }
            if (!helper && row == input.core) own = count;
            ++count;
        }
        // For N equal totals T, the helper must process N*(T-owner) tiles.
        const uint32_t budget = Max(maxFixed, large_li::CeilDiv(count * costs[0], count + 1));
        for (uint32_t i = 0; i < count; ++i) plans[i].quota = costs[0] - budget;
        return count;
    }
    large_li::RankRows(costs, input.shape.batch, order);
    // A primary finds its rank so that it joins exactly its assigned helper.
    uint32_t rank = 0;
    if (!helper) {
        while (order[rank] != input.core) {
            ++rank;
        }
    }
    uint32_t first = helper ? input.core - input.shape.batch : rank % input.helpers;
    uint32_t total[7], fixed[7], count = 0;
    for (uint32_t i = first; i < input.shape.batch; i += input.helpers) {
        auto& plan = plans[count];
        plan = RowCost<Cold>(input, order[i]);
        fixed[count] = plan.recent.end - plan.recent.begin;
        total[count] = costs[plan.row];
        if (!helper && plan.row == input.core) {
            own = count;
        }
        ++count;
    }
    uint32_t budget = large_li::OwnerBudget(total, fixed, count);
    for (uint32_t i = 0; i < count; ++i) {
        plans[i].quota = large_li::SuffixQuota(plans[i].tiles, fixed[i], budget);
    }
    return count;
}

__aicore__ inline void Hot(const Inputs& input, const RowPlan& plan, bool helper) {
    uint32_t cut = plan.end - plan.quota;
    if (!helper) {
        if ASCEND_IS_AIV {
            if (!input.vecLane) {
                for (uint32_t f = 0; f < 8; ++f) {
                    input.Signal(plan.row, f, 0);
                }
            }
        }
        if (cut || plan.recent.begin != plan.recent.end) {
            // Hot and recent ranges share a Q row and the same UB/L1 layout.
            TPipe pipe;
            Indexer op;
            input.Init(op, pipe, plan.row, false);
            op.ProcessLargeScore(0, cut, false);
            op.ProcessLargeScore(plan.recent.begin, plan.recent.end, false);
            pipe.Destroy();
        }
    } else {
        Score(input, plan.row, {cut, plan.end}, false);
    }
}

__aicore__ inline void Threshold(const Inputs& input, uint32_t row, bool helper) {
    if ASCEND_IS_AIV {
        const uint32_t first = input.vecLane * 2 + uint32_t(helper);
        const uint32_t count = !helper && row >= input.helpers ? 2 : 1;
        TPipe pipe;
        Indexer op;
        input.Init(op, pipe, row, true);
        for (uint32_t q = first; q < first + count; ++q) {
            if (input.Trace()) {
                input.Stamp(row, q, 0, GetSystemCycle());
            }
            op.ProcessLargeThreshold(q);
            if (input.Trace()) {
                input.Stamp(row, q, 1, GetSystemCycle());
            }
            input.Signal(row, q, 1);
        }
        pipe.Destroy();
    }
}

__aicore__ inline void PublishProgress(const Inputs& input, uint32_t row, uint32_t first, uint32_t count,
                                       uint32_t end) {
    auto progress = reinterpret_cast<__gm__ uint32_t*>(input.ctx->progress) + row * 4 * 32;
    for (uint32_t q = first; q < first + count; ++q) {
        progress[q * 32] = end * 128;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(progress + q * 32), 128);
    }
}

__aicore__ inline void JoinThreshold(const Inputs& input, uint32_t row) {
    for (uint32_t q = 0; q < 4; ++q) {
        input.Wait(row, q, 1);
    }
}

__aicore__ inline void ColdHelper(const Inputs& input, const RowPlan& plan, uint32_t epoch, uint32_t planCount) {
    const auto span = large_li::Window(plan.start, plan.end, input.shape.rounds, epoch);
    const uint32_t suffix = large_li::LaterSuffix(span, plan.start, plan.tiles, plan.quota, planCount, epoch);
    if (suffix) {
        Score(input, plan.row, {span.end - suffix, span.end}, true);
        if ASCEND_IS_AIV {
            input.Signal(plan.row, 6 + input.vecLane, epoch);
        }
    }
}

__aicore__ inline void FinishPair(const Inputs& input, const RowPlan& plan, Indexer& op, uint32_t epoch) {
    // The posting core needs both winner lists and both metadata writes. The
    // second vector may immediately start scoring its next range after this.
    auto line = input.Line(plan.row, 5);
    auto& vec = op.vectorService;
    if (input.vecLane) {
        if (input.Trace()) {
            line[1] = vec.incrementalCalls_;
            line[2] = vec.incrementalScanned_;
            line[3] = vec.incrementalCandidates_;
        }
        input.Signal(plan.row, 5, epoch);
        return;
    }
    input.Wait(plan.row, 5, epoch);
    if (input.Trace()) {
        auto event = reinterpret_cast<__gm__ rotating::Event*>(input.ctx->events) +
                     epoch * input.shape.serviceLanes + plan.row;
        auto stamp = reinterpret_cast<__gm__ uint64_t*>(event->pad);
        stamp[3] = GetSystemCycle();
        stamp[4] = uint64_t(vec.incrementalCalls_ + line[1]) |
                   (uint64_t(vec.incrementalScanned_ + line[2]) << 16) |
                   (uint64_t(vec.incrementalCandidates_ + line[3]) << 40);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(event), 128);
    }
    vec.PostWindow(epoch, 0);
}

__aicore__ inline void ColdOwner(const Inputs& input, const RowPlan& plan, uint32_t planCount) {
    JoinThreshold(input, plan.row);
    TPipe pipe;
    Indexer op;
    input.Init(op, pipe, plan.row, false);
    // The test binds only one BS32 request. Each role owns entire cache lines.
    // Record all valid tiles and the drain handshake after each of the three windows.
    const uint32_t tileLimit = large_li::CeilDiv(input.shape.history, 128);
    if (input.Trace() && input.coldProbe && input.shape.batch == 32 &&
        input.coldProbe[0] == plan.row && input.coldProbe[1] == tileLimit &&
        input.coldProbe[2] && input.shape.rounds == 4) {
        uint32_t role = 0;
        if ASCEND_IS_AIV { role = 1 + input.vecLane; }
        op.coldMask_ = uint32_t(input.coldProbe[2]) - 1;
        op.coldPhase_ = uint32_t(input.coldProbe[3]);
        op.coldLines_ = input.coldProbe + 16 + role * (tileLimit + 3) * 16;
    }
    const bool overlap = large_li::CanOverlapBanks(plan.tiles, plan.quota, input.shape.rounds);
    if ASCEND_IS_AIV {
        if (overlap) QLIKernel::PrimeLargeBanks();
        if (!input.vecLane) {
            if (input.Trace()) {
                uint64_t start = GetSystemCycle();
                for (uint32_t q = 0; q < 4; ++q) {
                    input.Stamp(plan.row, q, 2, start);
                }
            }
            PublishProgress(input, plan.row, 0, 4, plan.start);
            op.vectorService.AllocEventID();
            op.vectorService.RotateFetch(0, 0);
            op.vectorService.FreeEventID();
        }
    }
    bool thresholdLoaded = false;
    for (uint32_t epoch = 1; epoch < input.shape.rounds; ++epoch) {
        const auto span = large_li::Window(plan.start, plan.end, input.shape.rounds, epoch);
        const uint32_t suffix = large_li::LaterSuffix(span, plan.start, plan.tiles, plan.quota, planCount, epoch);
        const uint32_t cut = span.end - suffix;
        if (span.begin < cut) {
            if (op.coldLines_) {
                op.coldDrain_ = op.coldLines_ + (tileLimit + epoch - 1) * 16;
                op.coldDrain_[8] = span.begin;
                op.coldDrain_[9] = cut;
            }
            op.ProcessLargeScore(span.begin, cut, true, !thresholdLoaded, overlap);
            thresholdLoaded = true;
        }
        if ASCEND_IS_AIV {
            // After ProcessLargeScore acknowledges BOTH old credits per lane,
            // the cube may fill the next two banks while these lanes rank the
            // current window and its single owner rings the doorbell.
            if (overlap && epoch + 1 < input.shape.rounds) QLIKernel::PrimeLargeBanks();
            // One scorer always produces Q0/Q1; its sibling produces Q2/Q3.
            // Only the matching helper's suffix contributes to this pair.
            if (suffix) {
                input.Wait(plan.row, 6 + input.vecLane, epoch);
            }
            PublishProgress(input, plan.row, input.vecLane * 2, 2, span.end);
            op.vectorService.AllocEventID();
            op.vectorService.AdvanceLargeWindowPair(plan.row, epoch, span.end * 128);
            FinishPair(input, plan, op, epoch);
            op.vectorService.FreeEventID();
        }
    }
    if ASCEND_IS_AIV {
        if (input.stagedMisses) {
            // Probe once; a late row, an outstanding own Post or an ineligible Q
            // simply defers its exact classification to its original tail owner.
            auto last = reinterpret_cast<__gm__ rotating::Event*>(input.ctx->events) +
                        (input.shape.rounds - 1) * input.shape.serviceLanes;
            const uint32_t peer = (plan.row + input.shape.batch / 2) % input.shape.batch;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(last + peer), 128);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(last + plan.row), 128);
            auto events = reinterpret_cast<__gm__ volatile rotating::Event*>(last);
            if (!events[peer].complete && events[plan.row].complete) {
                auto ends = reinterpret_cast<__gm__ const uint32_t*>(input.ctx->queryEnds);
                const uint32_t scan = large_li::ScanHistory(input.shape, ends, plan.row);
                op.vectorService.AllocEventID();
                for (uint32_t q = 2 * input.vecLane; q < 2 * input.vecLane + 2; ++q) {
                    const uint32_t query = plan.row * 4 + q;
                    const uint32_t valid = ends ? Min(ends[query], input.shape.history) : input.shape.history;
                    const uint32_t staged = op.vectorService.StagePublishedLargeMisses(
                        query, scan, valid, input.stagedMisses);
                    if (staged) input.Signal(plan.row, q, staged);
                }
                op.vectorService.FreeEventID();
            }
        }
    }
    pipe.Destroy();
    if ASCEND_IS_AIV {
        if (!input.vecLane && input.Trace()) {
            uint64_t end = GetSystemCycle();
            for (uint32_t q = 0; q < 4; ++q) {
                input.Stamp(plan.row, q, 3, end);
            }
        }
    }
    if (op.coldLines_) {
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(op.coldLines_ + plan.start * 16),
                                     uint64_t(tileLimit + 3 - plan.start) * 128);
    }
}
} // namespace
#endif

extern "C" __global__ __aicore__ void v9_li_large_hot(GM_ADDR bindings, GM_ADDR terminal) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    Inputs input(bindings, terminal);
    bool helper = input.core >= input.shape.batch;
    RowPlan plans[7];
    uint32_t own;
    uint32_t planCount = Plan<false>(input, helper, plans, own);
    if (helper) {
        for (uint32_t i = 0; i < planCount; ++i) {
            Hot(input, plans[i], true);
        }
    } else {
        Hot(input, plans[own], false);
    }
#endif
}

extern "C" __global__ __aicore__ void v9_li_large_threshold_cold(GM_ADDR bindings, GM_ADDR terminal) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    Inputs input(bindings, terminal);
    bool helper = input.core >= input.shape.batch;
    uint32_t first = helper ? input.core - input.shape.batch : input.core;
    Threshold(input, first, helper);
    RowPlan plans[7];
    uint32_t own;
    uint32_t planCount = Plan<true>(input, helper, plans, own);
    if (!helper) {
        ColdOwner(input, plans[own], planCount);
    } else {
        // Finish the current window for every assigned row before moving to
        // the next window. Rows with no suffix never wait for this helper.
        bool ready[7]{};
        for (uint32_t epoch = 1; epoch < input.shape.rounds; ++epoch) {
            for (uint32_t i = 0; i < planCount; ++i) {
                if (!plans[i].quota) {
                    continue;
                }
                if (!ready[i]) {
                    JoinThreshold(input, plans[i].row);
                    ready[i] = true;
                }
                ColdHelper(input, plans[i], epoch, planCount);
            }
        }
    }
#endif
}
