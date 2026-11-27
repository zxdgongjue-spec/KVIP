// Native URMA large-batch entry points; the BS16 kernel stays a separate
// translation unit. Each block is a resident group of one cube and two vectors.
#define KVIP_URMA_WINDOW_LI 1
#define KVIP_LARGE_BATCH_LI 1
#include "large_batch_li.h"
#include "terminal/window_types.h"
#include "terminal/terminal_runtime_abi.h"

#if defined(__CCE_AICORE__)
#include "indexer/large_window_policy.h"
#include "indexer/window_q1.h"

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
        CrossCoreWaitFlag<Flags::QLI_SYNC_MODE4, PIPE_V>(acknowledged);
    } else {
        matmulService.FreeEventID();
        // Balance both return credits of both vector lanes, including an
        // untouched ping/pong bank on a one-tile job, before the next job.
        for (uint32_t v = 0; v < 2; ++v) {
            CrossCoreWaitFlag<Flags::QLI_SYNC_MODE4, PIPE_FIX>(Flags::CROSS_VC_EVENT +
                                                               v * Flags::AIV0_AIV1_OFFSET);
            CrossCoreWaitFlag<Flags::QLI_SYNC_MODE4, PIPE_FIX>(Flags::CROSS_VC_EVENT +
                                                               v * Flags::AIV0_AIV1_OFFSET + 1);
        }
        // These flags, separate from both score buffers, acknowledge both
        // credits. A fast V cannot prime a still-set credit for the next job.
        CrossCoreSetFlag<Flags::QLI_SYNC_MODE4, PIPE_FIX>(acknowledged);
        CrossCoreSetFlag<Flags::QLI_SYNC_MODE4, PIPE_FIX>(acknowledged + Flags::AIV0_AIV1_OFFSET);
    }
    PipeBarrier<PIPE_ALL>();
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
    GM_ADDR deferredReady;
    rotating::RuntimeShape shape;
    uint32_t core, vecLane, helpers;

    __aicore__ inline Inputs(GM_ADDR bindings, GM_ADDR terminal) {
        ptrs = reinterpret_cast<__gm__ uint64_t*>(bindings);
        ctx = reinterpret_cast<__gm__ rotating::Context*>(ptrs[14]);
        auto binding = reinterpret_cast<__gm__ terminal_runtime::Binding*>(terminal);
        stream = reinterpret_cast<__gm__ pipeline_ab::Args*>(binding->pipeline);
        flags = reinterpret_cast<__gm__ uint32_t*>(binding->reserved[2]);
        stagedMisses = reinterpret_cast<__gm__ uint32_t*>(binding->reserved[4]);
        deferredReady = reinterpret_cast<GM_ADDR>(binding->reserved[5]);
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

__aicore__ inline void CaptureLastWindowCost(const Inputs& input, uint32_t row, uint32_t ready) {
    const uint32_t epoch = input.shape.rounds - 1;
    const uint32_t end = Min(ready, rotating::Trigger(input.shape.history, input.shape.resident,
                                                     input.shape.rounds, epoch) * 128);
    auto headers = reinterpret_cast<__gm__ uint64_t*>(rotating::TopKHeader(input.ctx));
    auto ends = reinterpret_cast<__gm__ const uint32_t*>(input.ctx->queryEnds);
    uint32_t cost[4];
    for (uint32_t q = 0; q < 4; ++q) {
        const uint32_t query = row * 4 + q;
        auto header = headers + query * 16;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(header), 128);
        const uint32_t prior = uint32_t(reinterpret_cast<__gm__ volatile uint64_t*>(header)[1]);
        const uint32_t valid = ends ? Min(end, ends[query]) : end;
        // Even with no new tiles, maintaining/exporting the old winners costs work.
        cost[q] = 1 + large_li::CeilDiv(valid > prior ? valid - prior : 0, 128);
    }
    const auto estimate = large_window_policy::Estimate(cost[0], cost[1], cost[2], cost[3]);
    auto line = fixed_window_q1::Line(input.deferredReady, row, 0);
    auto words = reinterpret_cast<__gm__ volatile uint32_t*>(line);
    words[0] = estimate.q1;
    words[1] = estimate.q2;
    words[2] = estimate.q4;
    kvaf_device::CleanCacheLines(line, 128);
}

template<bool DeferLastWindow, bool AutoPolicy>
__aicore__ inline void ColdOwner(const Inputs& input, const RowPlan& plan, uint32_t planCount) {
    JoinThreshold(input, plan.row);
    TPipe pipe;
    Indexer op;
    input.Init(op, pipe, plan.row, false);
    const bool overlap = large_li::CanOverlapBanks(plan.tiles, plan.quota, input.shape.rounds);
    if ASCEND_IS_AIV {
        if (overlap) QLIKernel::PrimeLargeBanks();
        if (!input.vecLane) {
            if constexpr (DeferLastWindow) {
                for (uint32_t q = 0; q < 4; ++q) {
                    fixed_window_q1::Publish(input.deferredReady, plan.row, q, 0);
                }
            }
            if constexpr (AutoPolicy) {
                if (!plan.row) fixed_window_q1::Publish(input.deferredReady, input.shape.batch, 0, 0);
            }
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
            if constexpr (DeferLastWindow) {
                if (epoch + 1 == input.shape.rounds) {
                    if constexpr (AutoPolicy) {
                        // W2 Post has completed for this row and no AIV has
                        // started to mutate its W3 TopK headers.
                        if (!input.vecLane) CaptureLastWindowCost(input, plan.row, span.end * 128);
                    }
                    break;
                }
            }
            op.vectorService.AllocEventID();
            op.vectorService.AdvanceLargeWindowPair(plan.row, epoch, span.end * 128);
            FinishPair(input, plan, op, epoch);
            op.vectorService.FreeEventID();
        }
    }
    if ASCEND_IS_AIV {
        if (!DeferLastWindow && input.stagedMisses) {
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
}

template<bool DeferLastWindow, bool AutoPolicy = false>
__aicore__ inline void ThresholdCold(GM_ADDR bindings, GM_ADDR terminal) {
    Inputs input(bindings, terminal);
    if constexpr (DeferLastWindow) {
        if (!input.deferredReady) return;
    }
    bool helper = input.core >= input.shape.batch;
    uint32_t first = helper ? input.core - input.shape.batch : input.core;
    Threshold(input, first, helper);
    RowPlan plans[7];
    uint32_t own;
    uint32_t planCount = Plan<true>(input, helper, plans, own);
    if (!helper) {
        ColdOwner<DeferLastWindow, AutoPolicy>(input, plans[own], planCount);
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
}

template<uint32_t Queries>
__aicore__ inline void FinishDeferredRow(const Inputs& input, uint32_t row, uint32_t first) {
    constexpr uint32_t rowQueries = large_window_policy::kQueries;
    const uint32_t epoch = input.shape.rounds - 1;
    const bool trace = input.Trace();
    // Every row's two score producers and optional suffix producers have
    // returned before this launch. All policies use that same cut.
    auto progress = reinterpret_cast<__gm__ uint32_t*>(input.ctx->progress) + (row * rowQueries + first) * 32;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(progress), 128);
    const uint32_t ready = *reinterpret_cast<__gm__ volatile uint32_t*>(progress);
    TPipe pipe;
    Indexer op;
    input.Init(op, pipe, row, false);
    auto& vec = op.vectorService;
    vec.AllocEventID();
    vec.template AdvanceLargeWindowQueries<Queries>(row, epoch, ready, first);
    if (trace) {
        auto result = fixed_window_q1::Signal(input.deferredReady, row, first);
        result->calls = vec.incrementalCalls_;
        result->scanned = vec.incrementalScanned_;
        result->candidates = vec.incrementalCandidates_;
    }
    for (uint32_t q = first; q < first + Queries; ++q) {
        fixed_window_q1::Signal(input.deferredReady, row, q)->queriesPerWorker = Queries;
        if (q) fixed_window_q1::Publish(input.deferredReady, row, q, 1);
    }
    if (!first) {
        for (uint32_t q = Queries; q < rowQueries; q += Queries) {
            fixed_window_q1::Wait(input.deferredReady, row, q);
        }
        if (trace) {
            uint32_t calls = vec.incrementalCalls_;
            uint32_t scanned = vec.incrementalScanned_;
            uint32_t candidates = vec.incrementalCandidates_;
            for (uint32_t q = Queries; q < rowQueries; q += Queries) {
                const auto result = fixed_window_q1::Signal(input.deferredReady, row, q);
                calls += result->calls;
                scanned += result->scanned;
                candidates += result->candidates;
            }
            auto event = reinterpret_cast<__gm__ rotating::Event*>(input.ctx->events) +
                epoch * input.shape.serviceLanes + row;
            auto stamp = reinterpret_cast<__gm__ uint64_t*>(event->pad);
            stamp[3] = GetSystemCycle();
            stamp[4] = uint64_t(calls) | (uint64_t(scanned) << 16) | (uint64_t(candidates) << 40);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(event), 128);
        }
        vec.PostLargeWindowRow(row, epoch);
        fixed_window_q1::Publish(input.deferredReady, row, 0, 1);
    }
    vec.FreeEventID();
    pipe.Destroy();
}

template<uint32_t Queries, uint32_t TeamLanes = large_window_policy::kQueries / Queries>
__aicore__ inline void FinishDeferredWindow(GM_ADDR bindings, GM_ADDR terminal) {
    Inputs input(bindings, terminal);
    if (!input.deferredReady) return;
    constexpr uint32_t teams = large_window_policy::kAivs / TeamLanes;
    const uint32_t block = GetBlockIdx();
    const uint32_t first = (block % TeamLanes) * Queries;
    if (first >= large_window_policy::kQueries) return;
    for (uint32_t row = block / TeamLanes; row < input.shape.batch; row += teams) {
        FinishDeferredRow<Queries>(input, row, first);
    }
}

__aicore__ inline void PlanDeferredWindow(const Inputs& input) {
    using namespace large_window_policy;
    Cost costs[kMaxRows];
    for (uint32_t row = 0; row < input.shape.batch; ++row) {
        auto line = fixed_window_q1::Line(input.deferredReady, row, 0);
        kvaf_device::CleanCacheLines(line, 128);
        auto words = reinterpret_cast<__gm__ volatile uint32_t*>(line);
        costs[row] = {words[0], words[1], words[2]};
    }
    uint32_t lanes[kAivs];
    Plan(costs, input.shape.batch, lanes);
    // The extra row keeps its plan lanes separate from the ready marker.
    // Use lines 2 and 3 for 64 assignments, leaving line 1 for the marker.
    auto line = fixed_window_q1::Line(input.deferredReady, input.shape.batch, 2);
    auto words = reinterpret_cast<__gm__ volatile uint32_t*>(line);
    for (uint32_t lane = 0; lane < kAivs; ++lane) words[lane] = lanes[lane];
    kvaf_device::CleanCacheLines(line, kAivs * sizeof(uint32_t));
    fixed_window_q1::Publish(input.deferredReady, input.shape.batch, 0, 1);
}

__aicore__ inline void FinishDeferredAuto(GM_ADDR bindings, GM_ADDR terminal) {
    Inputs input(bindings, terminal);
    if (!input.deferredReady) return;
    const uint32_t block = GetBlockIdx();
    if (block == 0) {
        PlanDeferredWindow(input);
    } else {
        fixed_window_q1::Wait(input.deferredReady, input.shape.batch, 0);
    }
    auto line = fixed_window_q1::Line(input.deferredReady, input.shape.batch, 2 + block / 32);
    kvaf_device::CleanCacheLines(line, 128);
    const uint32_t assignment = reinterpret_cast<__gm__ volatile uint32_t*>(line)[block % 32];
    const uint32_t width = assignment & 7;
    const uint32_t query = assignment >> 3;
    const uint32_t row = query / large_window_policy::kQueries;
    const uint32_t first = query % large_window_policy::kQueries;
    switch (width) {
        case 1: FinishDeferredRow<1>(input, row, first); break;
        case 2: FinishDeferredRow<2>(input, row, first); break;
        case 4: FinishDeferredRow<4>(input, row, first); break;
        default: break;
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
    ThresholdCold<false>(bindings, terminal);
#endif
}

extern "C" __global__ __aicore__ void v9_li_large_deferred_cold(GM_ADDR bindings, GM_ADDR terminal) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    ThresholdCold<true>(bindings, terminal);
#endif
}

extern "C" __global__ __aicore__ void v9_li_large_deferred_cold_auto(GM_ADDR bindings, GM_ADDR terminal) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    ThresholdCold<true, true>(bindings, terminal);
#endif
}

extern "C" __global__ __aicore__ void v9_li_large_final_pair(GM_ADDR bindings, GM_ADDR terminal) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV { FinishDeferredWindow<2, 4>(bindings, terminal); }
#endif
}

extern "C" __global__ __aicore__ void v9_li_large_final_q1(GM_ADDR bindings, GM_ADDR terminal) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV { FinishDeferredWindow<1>(bindings, terminal); }
#endif
}

extern "C" __global__ __aicore__ void v9_li_large_final_pair_dense(GM_ADDR bindings, GM_ADDR terminal) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV { FinishDeferredWindow<2>(bindings, terminal); }
#endif
}

extern "C" __global__ __aicore__ void v9_li_large_final_q4(GM_ADDR bindings, GM_ADDR terminal) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV { FinishDeferredWindow<4>(bindings, terminal); }
#endif
}

extern "C" __global__ __aicore__ void v9_li_large_final_auto(GM_ADDR bindings, GM_ADDR terminal) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV { FinishDeferredAuto(bindings, terminal); }
#endif
}
