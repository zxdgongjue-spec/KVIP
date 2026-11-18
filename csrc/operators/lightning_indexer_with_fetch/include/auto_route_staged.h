// SPDX-License-Identifier: Apache-2.0
// Stage immutable routing inputs in UB; prediction and publication contracts
// match Evaluate<true, true>, including request identity and owner generations.
#pragma once
#include "auto_route.h"
namespace kvip_auto_route {
struct FeedbackHead {
    uint64_t generation, step, necessary, queries, scoreWork, tokenBytes, mode, valid;
};
struct RouteTables {
    Config config;
    Request requests[16];
    uint32_t owners[16], ends[64];
    FeedbackHead feedback[64];
};
static_assert(sizeof(RouteTables)<=16384,"Route tables exceed existing init UB");
__aicore__ inline bool ValidStaged(const __ubuf__ Config* c) {
    return c && c->version == kVersion && c->owners && c->owners <= 65536 &&
        c->requests && c->feedback && c->pending &&
        c->lowBudget && c->lowBudget < c->highBudget && c->highBudget <= 65535 &&
        c->maxStepGap && c->maxStepGap <= 1024 && c->learning <= 1 && c->fullTotal && c->fullTotal <= 2048 &&
        c->fullWindow && c->fullWindow <= 480 && c->fullSteps <= 65535 && c->fullCarry <= 1;
}

template<class Context>
__aicore__ inline bool EvaluateStaged(__gm__ Context* ctx, __ubuf__ RouteTables* table, bool& noPrefetch, uint32_t publishRow) {
    constexpr bool Publish=true, RowOnly=true;
    noPrefetch = false;
    auto c = &table->config;
    if (!ctx->admissionControls || !ValidStaged(c) || !ctx->shape.batch || ctx->shape.batch > 16)
        return false;
    if constexpr (RowOnly) {
        static_assert(Publish, "A row-local evaluator must publish its row");
        if (publishRow >= ctx->shape.batch || c->pending % 128 || ctx->admissionControls % 128)
            return false;
    }
    auto requests = table->requests;
    auto feedback = table->feedback;
    auto pending = reinterpret_cast<__gm__ Pending*>(c->pending);
    auto controls = reinterpret_cast<__gm__ int32_t*>(ctx->admissionControls);
    auto owners = ctx->rowOwners ? table->owners : nullptr;
    auto ends = ctx->queryEnds ? table->ends : nullptr;
    uint64_t bytes = 0, work = 0, active = 0, reason = Predicted;
    bool known = true, wasLow = true;
    // The common owner table fits in one word: detect duplicate active owners
    // in one pass instead of rereading B request records for every row. Larger
    // owner tables retain the same exact fallback; no IDs are truncated.
    uint64_t seenOwners = 0, duplicateOwners = 0;
    bool smallOwners = true;
    for (uint32_t b = 0; b < ctx->shape.batch; ++b) {
        if (!requests[b].queries) continue;
        uint64_t owner = requests[b].owner;
        if (owner >= 64) { smallOwners = false; break; }
        uint64_t bit = uint64_t(1) << owner;
        duplicateOwners |= seenOwners & bit;
        seenOwners |= bit;
    }
    for (uint32_t b = 0; b < ctx->shape.batch; ++b) {
        auto r = requests + b;
        const bool writeRow = Publish && (!RowOnly || b == publishRow);
        if (writeRow) {
            auto p = pending + b;
            auto words = reinterpret_cast<__gm__ uint64_t*>(p);
            for (uint32_t i = 0; i < 16; ++i) words[i] = 0;
            p->owner = r->owner; p->generation = r->generation; p->step = r->step;
            p->queries = r->queries; p->tokenBytes = ctx->shape.tokenBytes;
        }
        uint64_t q = 0, rowWork = 0;
        for (uint32_t i = 0; i < 4; ++i) {
            uint64_t end = ends ? ends[b * 4 + i] : (i < r->queries ? ctx->shape.history : 0);
            if (end) ++q;
            rowWork += end;
            if (end > ctx->shape.history) q = 100;
        }
        if (writeRow) pending[b].scoreWork = rowWork;
        if (!r->queries && !q) continue; // Padding cannot train an owner.
        ++active; work += rowWork;
        bool identity = r->owner < c->owners && r->owner == (owners ? owners[b] : b) &&
            r->generation && r->step && r->queries >= 1 && r->queries <= 4 && q == r->queries;
        if (smallOwners) {
            if (r->owner >= 64 || (duplicateOwners & (uint64_t(1) << r->owner))) identity = false;
        } else {
            for (uint32_t j = 0; j < ctx->shape.batch; ++j)
                if (j != b && requests[j].queries && requests[j].owner == r->owner) identity = false;
        }
        uint64_t rowReason = Predicted;
        bool predict = false;
        if (!identity) rowReason = InvalidIdentity;
        else if (!c->learning) rowReason = LearningDisabled;
        else {
            auto f = feedback + r->owner;
            bool valid = f->valid == kValid && f->generation && f->step &&
                f->queries >= 1 && f->queries <= 4 && f->necessary <= 8192 && f->mode <= 1;
            bool learnable = !valid || r->generation > f->generation ||
                (r->generation == f->generation && r->step > f->step);
            if (writeRow) {
                pending[b].previousStep = valid ? f->step : 0;
                pending[b].learnable = learnable;
            }
            if (!learnable) rowReason = StaleStep;
            else if (!valid || r->generation != f->generation) rowReason = Unknown;
            else if (r->step - f->step > c->maxStepGap) rowReason = StaleStep;
            else if (f->tokenBytes != ctx->shape.tokenBytes) rowReason = ShapeChange;
            else {
                uint64_t predicted = ScaleReads(uint32_t(f->necessary), uint32_t(r->queries),
                                                uint32_t(f->queries));
                bytes += predicted * ctx->shape.tokenBytes;
                wasLow = wasLow && f->mode == 0;
                predict = true;
            }
        }
        if (!predict) {
            if constexpr (!Publish && KVIP_AUTO_ROUTE_FAST_PROBE) {
                // One untrusted active row forces the whole graph high. This
                // read-only init block has no pending records left to publish.
                return true;
            }
            known = false;
            if (reason == Predicted) reason = rowReason;
        }
        if constexpr (!Publish && KVIP_AUTO_ROUTE_FAST_PROBE) {
            // Remaining rows can only increase bytes. Their total score work
            // cannot exceed B * Q4 * history. Once wasLow is false it stays
            // false, so the entry budget is already an upper bound as well.
            // B<=16, Q<=4, uint32 history/tokenBytes, necessary<=8192 and
            // budgets<=65535 keep both products below 2^62 (no division).
            const uint64_t maximumWork = uint64_t(ctx->shape.batch) * 4 * ctx->shape.history;
            const uint64_t maximumBudget = wasLow ? c->highBudget : c->lowBudget;
            if (bytes * 1024 > maximumBudget * maximumWork) return true;
        }
    }
    uint64_t budget = wasLow ? c->highBudget : c->lowBudget;
    uint64_t mode = known && active && work && bytes * 1024 <= budget * work ? 0 : 1;
    noPrefetch = mode == 0;
    if (!active) reason = Unknown;
    if constexpr (Publish) {
        if constexpr (RowOnly) {
            pending[publishRow].mode = mode; pending[publishRow].reason = reason;
            // Eight 16-byte control rows share one cache line. Only its first
            // row's block writes the line; Pending records have distinct lines.
            if (publishRow % 8 == 0) {
                const uint32_t end = publishRow + 8 < ctx->shape.batch ? publishRow + 8 : ctx->shape.batch;
                for (uint32_t b = publishRow; b < end; ++b) {
                    controls[b * 4] = mode ? int32_t(c->fullTotal) : 0;
                    controls[b * 4 + 1] = int32_t(c->fullWindow);
                    controls[b * 4 + 2] = int32_t(c->fullSteps);
                    controls[b * 4 + 3] = int32_t(c->fullCarry);
                }
            }
        } else {
            for (uint32_t b = 0; b < ctx->shape.batch; ++b) {
                pending[b].mode = mode; pending[b].reason = reason;
                controls[b * 4] = mode ? int32_t(c->fullTotal) : 0;
                controls[b * 4 + 1] = int32_t(c->fullWindow);
                controls[b * 4 + 2] = int32_t(c->fullSteps);
                controls[b * 4 + 3] = int32_t(c->fullCarry);
            }
        }
    }
    return true;
}

template<class Context>
__aicore__ inline bool CanStage(const __gm__ Context* ctx,uint32_t row) {
    auto c=reinterpret_cast<const __gm__ Config*>(ctx->autoRoute);
    return ctx->admissionControls && Valid(c) && c->owners<=64 &&
        ctx->shape.batch && ctx->shape.batch<=16 && row<ctx->shape.batch &&
        !(c->pending%128) && !(ctx->admissionControls%128);
}
#if defined(__CCE_AICORE__)
__aicore__ inline void CopyRouteTable(uint64_t source, uint32_t bytes,
                                     AscendC::LocalTensor<uint32_t> target) {
    AscendC::GlobalTensor<uint32_t> input;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(source));
    AscendC::DataCopyExtParams copy{1,bytes,0,0,0};
    AscendC::DataCopyPadExtParams<uint32_t> pad{false,0,0,0};
    AscendC::DataCopyPad(target,input,copy,pad);
}
// A warp proves the common all-Q4 case using only the staged table. At
// most sixteen lanes participate. An inconclusive proof selects the general
// evaluator, before any pending or control record has been changed.
__simt_vf__ __aicore__ LAUNCH_BOUND(32) inline void ScoreQ4Route(
    __ubuf__ RouteTables* table, uint32_t batch, uint32_t history,
    uint32_t tokenBytes, uint32_t ownRow, __gm__ int32_t* controls, __ubuf__ uint32_t* proof) {
    const uint32_t lane = threadIdx.x;
    const bool active = lane < batch;
    auto request = &table->requests[active ? lane : 0];
    bool valid = request->owner < table->config.owners && request->owner == table->owners[active ? lane : 0] &&
                 request->queries == 4 && request->generation;
    const uint32_t owner = valid ? uint32_t(request->owner) : 0;
    auto previous = &table->feedback[owner];
    auto ends = table->ends + 4 * (active ? lane : 0);
    valid = valid && previous->valid == kValid && previous->generation == request->generation && previous->step &&
            request->step > previous->step && request->step - previous->step <= table->config.maxStepGap &&
            previous->queries == 4 && previous->necessary <= 8192 && previous->mode <= 1 && previous->tokenBytes == tokenBytes &&
            ends[0] - 1u < history && ends[1] - 1u < history && ends[2] - 1u < history && ends[3] - 1u < history;
    uint32_t lowerOwners = active && owner < 32 ? (1U << (owner & 31)) : 0;
    uint32_t upperOwners = active && owner >= 32 ? (1U << (owner & 31)) : 0;
    uint32_t work = active ? ends[0] + ends[1] + ends[2] + ends[3] : 0;
    uint32_t demand = active ? uint32_t(previous->necessary) : 0;
    const uint32_t invalidRows = asc_ballot(active && !valid);
    const uint32_t previousHighRows = asc_ballot(active && previous->mode != 0);
    const uint32_t ownWork = asc_shfl(work, ownRow, 32);
    for (uint32_t distance = 1; distance < 16; distance *= 2) {
        const uint32_t src = lane ^ distance;
        lowerOwners |= asc_shfl(lowerOwners, src, 32);
        upperOwners |= asc_shfl(upperOwners, src, 32);
        work += asc_shfl(work, src, 32);
        demand += asc_shfl(demand, src, 32);
    }
    const bool unique = uint32_t(__builtin_popcount(lowerOwners) + __builtin_popcount(upperOwners)) == batch;
    const uint64_t budget = previousHighRows ? table->config.lowBudget : table->config.highBudget;
    // Lanes 0..15 participated in the reduction. Broadcast its verified
    // result before touching any global state, including on fallback.
    const uint32_t mode = asc_shfl((invalidRows || !unique) ? 2U :
        uint32_t(uint64_t(demand) * tokenBytes * 1024 > budget * work), 0, 32);
    if (!lane) proof[0] = mode;
    if (mode > 1) return;

    // Lanes own disjoint words of one aligned Pending record. Publish all
    // 16 words even when a previous replay took the general fallback. Both
    // completion paths acquire this record before committing feedback.
    if (lane < 16) {
        auto own = &table->requests[ownRow];
        uint64_t value = 0;
        switch (lane) {
        case offsetof(Pending, owner) / sizeof(value): value = own->owner; break;
        case offsetof(Pending, generation) / sizeof(value): value = own->generation; break;
        case offsetof(Pending, step) / sizeof(value): value = own->step; break;
        case offsetof(Pending, learnable) / sizeof(value): value = 1; break;
        case offsetof(Pending, mode) / sizeof(value): value = mode; break;
        case offsetof(Pending, queries) / sizeof(value): value = 4; break;
        case offsetof(Pending, scoreWork) / sizeof(value): value = ownWork; break;
        case offsetof(Pending, tokenBytes) / sizeof(value): value = tokenBytes; break;
        case offsetof(Pending, previousStep) / sizeof(value): value = table->feedback[own->owner].step; break;
        }
        static_assert(sizeof(Pending) == 16 * sizeof(uint64_t) && Predicted == 0,
                      "Q4 pending publication requires the routing ABI");
        auto pending = reinterpret_cast<__gm__ uint64_t*>(table->config.pending);
        asc_stcg(pending + ownRow * 16 + lane, value);
    }
    if (ownRow % 8 == 0 && ownRow + lane / 4 < batch) {
        const uint32_t field = lane % 4;
        const uint32_t value = field == 0 ? uint32_t(mode ? table->config.fullTotal : 0) :
                               field == 1 ? uint32_t(table->config.fullWindow) :
                               field == 2 ? uint32_t(table->config.fullSteps) : uint32_t(table->config.fullCarry);
        asc_stcg(controls + ownRow * 4 + lane, int32_t(value));
    }
    asc_threadfence();
}

template<class Context>
__aicore__ inline bool StageAndEvaluate(__gm__ Context* ctx, uint32_t row,
    AscendC::LocalTensor<uint32_t> storage,bool& noPrefetch) {
    auto c=reinterpret_cast<const __gm__ Config*>(ctx->autoRoute);
    // The caller admitted this immutable configuration with CanStage before
    // selecting this implementation. Request and feedback data still need
    // to be checked for each replay.
    auto table=reinterpret_cast<__ubuf__ RouteTables*>(storage.GetPhyAddr());
    CopyRouteTable(ctx->autoRoute,sizeof(Config),storage);
    CopyRouteTable(c->requests,ctx->shape.batch*sizeof(Request),storage[offsetof(RouteTables,requests)/4]);
    if(ctx->rowOwners)CopyRouteTable(ctx->rowOwners,ctx->shape.batch*4,storage[offsetof(RouteTables,owners)/4]);
    if(ctx->queryEnds)CopyRouteTable(ctx->queryEnds,ctx->shape.batch*16,storage[offsetof(RouteTables,ends)/4]);
    AscendC::GlobalTensor<uint32_t> input;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(c->feedback));
    AscendC::DataCopyExtParams copy{uint16_t(c->owners),64,64,0,0};
    AscendC::DataCopyPadExtParams<uint32_t> pad{false,0,0,0};
    AscendC::DataCopyPad(storage[offsetof(RouteTables,feedback)/4],input,copy,pad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    constexpr uint32_t resultWord = (sizeof(RouteTables) + 31) / 32 * 8;
    static_assert((resultWord + 8) * 4 <= 16384, "route result must fit init scratch");
    // Sixty-four query ends fit in a 32-bit warp sum under this bound.
    if (ctx->rowOwners && ctx->queryEnds && ctx->shape.history <= UINT32_MAX / 64 &&
        table->config.learning) {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        Simt::VF_CALL<ScoreQ4Route>(Simt::Dim3{32,1,1}, table, ctx->shape.batch,
            ctx->shape.history, ctx->shape.tokenBytes, row, reinterpret_cast<__gm__ int32_t*>(ctx->admissionControls),
            reinterpret_cast<__ubuf__ uint32_t*>(storage.GetPhyAddr()) + resultWord);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        uint32_t mode = storage.GetValue(resultWord);
        if (mode <= 1) {
            noPrefetch = !mode;
            return true;
        }
    }
    const bool result = EvaluateStaged(ctx, table, noPrefetch, row);
    if(result){
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(c->pending)+row*128,128);
        if(row%8==0){uint32_t count=ctx->shape.batch-row;if(count>8)count=8;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->admissionControls)+row*16,count*16);}
    }
    return result;
}
#endif
} // namespace kvip_auto_route
