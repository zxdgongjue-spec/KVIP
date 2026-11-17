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
template<class Context>
__aicore__ inline bool StageAndEvaluate(__gm__ Context* ctx, uint32_t row,
    AscendC::LocalTensor<uint32_t> storage,bool& noPrefetch) {
    auto c=reinterpret_cast<const __gm__ Config*>(ctx->autoRoute);
    // Original scalar path remains available for unsupported metadata layouts.
    if(!ctx->admissionControls || !Valid(c) || c->owners>64 || !ctx->shape.batch ||
       ctx->shape.batch>16 || row>=ctx->shape.batch || c->pending%128 ||
       ctx->admissionControls%128)return false;
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
    bool result=EvaluateStaged(ctx,table,noPrefetch,row);
    if(result){
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(c->pending)+row*128,128);
        if(row%8==0){uint32_t count=ctx->shape.batch-row;if(count>8)count=8;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->admissionControls)+row*16,count*16);}
    }
    return result;
}
#endif
} // namespace kvip_auto_route
