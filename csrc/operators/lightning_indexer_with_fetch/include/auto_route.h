// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

#if defined(__CCE_AICORE__)
#define KVIP_ROUTE_FN __aicore__
#define KVIP_ROUTE_GM __gm__
#else
#define KVIP_ROUTE_FN
#define KVIP_ROUTE_GM
#endif

namespace kvip_auto_route {
constexpr uint64_t kVersion = 1, kValid = 0x4b565250;
enum Reason : uint64_t {
    Predicted = 0, Unknown = 1, LearningDisabled = 2,
    InvalidIdentity = 3, StaleStep = 4, ShapeChange = 5,
};
struct Config {
    uint64_t version, owners, requests, feedback, pending;
    uint64_t lowBudget, highBudget, maxStepGap, learning;
    uint64_t fullTotal, fullWindow, fullSteps, fullCarry;
    uint64_t reserved[3];
};
struct Request { uint64_t owner, generation, step, queries; };
struct Feedback {
    uint64_t generation, step, necessary, queries, scoreWork, tokenBytes;
    uint64_t mode, valid, reserved[8];
};
struct Pending {
    uint64_t owner, generation, step, learnable, mode, reason;
    uint64_t queries, scoreWork, tokenBytes, necessary, usefulPrefetch, previousStep;
    uint64_t reserved[4];
};
static_assert(sizeof(Config) == 128 && sizeof(Request) == 32 &&
              sizeof(Feedback) == 128 && sizeof(Pending) == 128, "Auto route ABI");

KVIP_ROUTE_FN inline bool Valid(const KVIP_ROUTE_GM Config* c) {
    return c && c->version == kVersion && c->owners && c->owners <= 65536 &&
        c->requests && c->feedback && c->pending &&
        c->lowBudget && c->lowBudget < c->highBudget && c->highBudget <= 65535 &&
        c->maxStepGap && c->maxStepGap <= 1024 && c->learning <= 1 && c->fullTotal && c->fullTotal <= 2048 &&
        c->fullWindow && c->fullWindow <= 480 && c->fullSteps <= 65535 && c->fullCarry <= 1;
}

// Called only after validating previousQueries/currentQueries in 1..4 and
// necessary <= 8192. Constant small divisors avoid a device-side uint64 divide.
KVIP_ROUTE_FN inline uint32_t ScaleReads(uint32_t necessary, uint32_t currentQueries,
                                       uint32_t previousQueries) {
    uint32_t numerator = necessary * currentQueries + previousQueries - 1;
    if (previousQueries == 1) return numerator;
    if (previousQueries == 2) return numerator >> 1;
    if (previousQueries == 3) return numerator / 3;
    return numerator >> 2;
}

// Every init block may evaluate immutable requests and previous successful
// feedback. Only block 0 publishes pending/control state; evaluating without
// publication must never read that concurrently written state.
template<bool Publish, class Context>
KVIP_ROUTE_FN inline bool Evaluate(KVIP_ROUTE_GM Context* ctx, bool& noPrefetch) {
    noPrefetch = false;
    auto c = reinterpret_cast<const KVIP_ROUTE_GM Config*>(ctx->autoRoute);
    if (!ctx->admissionControls || !Valid(c) || !ctx->shape.batch || ctx->shape.batch > 16)
        return false;
    auto requests = reinterpret_cast<const KVIP_ROUTE_GM Request*>(c->requests);
    auto feedback = reinterpret_cast<const KVIP_ROUTE_GM Feedback*>(c->feedback);
    auto pending = reinterpret_cast<KVIP_ROUTE_GM Pending*>(c->pending);
    auto controls = reinterpret_cast<KVIP_ROUTE_GM int32_t*>(ctx->admissionControls);
    auto owners = reinterpret_cast<const KVIP_ROUTE_GM uint32_t*>(ctx->rowOwners);
    auto ends = reinterpret_cast<const KVIP_ROUTE_GM uint32_t*>(ctx->queryEnds);
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
        if constexpr (Publish) {
            auto p = pending + b;
            auto words = reinterpret_cast<KVIP_ROUTE_GM uint64_t*>(p);
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
        if constexpr (Publish) pending[b].scoreWork = rowWork;
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
            if constexpr (Publish) {
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
            known = false;
            if (reason == Predicted) reason = rowReason;
        }
    }
    uint64_t budget = wasLow ? c->highBudget : c->lowBudget;
    uint64_t mode = known && active && work && bytes * 1024 <= budget * work ? 0 : 1;
    noPrefetch = mode == 0;
    if (!active) reason = Unknown;
    if constexpr (Publish) {
        for (uint32_t b = 0; b < ctx->shape.batch; ++b) {
            pending[b].mode = mode; pending[b].reason = reason;
            controls[b * 4] = mode ? int32_t(c->fullTotal) : 0;
            controls[b * 4 + 1] = int32_t(c->fullWindow);
            controls[b * 4 + 2] = int32_t(c->fullSteps);
            controls[b * 4 + 3] = int32_t(c->fullCarry);
        }
    }
    return true;
}

template<class Context>
KVIP_ROUTE_FN inline bool Prepare(KVIP_ROUTE_GM Context* ctx) {
    bool noPrefetch;
    return Evaluate<true>(ctx, noPrefetch);
}

// Called only after this request's Get completion and Index update have joined.
// Duplicate/backward submissions cannot replace the last successful sample.
KVIP_ROUTE_FN inline bool Commit(const KVIP_ROUTE_GM Config* c, uint32_t b, uint64_t error) {
    if (error || !Valid(c) || !c->learning) return false;
    auto p = reinterpret_cast<const KVIP_ROUTE_GM Pending*>(c->pending) + b;
    if (!p->learnable || p->owner >= c->owners || !p->generation || !p->step ||
        !p->queries || p->queries > 4 || !p->scoreWork || p->necessary > 8192) return false;
    auto f = reinterpret_cast<KVIP_ROUTE_GM Feedback*>(c->feedback) + p->owner;
    if (f->valid == kValid && (p->generation < f->generation ||
        (p->generation == f->generation && p->step <= f->step))) return false;
    f->valid = 0;
    f->generation = p->generation; f->step = p->step; f->necessary = p->necessary;
    f->queries = p->queries; f->scoreWork = p->scoreWork; f->tokenBytes = p->tokenBytes;
    f->mode = p->mode;
    for (uint32_t i = 0; i < 8; ++i) f->reserved[i] = 0;
    f->valid = kValid;
    return true;
}
}  // namespace kvip_auto_route
#undef KVIP_ROUTE_FN
#undef KVIP_ROUTE_GM
