#pragma once
// Included by lifecycle.cpp after the existing CQ and Fetch helpers. One AIV
// owns each packed request; no same-kernel cross-request completion wait.
#include "fusion3_plan.h"

namespace fusion3 {
using namespace AscendC;
using namespace rotating;
static_assert(kControlWord * 4 == kFinalControlOffset * 4, "Resolver data layout changed");
static_assert(kArenaBytes <= V9_USER_UB_BYTES, "Tail arena exceeds reserved user UB");
static_assert(kMoveScratchBytes < kControlWord * 4, "Small Apply overlaps retained control");
static_assert(kExtendedRecords * 8 < kControlWord * 4, "Large Apply overlaps retained control");
static_assert(kColdStatsWord + 16 <= kStateWord && kStateWord + 128 <= kControlWords,
              "Tail state overlaps resolver statistics or control boundary");

__aicore__ inline void JoinVector() {
    SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
}

__aicore__ inline bool Active(__gm__ Context* ctx, uint32_t b) {
    if (!ctx->queryEnds) return true;
    auto ends = reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds) + b * 4;
    return ends[0] || ends[1] || ends[2] || ends[3];
}

// The inputs are immutable during the launch. Every block performs this small
// check so invalid global unpack/ownership cannot race another block's Apply.
// Padding owners may repeat; only active Index owners must be exclusive.
__aicore__ inline uint32_t Preflight(__gm__ Context* ctx, __gm__ uint64_t* a,
                                    const RuntimeShape& s) {
    if (!s.batch || s.batch > 32 ||
        (s.finalWorkers != 1 && s.finalWorkers != 2 && s.finalWorkers != 4) ||
        a[24] != s.history || a[25] != s.resident || a[14] != s.finalWorkers) return kLogicInvalid;
    auto owners = reinterpret_cast<__gm__ uint32_t*>(a[9]);
    for (uint32_t b = 0; b < s.batch; ++b) {
        if (!Active(ctx, b)) continue;
        uint32_t owner = owners[b];
        uint32_t contextOwner = ctx->rowOwners ? reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b] : b;
        if (contextOwner != owner)
            return kLogicInvalid;
        for (uint32_t prior = 0; prior < b; ++prior)
            if (Active(ctx, prior) && owners[prior] == owner) return kLogicInvalid;
    }
    auto unpack = reinterpret_cast<__gm__ int64_t*>(a[20]);
    auto valid = reinterpret_cast<__gm__ int64_t*>(a[21]);
    for (uint32_t row = 0; row < s.batch * 4; ++row) {
        if (!valid[row]) continue; // Invalid entries may contain poison indices.
        int64_t source = unpack[row];
        if (source < 0 || uint64_t(source) >= uint64_t(s.batch) * 4 ||
            !Active(ctx, uint32_t(source) / 4)) return kLogicInvalid;
    }
    return 0;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void InvalidateContextOwner(
    __gm__ Context* ctx, uint32_t b, uint32_t resident) {
    if (!ctx->hotValid) return;
    uint32_t owner = ctx->rowOwners ? asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners) + b) : b;
    auto valid = reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid) + uint64_t(owner) * resident;
    for (uint32_t i = threadIdx.x; i < resident; i += kThreads) asc_stcg(valid + i, uint32_t(0));
    asc_threadfence();
}

__aicore__ inline uint32_t ResolveOwned(__gm__ Context* ctx, const RuntimeShape& s,
    uint32_t b, LocalTensor<uint32_t> local, uint32_t initialError, bool active) {
    auto row = reinterpret_cast<__gm__ Row*>(ctx->rows) + b;
    auto info = reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus) + b * kRowStatusWords;
    auto trace = reinterpret_cast<__gm__ uint64_t*>(ctx->owners) + b * 16;
    auto selected = local, loc = local[kSlots], keys = local[2 * kSlots], ranks = local[3 * kSlots];
    auto ids = local[4 * kSlots].ReinterpretCast<uint16_t>();
    auto slots = local[kFinalSlotsOffset], positions = local[kFinalPositionsOffset], ctl = local[kControlWord];
    uint64_t started = GetSystemCycle();
    uint32_t error = initialError, cold = 0, count = 0, safe = 0;
    if (active && !error) {
        GlobalTensor<uint32_t> input;
        input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->selected) + b * kSlots);
        DataCopy(selected, input, kSlots);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        if (s.resident <= 2 * kSlots) {
            GlobalTensor<uint32_t> metadata;
            uint32_t owner = ctx->rowOwners ? reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b] : b;
            if (ctx->hotValid) {
                metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid) + uint64_t(owner) * s.resident);
                DataCopy(keys, metadata, s.resident);
            }
            if (ctx->reservedHot) {
                metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->reservedHot) + uint64_t(b) * s.resident);
                DataCopy(local[4 * kSlots], metadata, s.resident);
            }
            uint32_t begin = 0, end = 0;
            if (ctx->tailBounds) {
                auto bounds = reinterpret_cast<__gm__ uint32_t*>(ctx->tailBounds) + b * 2;
                begin = bounds[0]; end = bounds[1];
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Simt::VF_CALL<AdaptColdPredicateLocal>(Simt::Dim3{kThreads,1,1}, s.resident,
                FinalPtr(selected), FinalPtr(loc), FinalPtr(keys),
                reinterpret_cast<__ubuf__ int32_t*>(local[4 * kSlots].GetPhyAddr()),
                ctx->hotValid != 0, ctx->reservedHot != 0, begin, end);
            PipeBarrier<PIPE_V>();
            ColdPositions(s.history, s.resident, FinalPtr(loc), FinalPtr(loc), FinalPtr(positions),
                          FinalPtr(ctl[kColdStatsWord]));
        } else {
            Simt::VF_CALL<AdaptColdPredicate>(Simt::Dim3{kThreads,1,1}, s.resident, ctx, b,
                FinalPtr(selected), FinalPtr(keys));
            PipeBarrier<PIPE_V>();
            ColdPositions(s.history, s.resident, FinalPtr(keys), FinalPtr(loc), FinalPtr(positions),
                          FinalPtr(ctl[kColdStatsWord]));
        }
        JoinVector();
        cold = ctl.GetValue(kColdStatsWord);
        error = ctl.GetValue(kColdStatsWord + 8) | (row->slots > kSlots ? kLogicInvalid : 0U);
        // Transport error remains in Row. Do not feed logic errors into it.
        if (row->error) error |= kTransportError;
        trace[0] = started; trace[1] = GetSystemCycle();
        if (!error) {
            Simt::VF_CALL<ReplicatedResolve<kThreads>>(Simt::Dim3{kThreads,1,1},
                s.history, s.resident, 1, s.tokenBytes, FinalPtr(selected), FinalPtr(loc),
                FinalPtr(positions), cold, FinalPtr(keys), FinalPtr(ranks),
                reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()), FinalPtr(slots), FinalPtr(ctl),
                reinterpret_cast<__gm__ uint32_t*>(ctx->mapping) + uint64_t(b) * s.history,
                reinterpret_cast<__gm__ uint64_t*>(row->addresses),
                reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks),
                reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets),
                reinterpret_cast<__gm__ int32_t*>(ctx->destinations),
                reinterpret_cast<__gm__ uint32_t*>(ctx->count), 0, row->slots, b, ctx);
            JoinVector();
            error |= ctl.GetValue(1);
            if (!error) { count = ctl.GetValue(0); safe = ctl.GetValue(5); }
        }
    }
    if (!active || initialError) {
        Duplicate(selected, UINT32_MAX, kSlots);
        Duplicate(loc, UINT32_MAX, kSlots);
    }
    GlobalTensor<uint32_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations) + b * kSlots);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(output, loc, kSlots);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0); WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    for (uint32_t w = 0; w < s.finalWorkers; ++w)
        PrepareFinalFetch(s.finalWorkers, s.finalPostOffset, ctx, b, w, w ? 0 : count,
                          b * kSlots, w ? 0 : error, w ? 0 : safe);
    info[0] = error; info[1] = count; info[2] = b * kSlots; info[3] = 0; info[4] = 0;
    row->finalBegin = started; row->finalRequests = count; row->finalHead = 0;
    row->finalPending = count ? 1 : 0; row->finalEnd = GetSystemCycle();
    trace[0] = started; trace[2] = GetSystemCycle(); trace[3] = row->finalEnd;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(trace), 64);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(info), 32);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row) + 960, 64);
    return error;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void GatherLogical(
    __gm__ int64_t* logical, __ubuf__ uint32_t* tokens, __ubuf__ int32_t* result,
    uint32_t history) {
    for (uint32_t k = threadIdx.x; k < kTop; k += kThreads) {
        uint32_t token = tokens[k];
        result[k] = token < history ? int32_t(asc_ldcg(logical + token)) : -1;
    }
}

// Source-owner scatter: a request reads only its own old logical IDs. Fanout
// and padding are supported without a cross-request Publish->Apply barrier.
__aicore__ inline void PublishOwned(__gm__ uint64_t* a, const RuntimeShape& s,
    uint32_t b, LocalTensor<uint32_t> local, bool poisonAll, bool poisonSource) {
    auto pipe = GetTPipePtr();
    TEventID ready = pipe->AllocEventID<HardEvent::V_MTE3>();
    TEventID done[2] = {pipe->AllocEventID<HardEvent::MTE3_S>(), pipe->AllocEventID<HardEvent::MTE3_S>()};
    bool pending[2] = {false, false};
    auto unpack = reinterpret_cast<__gm__ int64_t*>(a[20]);
    auto valid = reinterpret_cast<__gm__ int64_t*>(a[21]);
    auto owners = reinterpret_cast<__gm__ uint32_t*>(a[9]);
    auto logical = reinterpret_cast<__gm__ int64_t*>(a[3]);
    GlobalTensor<int32_t> logicalOut, slotsOut;
    logicalOut.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a[22]));
    slotsOut.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a[23]));
    for (uint32_t q = 0; q < 4 && !poisonAll; ++q) {
        uint32_t bank = q % 2;
        if (pending[bank]) { WaitFlag<HardEvent::MTE3_S>(done[bank]); pending[bank] = false; }
        auto ids = local[kLogicalWord + bank * kTop].ReinterpretCast<int32_t>();
        bool generated = false;
        for (uint32_t dst = 0; dst < s.batch * 4; ++dst) {
            if (!valid[dst] || uint64_t(unpack[dst]) != uint64_t(b) * 4 + q) continue;
            if (!generated) {
                if (poisonSource) Duplicate(ids, int32_t(-1), kTop);
                else Simt::VF_CALL<GatherLogical>(Simt::Dim3{kThreads,1,1},
                    logical + uint64_t(owners[b]) * s.history, FinalPtr(local[q * kTop]),
                    reinterpret_cast<__ubuf__ int32_t*>(ids.GetPhyAddr()), s.history);
                SetFlag<HardEvent::V_MTE3>(ready); WaitFlag<HardEvent::V_MTE3>(ready);
                generated = true;
            }
            DataCopy(logicalOut[uint64_t(dst) * kTop], ids, kTop);
            DataCopy(slotsOut[uint64_t(dst) * kTop],
                poisonSource ? ids : local[kSlots + q * kTop].ReinterpretCast<int32_t>(), kTop);
        }
        if (generated) { SetFlag<HardEvent::MTE3_S>(done[bank]); pending[bank] = true; }
    }
    for (uint32_t bank = 0; bank < 2; ++bank)
        if (pending[bank]) { WaitFlag<HardEvent::MTE3_S>(done[bank]); pending[bank] = false; }
    // Padding has a destination owner, independent of the source-owner fanout.
    // Drain source copies before reusing bank0 for the all-minus-one row.
    auto padding = local[kLogicalWord].ReinterpretCast<int32_t>();
    bool filled = false;
    for (uint32_t dst = b * 4; dst < b * 4 + 4; ++dst) {
        if (!poisonAll && valid[dst]) continue;
        if (!filled) {
            Duplicate(padding, int32_t(-1), kTop);
            SetFlag<HardEvent::V_MTE3>(ready); WaitFlag<HardEvent::V_MTE3>(ready);
            filled = true;
        }
        DataCopy(logicalOut[uint64_t(dst) * kTop], padding, kTop);
        DataCopy(slotsOut[uint64_t(dst) * kTop], padding, kTop);
    }
    if (filled) { SetFlag<HardEvent::MTE3_S>(done[0]); pending[0] = true; }
    for (uint32_t bank = 0; bank < 2; ++bank) {
        if (pending[bank]) WaitFlag<HardEvent::MTE3_S>(done[bank]);
        pipe->ReleaseEventID<HardEvent::MTE3_S>(done[bank]);
    }
    pipe->ReleaseEventID<HardEvent::V_MTE3>(ready);
}

__aicore__ inline uint32_t PlanAndApply(__gm__ uint64_t* a, uint32_t b,
    uint32_t owner, LocalTensor<uint32_t> local, bool active, uint32_t logicError) {
    auto status = reinterpret_cast<__gm__ uint32_t*>(a[6]) + b;
    auto plan = reinterpret_cast<__gm__ uint32_t*>(a[17]) + uint64_t(b) * kExtendedTransferWords;
    if (!active || logicError) {
        Simt::VF_CALL<StoreUpdateResult>(Simt::Dim3{kThreads,1,1}, a, b, logicError, true);
        JoinVector();
        return logicError;
    }
    uint32_t producerCount = reinterpret_cast<__gm__ uint32_t*>(a[26])[b];
    bool extended = producerCount > kPlanCandidates;
    if (extended) Simt::VF_CALL<MakePlan<kThreads, true>>(Simt::Dim3{kThreads,1,1}, a, b, FinalPtr(local));
    else Simt::VF_CALL<MakePlan<kThreads, false>>(Simt::Dim3{kThreads,1,1}, a, b, FinalPtr(local));
    JoinVector();
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status), 4);
    logicError = *status;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(plan), 32);
    uint32_t count = extended ? plan[0] : local.GetValue(0);
    if (count > (extended ? kExtendedRecords : kPlanRecords)) logicError |= kPlanError;
    if (!logicError && count) {
        // Planner capacity follows producer high-water mark; mover capacity
        // follows the actual compact count, as in the standalone baseline.
        // Once count/error have been saved, the large hashes/status are dead.
        bool largeMove = count > kPlanRecords;
        if (extended && !largeMove) {
            GlobalTensor<uint32_t> compact;
            compact.SetGlobalBuffer(plan);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            DataCopy(local, compact, kTransferHeaderWords + ((2 * count + 7) / 8) * 8);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
        for (uint32_t group = 0; group < kCopyWorkers; ++group) {
            if (largeMove) Simt::VF_CALL<MoveExtended<kThreads>>(Simt::Dim3{kThreads,1,1},
                a, owner, group, true, plan, reinterpret_cast<__ubuf__ uint64_t*>(local.GetPhyAddr()), count);
            else Simt::VF_CALL<Move<kThreads>>(Simt::Dim3{kThreads,1,1},
                a, owner, group, true, FinalPtr(local),
                reinterpret_cast<__ubuf__ uint64_t*>(local[kTransferPlanWords].GetPhyAddr()), count);
            // The next group reuses the same snapshot; VF return alone is not
            // used as a Scalar/MTE completion fence with auto-sync disabled.
            JoinVector();
        }
    }
    Simt::VF_CALL<StoreUpdateResult>(Simt::Dim3{kThreads,1,1}, a, b, logicError, false);
    JoinVector();
    return logicError;
}

__aicore__ inline void FinishOwned(GM_ADDR bindings, GM_ADDR update, uint32_t b,
    uint32_t logicError, bool active) {
    auto ctx = GetContext(bindings);
    auto s = LoadRuntimeShape(ctx);
    auto row = reinterpret_cast<__gm__ Row*>(ctx->rows) + b;
    auto info = reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus) + b * kRowStatusWords;
    auto a = reinterpret_cast<__gm__ uint64_t*>(update);
    uint64_t transportError = row->error;
    uint32_t total = 0, successfulCalls = 0;
    uint64_t lastHead = 0, lastCq = 0;
    for (uint32_t phase = 0; phase < 2; ++phase) {
        if (phase) { CompletePrefetch(ctx, b); transportError |= row->error; }
        for (uint32_t w = 0; w < s.finalWorkers; ++w) {
            auto post = FinalPostAt(s.finalPostOffset, ctx, b * s.finalWorkers + w);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post), sizeof(*post));
            bool range = RangeValid(post->descriptor, post->count, b * kSlots, kSlots, post->nonconflicting);
            if (!range || post->error) logicError |= kLogicInvalid;
            if (!phase && range) total += post->count;
            uint32_t begin = range ? post->descriptor + (phase ? post->nonconflicting : 0) : 0;
            uint32_t count = !range ? 0 : (phase ? post->count - post->nonconflicting : post->nonconflicting);
            auto args = GetArgs(ctx, FinalQueue(s.batch, s.finalWorkers, b, w));
            auto queue = reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args.queue_state);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue), sizeof(*queue));
            transportError |= queue->last_error;
            if (queue->poisoned) transportError |= kvaf_device::kInvalidArgs;
            if (CanSubmit(logicError, transportError, count, phase != 0, row->pending)) {
                args.indices = ctx->sourceBlocks; args.offsets = ctx->sourceOffsets;
                args.output_indices = ctx->destinations; args.miss_count = ctx->count;
                args.request_capacity = s.batch * kSlots;
                if (ctx->fallbackAddresses) {
                    args.address_table = ctx->fallbackAddresses; args.valid_table = ctx->fallbackValid;
                    args.table_capacity = s.batch * kSlots;
                }
                auto done = reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(args.core_status);
                // Native may return before WriteStatus. A previous successful
                // result with the same count must never pass this attempt.
                done->magic = 0;
                kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(done), sizeof(*done));
#if defined(KVIP_FUSION3_TEST_TRANSPORT) && KVIP_FUSION3_TEST_TRANSPORT
                fusion3_test::DeviceGet(args, begin, count);
#else
                kvaf_device::KvafBatchGetDevice(args, 0, begin, count);
#endif
                JoinVector();
                kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(done), sizeof(*done));
                bool current = done->magic == KVAF_AIV_BATCH_GET_STATUS_MAGIC && done->processed_request_count == count;
                transportError |= current ? done->result : kvaf_device::kInvalidArgs;
                if (!current) { queue->poisoned = 1; queue->last_error = kvaf_device::kInvalidArgs; }
                if (current && !done->result) {
                    ++successfulCalls;
                    if (phase || post->nonconflicting == post->count) post->deferred = 0;
                }
            }
            if (post->count) { lastHead = queue->sq_producer_index; lastCq = queue->cq_consumer_index; }
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue), sizeof(*queue));
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post), sizeof(*post));
        }
    }
    bool deferred = false;
    for (uint32_t w = 0; w < s.finalWorkers; ++w)
        deferred |= FinalPostAt(s.finalPostOffset, ctx, b * s.finalWorkers + w)->deferred != 0;
    if (total != info[1]) logicError |= kLogicInvalid;
    if (row->pending) transportError |= kvaf_device::kInvalidArgs;
    bool ready = ConsumerReady(logicError, transportError, row->pending, total, info[1], deferred);
    uint64_t error = logicError ? kvaf_device::kInvalidArgs : 0;
    error |= transportError;
    if (!ready && !error) error = kvaf_device::kInvalidArgs;
    if (ready && active) {
        Simt::VF_CALL<PublishDestinationResidency>(Simt::Dim3{256,1,1}, s.resident, ctx, b, total);
        JoinVector();
    }
    Simt::VF_CALL<StoreUpdateResult>(Simt::Dim3{kThreads,1,1}, a, b,
        logicError | (transportError ? kTransportError : 0U), false);
    JoinVector();
    if (!ready && active) {
        // Preflight may have rejected the update owner/stride. Cleanup must
        // use the same trusted Context owner that LI/Resolve used.
        Simt::VF_CALL<InvalidateContextOwner>(Simt::Dim3{kThreads,1,1}, ctx, b, s.resident);
        JoinVector();
    }
    if (!ready) {
        Simt::VF_CALL<PoisonFailedOutputs>(Simt::Dim3{kThreads,1,1}, a, b, s.batch, false);
        JoinVector();
    }
    auto status = reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(ctx->finalStatus) + b * 2;
    kvaf_device::WriteStatus(status, error, total, successfulCalls, lastHead, lastCq, UINT64_MAX, 0);
    kvaf_device::WriteStatus(status + 1, 0, 0, 0, 0, 0, UINT64_MAX, 0);
    info[0] = uint32_t(error); info[3] = deferred ? 1 : 0; info[4] = successfulCalls;
    info[5] = 1; // Explicit phase completion, including the zero-count path.
    row->finalPending = ready ? 0 : 1; row->finalEnd = GetSystemCycle();
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status), 2 * sizeof(*status));
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(info), 32);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row) + 960, 64);
}

__aicore__ inline void Run(GM_ADDR bindings, GM_ADDR update) {
    auto ctx = GetContext(bindings); auto s = LoadRuntimeShape(ctx);
    uint32_t b = GetBlockIdx(); if (b >= s.batch) return;
    auto a = reinterpret_cast<__gm__ uint64_t*>(update);
    // Device-side diagnostics for missing mandatory binding fields. Normal
    // service calls are constructed with the existing 28-field ABI. Even this
    // early rejection must retire healthy prefetched work and publish failure.
    bool present = true;
    for (uint32_t i = 0; i < 28; ++i) {
        if (i == 7 || i == 12 || i == 13 || i == 15 || i == 24 || i == 25) continue;
        if (!a[i]) present = false;
    }
    if (!present) {
        CompletePrefetch(ctx, b);
        if (Active(ctx, b)) {
            Simt::VF_CALL<InvalidateContextOwner>(Simt::Dim3{kThreads,1,1}, ctx, b, s.resident);
            JoinVector();
        }
        if (a[22] && a[23]) {
            Simt::VF_CALL<PoisonFailedOutputs>(Simt::Dim3{kThreads,1,1}, a, b, s.batch, true);
            JoinVector();
        }
        Simt::VF_CALL<StoreUpdateResult>(Simt::Dim3{kThreads,1,1}, a, b, kLogicInvalid, a[17] != 0);
        JoinVector();
        auto info = reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus) + b * kRowStatusWords;
        auto status = reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(ctx->finalStatus) + b * 2;
        kvaf_device::WriteStatus(status, kvaf_device::kInvalidArgs, 0, 0, 0, 0, UINT64_MAX, 0);
        auto row = reinterpret_cast<__gm__ Row*>(ctx->rows) + b;
        row->finalPending = 1; row->finalEnd = GetSystemCycle();
        info[0] = kLogicInvalid; info[5] = 1;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status), sizeof(*status));
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(info), 32);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row) + 960, 64);
        return;
    }
    TPipe pipe; TBuf<TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, kArenaBytes); auto local = storage.Get<uint32_t>();
    uint32_t preflight = Preflight(ctx, a, s);
    bool active = Active(ctx, b);
    uint32_t error = ResolveOwned(ctx, s, b, local, preflight, active);
    PublishOwned(a, s, b, local, preflight != 0, error != 0 || !active);
    uint32_t owner = reinterpret_cast<__gm__ uint32_t*>(a[9])[b];
    error |= PlanAndApply(a, b, owner, local, active, error);
    // Every low-arena user and event has retired before native Get reuses UB0.
    JoinVector();
    FinishOwned(bindings, update, b, error, active);
    pipe.Destroy();
}
} // namespace fusion3
