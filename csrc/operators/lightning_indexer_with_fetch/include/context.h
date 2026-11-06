#pragma once
#include <stdint.h>
#include "batch_get_abi.h"
#include "runtime_shape.h"
namespace rotating {
constexpr uint32_t kTop=2048, kSlots=8192, kMaxRounds=32, kRowStatusWords=32;
constexpr bool kFinalReplicated=true;
constexpr uint32_t kFinalStateWords=288;
struct alignas(128) FinalPost {
    uint32_t count, pending, error, deferred, descriptor;
    uint32_t reserved;
    uint64_t head;
};
static_assert(sizeof(FinalPost)==128,"Final post must own a cache line");
struct alignas(128) Progress { uint32_t end[32]; };
struct alignas(128) Row {
    uint64_t entity, queue, addresses, output;
    uint32_t slots, requests, suppressed, error;
    uint32_t consumed[4];
    uint32_t epochCount[kMaxRounds], epochSuppressed[kMaxRounds];
    uint32_t snapshot[kMaxRounds][4];
    uint32_t limit, prefetchLimit, queueLimit, pending;
    uint64_t initialHead;
    uint32_t epochOrder[kMaxRounds], sequence;
    uint32_t finalRequests, finalPending;
    uint64_t finalHead, finalBegin, finalEnd;
};
struct alignas(128) Event {
    uint64_t begin, end;
    uint32_t core, epoch, complete, skipped;
    uint64_t waitBegin;
    uint64_t scanTicks, candidateTicks, postTicks, bitmapBytes, candidateCount, postedCount;
    uint32_t pad[10];
};
struct Context {
    uint64_t progress, rows, mapping, locations, events, hits, selected, owners;
    uint64_t args, sourceBlocks, sourceOffsets, destinations, count, rowStatus, finalStatus;
    uint32_t prefetchLimit, windowLimit;
    uint64_t lruSlots, reservedHot, changes; // Optional direct-hot interface; 2048 LRU candidates/request.
    uint64_t queryEnds; // uint32[B,4], exclusive causal token bounds; zero for padding.
    uint64_t fallbackAddresses, fallbackValid; // Per-descriptor sources after Index permutations.
    uint64_t hotValid; // uint32[owners,resident]; empty service slots are not hits.
    uint64_t rowOwners, tokenAddresses, addressStride, outputStride, tailBounds;
    RuntimeShape shape;
    uint64_t changeSlots, changeCount; // Extended direct-hot producer records, ABI 3.
    // Optional serving layout: active row -> stable owner; addresses stay with
    // that owner's Index permutation. tailBounds[B,2] are [committed,write_end).

};
static_assert(sizeof(Row) == 1024 && sizeof(Event) == 128 && sizeof(Context) == 320,
              "LIFetch runtime ABI layout changed");
// Workload geometry is bound before capture and shared by host/device launches.
#if defined(__CCE_AICORE__)
#define V9_DEVICE __aicore__
#else
#define V9_DEVICE
#endif
#if defined(__CCE_AICORE__)
__aicore__ inline RuntimeShape LoadRuntimeShape(const __gm__ Context* ctx) {
    RuntimeShape result;
    auto input=reinterpret_cast<const __gm__ uint32_t*>(&ctx->shape);
    auto output=reinterpret_cast<uint32_t*>(&result);
    for(uint32_t i=0;i<sizeof(result)/sizeof(uint32_t);++i)output[i]=input[i];
    return result;
}
#endif
// Even views remain prefetch-owned. Reuse spare odd views, then extra views.
V9_DEVICE inline uint32_t FinalQueue(uint32_t rt_batch,uint32_t rt_finalWorkers,uint32_t batch,uint32_t worker) {
    return worker==0 ? batch*2+1 : rt_batch*2+batch*(rt_finalWorkers-1)+worker-1;
}
V9_DEVICE inline uint32_t Trigger(uint32_t rt_history,uint32_t rt_resident,uint32_t rt_rounds,uint32_t epoch) {
    // Zero rounds disables service; keep this helper valid without division by zero.
    if (rt_rounds == 0) return rt_resident / 128;
    return rt_resident / 128 +
        ((rt_history - rt_resident) / 128) * epoch / (rt_rounds > 1 ? rt_rounds - 1 : 1);
}
V9_DEVICE inline bool IsTrigger(uint32_t rt_history,uint32_t rt_resident,uint32_t rt_rounds,uint32_t tileEnd) {
    for (uint32_t e = 0; e < rt_rounds; ++e) if (Trigger(rt_history,rt_resident,rt_rounds,e) == tileEnd) return true;
    return false;
}
// Prefer one split-query AIC from each batch before borrowing its sibling.
V9_DEVICE inline uint32_t ServiceRank(uint32_t rt_batch,uint32_t rt_splitQ,uint32_t core) {
    return rt_splitQ ? (core%2)*rt_batch+core/2 : core;
}
V9_DEVICE inline uint32_t ServiceCore(uint32_t rt_batch,uint32_t rt_splitQ,uint32_t rt_serviceAics,uint32_t epoch,uint32_t group) {
    uint32_t rank=epoch*rt_serviceAics+group;
    return rt_splitQ ? (rank%rt_batch)*2+rank/rt_batch : rank;
}
V9_DEVICE inline uint32_t ServiceLane(uint32_t rt_batch,uint32_t rt_splitQ,uint32_t rt_serviceAics,uint32_t core,uint32_t sub) {
    return (ServiceRank(rt_batch,rt_splitQ,core)%rt_serviceAics)*2+sub;
}
V9_DEVICE inline int32_t EpochAt(uint32_t rt_batch,uint32_t rt_history,uint32_t rt_resident,uint32_t rt_splitQ,uint32_t rt_rounds,uint32_t rt_serviceAics,uint32_t tileEnd, uint32_t core) {
    uint32_t rank=ServiceRank(rt_batch,rt_splitQ,core),epoch=rank/rt_serviceAics;
    return epoch<rt_rounds && Trigger(rt_history,rt_resident,rt_rounds,epoch)==tileEnd ? int32_t(epoch) : -1;
}
}
