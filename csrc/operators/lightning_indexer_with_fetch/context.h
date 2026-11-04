#pragma once
#include <stdint.h>
#include "batch_get_abi.h"
namespace rotating {
#ifndef V9_TOKEN_BYTES
#define V9_TOKEN_BYTES 656
#endif
// Transport row width only; LI keys and tiling are unchanged.
constexpr uint32_t kTop = 2048, kSlots = 8192, kTokenBytes = V9_TOKEN_BYTES;
static_assert(kTokenBytes == 656 || kTokenBytes == 4 * 656, "Expected one or four Target layers");
constexpr uint32_t kMaxRounds = 32;
// Scalar cache maintenance must never write back another AIV's status.
constexpr uint32_t kRowStatusWords = 32;
constexpr uint32_t kCoresPerBatch = V9_SPLIT_Q ? 2 : 1;
constexpr uint32_t kLogicalTasks = SIM_BATCH * kCoresPerBatch;
constexpr uint32_t kLiCores = kLogicalTasks < V9_AIC_COUNT ? kLogicalTasks : V9_AIC_COUNT;
static_assert(!V9_SPLIT_Q || kLogicalTasks <= V9_AIC_COUNT, "Split-Q exceeds available AICs");
constexpr uint32_t kRounds = V9_INTERMEDIATE_FETCHES < kLiCores ? V9_INTERMEDIATE_FETCHES : kLiCores;
static_assert(kRounds <= kMaxRounds, "Too many service windows");
#ifndef V9_SERVICE_AICS
#define V9_SERVICE_AICS 0
#endif
constexpr uint32_t kServiceAics=kRounds ? (V9_SERVICE_AICS ? V9_SERVICE_AICS : kLiCores/kRounds) : 1;
constexpr uint32_t kServiceLanes=2*kServiceAics;
static_assert(!kRounds || kServiceAics*kRounds<=kLiCores,"Service groups exceed active AICs");

#ifndef V9_FINAL_AIVS
#define V9_FINAL_AIVS 0
#endif
#ifndef V9_FINAL_REPLICATED
#define V9_FINAL_REPLICATED 1
#endif
constexpr bool kFinalReplicated=V9_FINAL_REPLICATED;
constexpr uint32_t kFinalWorkers=V9_FINAL_AIVS ? V9_FINAL_AIVS : (2*kLiCores>=4*SIM_BATCH?4:(2*kLiCores>=2*SIM_BATCH?2:1));
constexpr uint32_t kFinalCores=SIM_BATCH*kFinalWorkers;
constexpr uint32_t kFinalStateWords=288;
constexpr uint32_t kFinalPostOffset=SIM_BATCH*128+kFinalCores*kFinalStateWords*4;
constexpr uint32_t kFinalOwnerBytes=kFinalPostOffset+kFinalCores*128;
struct alignas(128) FinalPost {
    uint32_t count, pending, error, deferred, descriptor;
    uint32_t reserved;
    uint64_t head;
};
constexpr uint32_t kQueueViews=SIM_BATCH*(kFinalWorkers+1);
static_assert(kQueueViews<=2*KVAF_AIV_BATCH_GET_MAX_AIV_CORES,"Too many leased KVAF queues");
static_assert(sizeof(FinalPost)==128,"Final post must own a cache line");
static_assert((kFinalWorkers==1 || kFinalWorkers==2 || kFinalWorkers==4) && kFinalCores<=2*kLiCores,"Invalid final AIV budget");
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
    // Optional serving layout: active row -> stable owner; addresses stay with
    // that owner's Index permutation. tailBounds[B,2] are [committed,write_end).

};
static_assert(sizeof(Row) == 1024 && sizeof(Event) == 128 && sizeof(Context) == 224,
              "V9 ABI layout changed");
// Workload configuration is compiled into each benchmark binary, as in V8.
// One selected AIC pair per window; zero disables intermediate Fetch.
#if defined(__CCE_AICORE__)
#define V9_DEVICE __aicore__
#else
#define V9_DEVICE
#endif
// Even views remain prefetch-owned. Reuse spare odd views, then extra views.
V9_DEVICE constexpr uint32_t FinalQueue(uint32_t batch,uint32_t worker) {
    return worker==0 ? batch*2+1 : SIM_BATCH*2+batch*(kFinalWorkers-1)+worker-1;
}
V9_DEVICE constexpr uint32_t Trigger(uint32_t epoch) {
    // Zero rounds disables service; keep this helper valid without division by zero.
    if (kRounds == 0) return SIM_RESIDENT / 128;
    return SIM_RESIDENT / 128 +
        ((SIM_HISTORY - SIM_RESIDENT) / 128) * epoch / (kRounds > 1 ? kRounds - 1 : 1);
}
V9_DEVICE constexpr bool IsTrigger(uint32_t tileEnd) {
    for (uint32_t e = 0; e < kRounds; ++e) if (Trigger(e) == tileEnd) return true;
    return false;
}
// Prefer one split-query AIC from each batch before borrowing its sibling.
V9_DEVICE constexpr uint32_t ServiceRank(uint32_t core) {
    return V9_SPLIT_Q ? (core%2)*SIM_BATCH+core/2 : core;
}
V9_DEVICE constexpr uint32_t ServiceCore(uint32_t epoch,uint32_t group) {
    uint32_t rank=epoch*kServiceAics+group;
    return V9_SPLIT_Q ? (rank%SIM_BATCH)*2+rank/SIM_BATCH : rank;
}
V9_DEVICE constexpr uint32_t ServiceLane(uint32_t core,uint32_t sub) {
    return (ServiceRank(core)%kServiceAics)*2+sub;
}
V9_DEVICE constexpr int32_t EpochAt(uint32_t tileEnd, uint32_t core) {
    uint32_t rank=ServiceRank(core),epoch=rank/kServiceAics;
    return epoch<kRounds && Trigger(epoch)==tileEnd ? int32_t(epoch) : -1;
}
}
