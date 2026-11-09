#pragma once
#include <stdint.h>
static_assert(V9_UBPIPE_DEPTH == 1 || V9_UBPIPE_DEPTH == 8, "Invalid UBmem pipeline depth");
static_assert(V9_UBPIPE_ROWS == 1 || V9_UBPIPE_ROWS == 4, "Invalid UBmem rows per pipeline stage");
static_assert(V9_UBPIPE_DEPTH != 1 || V9_UBPIPE_ROWS == 1, "Serial UBmem requires one row per stage");
#include "ubmem_abi.h"
namespace rotating {
constexpr bool kDedicatedService = V9_DEDICATED_SERVICE;
constexpr uint32_t kTop = 2048, kSlots = 8192, kTokenBytes = TOKEN_BYTES;
static_assert(uint64_t(SIM_RESIDENT)*kTokenBytes%32==0,
              "UBmem batch outputs must not share a 32-byte GM sector");
constexpr uint32_t kMaxRounds = 32;
// Scalar cache maintenance must never write back another AIV's status.
constexpr uint32_t kRowStatusWords = 32;
constexpr uint32_t kCoresPerBatch = V9_SPLIT_Q ? 2 : 1;
constexpr uint32_t kLogicalTasks = SIM_BATCH * kCoresPerBatch;
constexpr uint32_t kLiCores =
    kLogicalTasks < V9_AIC_COUNT ? kLogicalTasks : V9_AIC_COUNT;
static_assert(!V9_SPLIT_Q || kLogicalTasks <= V9_AIC_COUNT, "Split-Q exceeds available AICs");
constexpr uint32_t kRounds = kDedicatedService ? V9_INTERMEDIATE_FETCHES :
    (V9_INTERMEDIATE_FETCHES < kLiCores ? V9_INTERMEDIATE_FETCHES : kLiCores);
static_assert(kRounds <= kMaxRounds, "Too many service windows");
#ifndef V9_SERVICE_AICS
#define V9_SERVICE_AICS 0
#endif
constexpr uint32_t kServiceAics=kDedicatedService ? (SIM_BATCH+1)/2 :
    (kRounds ? (V9_SERVICE_AICS ? V9_SERVICE_AICS : kLiCores/kRounds) : 1);
constexpr uint32_t kServiceLanes=2*kServiceAics;
constexpr uint32_t kLaunchCores=kLiCores+(kDedicatedService?kServiceAics:0);
static_assert(!kDedicatedService || (V9_SPLIT_Q && SIM_BATCH<=12 && kRounds==8 && kLaunchCores<=V9_AIC_COUNT),
              "Dedicated service needs Q2, eight epochs and spare physical mixed slots");
static_assert(kDedicatedService || !kRounds || kServiceAics*kRounds<=kLiCores,"Service groups exceed active AICs");

#ifndef V9_FINAL_AIVS
#define V9_FINAL_AIVS 0
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

static_assert(sizeof(FinalPost)==128,"Final post must own a cache line");
static_assert((kFinalWorkers==1 || kFinalWorkers==2 || kFinalWorkers==4) && kFinalCores<=2*kLiCores,"Invalid final AIV budget");
struct alignas(128) Progress { uint32_t end[32]; };
struct alignas(128) Row {
    uint64_t reserved0, reserved1, addresses, output; // former transport slots; always zero
    uint32_t slots, requests, suppressed, error;
    uint32_t consumed[4];
    uint32_t epochCount[kMaxRounds], epochSuppressed[kMaxRounds];
    uint32_t snapshot[kMaxRounds][4];
    uint32_t limit, prefetchLimit, copyLimit, pending;
    uint64_t reserved2;
    uint32_t epochOrder[kMaxRounds], sequence;
    uint32_t finalRequests, finalPending;
    uint64_t finalCompleted, finalBegin, finalEnd;
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
    uint64_t reservedArgs, sourceBlocks, sourceOffsets, destinations, count, rowStatus, reservedFinalStatus;
    uint32_t prefetchLimit, windowLimit;
    uint64_t ubmemRegion, ubmemStatus;
    uint64_t lruSlots, reservedHot, changes, queryEnds;
    uint64_t fallbackAddresses, reservedFallbackValid, hotValid;
    uint64_t rowOwners, tokenAddresses, addressStride, outputStride, tailBounds;
    uint64_t changeSlots, changeCount;
};
static_assert(sizeof(Row) == 1024 && sizeof(Event) == 128 && sizeof(Context) == 256,
              "V9 ABI layout changed");
// Workload configuration is compiled into each benchmark binary, as in V8.
// One selected AIC pair per window; zero disables intermediate Fetch.
#if defined(__CCE_AICORE__)
#define V9_DEVICE __aicore__
#else
#define V9_DEVICE
#endif
// Match URMA: epoch zero handles displaced hot candidates immediately after
// the resident scores; the remaining epochs span the cold region.
V9_DEVICE constexpr uint32_t Trigger(uint32_t epoch) {
    if (kRounds == 0) return SIM_RESIDENT / 128;
    return SIM_RESIDENT / 128 +
        ((SIM_HISTORY - SIM_RESIDENT) / 128) * epoch / (kRounds > 1 ? kRounds - 1 : 1);
}
V9_DEVICE constexpr bool IsTrigger(uint32_t tileEnd) {
    constexpr uint32_t coldTiles = (SIM_HISTORY - SIM_RESIDENT) / 128;
    if constexpr (kRounds == 1) return tileEnd == SIM_RESIDENT / 128;
    if constexpr (kRounds > 1 && coldTiles != 0 && coldTiles % (kRounds > 1 ? kRounds - 1 : 1) == 0) {
        constexpr uint32_t step = coldTiles / (kRounds > 1 ? kRounds - 1 : 1);
        constexpr uint32_t first = SIM_RESIDENT / 128;
        return tileEnd >= first && tileEnd <= first + coldTiles &&
            (tileEnd - first) % (step ? step : 1) == 0;
    }
    for (uint32_t e = 0; e < kRounds; ++e) if (Trigger(e) == tileEnd) return true;
    return false;
}
// Prefer one split-query AIC from each batch before borrowing its sibling.
V9_DEVICE constexpr uint32_t ServiceRank(uint32_t core) {
    return V9_SPLIT_Q ? (core%2)*SIM_BATCH+core/2 : core;
}
V9_DEVICE constexpr uint32_t ServiceCore(uint32_t epoch,uint32_t group) {
    if (kDedicatedService) return kLiCores+group;
    uint32_t rank=epoch*kServiceAics+group;
    return V9_SPLIT_Q ? (rank%SIM_BATCH)*2+rank/SIM_BATCH : rank;
}
V9_DEVICE constexpr uint32_t ServiceLane(uint32_t core,uint32_t sub) {
    return (ServiceRank(core)%kServiceAics)*2+sub;
}
V9_DEVICE constexpr int32_t EpochAt(uint32_t tileEnd, uint32_t core) {
    if (kDedicatedService) return -1;
    uint32_t rank=ServiceRank(core),epoch=rank/kServiceAics;
    return epoch<kRounds && Trigger(epoch)==tileEnd ? int32_t(epoch) : -1;
}
V9_DEVICE constexpr bool DedicatedProgressReady(uint32_t a,uint32_t b,uint32_t c,uint32_t d,uint32_t epoch) {
    const uint32_t ready=Trigger(epoch)*128;
    return a>=ready && b>=ready && c>=ready && d>=ready;
}
}
