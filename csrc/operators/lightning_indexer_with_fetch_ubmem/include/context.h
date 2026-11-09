#pragma once
#include <stdint.h>
static_assert(V9_UBPIPE_DEPTH == 1 || V9_UBPIPE_DEPTH == 8, "Invalid UBmem pipeline depth");
static_assert(V9_UBPIPE_ROWS == 1 || V9_UBPIPE_ROWS == 4, "Invalid UBmem rows per pipeline stage");
static_assert(V9_UBPIPE_DEPTH != 1 || V9_UBPIPE_ROWS == 1, "Serial UBmem requires one row per stage");
#include "ubmem_abi.h"
#include "../../common/lifetch/runtime_shape.h"
namespace rotating {
constexpr uint32_t kTop = 2048, kSlots = 8192, kTokenBytes = TOKEN_BYTES;
constexpr uint32_t kMaxRounds = 32, kRowStatusWords = 32, kFinalStateWords = 288;
struct alignas(128) FinalPost {
    uint32_t count, pending, error, deferred, descriptor;
    uint32_t reserved;
    uint64_t head;
};

static_assert(sizeof(FinalPost)==128,"Final post must own a cache line");
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
    RuntimeShape shape;
};
static_assert(sizeof(Row) == 1024 && sizeof(Event) == 128 && sizeof(Context) == 336,
              "V9 ABI layout changed");
#if defined(__CCE_AICORE__)
__aicore__ inline RuntimeShape LoadRuntimeShape(const __gm__ Context* ctx) {
    RuntimeShape result;
    auto input=reinterpret_cast<const __gm__ uint32_t*>(&ctx->shape);
    auto output=reinterpret_cast<uint32_t*>(&result);
    for(uint32_t i=0;i<sizeof(result)/sizeof(uint32_t);++i)output[i]=input[i];
    return result;
}
#endif
// Runtime schedule follows the same hot-first geometry as URMA.
#if defined(__CCE_AICORE__)
#define UB_SCHEDULE_DEVICE __aicore__
#else
#define UB_SCHEDULE_DEVICE
#endif
UB_SCHEDULE_DEVICE inline uint32_t RuntimeTrigger(uint32_t history,uint32_t resident,uint32_t rounds,uint32_t epoch) {
    return resident/128 + ((history-resident)/128)*epoch/(rounds>1?rounds-1:1);
}
UB_SCHEDULE_DEVICE inline bool IsTrigger(uint32_t history,uint32_t resident,uint32_t rounds,uint32_t tile) {
    for(uint32_t epoch=0;epoch<rounds;++epoch)
        if(RuntimeTrigger(history,resident,rounds,epoch)==tile)return true;
    return false;
}
UB_SCHEDULE_DEVICE inline int32_t EpochAt(uint32_t batch,uint32_t history,uint32_t resident,uint32_t split,
    uint32_t rounds,uint32_t serviceAics,uint32_t tile,uint32_t core) {
    uint32_t rank=split?(core%2)*batch+core/2:core,epoch=rank/serviceAics;
    return epoch<rounds && RuntimeTrigger(history,resident,rounds,epoch)==tile ? int32_t(epoch) : -1;
}
#undef UB_SCHEDULE_DEVICE
}
