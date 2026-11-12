#pragma once
#include <stdint.h>
#ifndef KVIP_OUTPUT_WORKER
#define KVIP_OUTPUT_WORKER 0
#endif
namespace pipeline_ab {
constexpr uint32_t kResolveCores=KVIP_OUTPUT_WORKER?3:2;
constexpr uint32_t kRawQueryFlag=12;
constexpr uint32_t kPrefetch=2048, kQueries=4;
// Only the current Q needs first-occurrence IDs; slots/hash ranks span all Qs.
constexpr uint32_t kStagedSlotsOffset=4*8192+1024;
constexpr uint32_t kStagedPositionsOffset=kStagedSlotsOffset+8192;
constexpr uint32_t kStagedControlOffset=kStagedPositionsOffset+8192;
static_assert((kStagedControlOffset+1024+4096)*4<=223232,"streaming resolver UB budget");
constexpr uint32_t kMaxFinal=8192, kMapSize=4096, kChunk=32;
constexpr uint32_t kFlagWords=32, kFlagCount=KVIP_OUTPUT_WORKER?16:12;
// Each flag is isolated to avoid stale reads/false sharing across AIVs.
struct alignas(128) BatchFlag { uint32_t ready, count, error, reserved[29]; };
static_assert(sizeof(BatchFlag)==128,"one cache line per flag");
struct alignas(128) Args {
 uint64_t context, keys, scales, addresses, logical, valid;
 uint64_t flags, eventsSrc, eventsDst, workerStatus, times;
 uint32_t history, resident, prefetchCount, maxEvents;
 uint64_t queryBitmaps, queryErrors, updateBindings, denseShadow, packedFetch;
};
}
