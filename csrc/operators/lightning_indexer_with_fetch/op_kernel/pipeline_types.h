#pragma once
#include <stdint.h>
namespace pipeline_ab {
constexpr uint32_t kResolveCores=KVIP_OUTPUT_WORKER?3:2;
constexpr uint32_t kRawQueryFlag=12;
// One word in the first post-Index record. Only a value of 1 enters its tail.
constexpr uint32_t kPostIndexRouteWord = 14;
constexpr uint64_t kReadAdmissionRows = 2;
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
 // Optional invocation-local phase records, inside the existing 256-byte ABI.
 uint64_t postIndex;
};
static_assert(sizeof(Args)==256,"streaming Args ABI");
}

namespace rotating {
// Six uint32 banks, one uint16 first-occurrence-position bank, and control.
// Storing a TopK position (0..8191) instead of its token saves 16 KiB without
// narrowing token IDs. Recover tokens from selected[position] at address gather.
constexpr uint32_t kFinalSlotsOffset = 4 * 8192 + 4096;
constexpr uint32_t kFinalPositionsOffset = kFinalSlotsOffset + 8192;
constexpr uint32_t kFinalControlOffset = kFinalPositionsOffset + 8192;
constexpr uint32_t kFinalScratchWords = kFinalControlOffset + 2560;
constexpr uint32_t kFinalScratchBytes = kFinalScratchWords * sizeof(uint32_t);
static_assert(kFinalScratchBytes <= 223232,
              "final scratch exceeds the compiler user UB reservation");
static_assert(223232 + 6144 <= 229376,
              "user plus CANN 9.1 compiler UB exceeds the SIMT limit");
}
