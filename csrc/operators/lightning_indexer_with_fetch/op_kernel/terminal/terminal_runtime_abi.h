#pragma once
#include <stddef.h>
#include <stdint.h>

namespace terminal_runtime {
constexpr uint32_t kVersion=1;
constexpr uint32_t kQueries=4;
constexpr uint32_t kTop=2048;
constexpr uint32_t kFlagWordsPerRow=32*32;
constexpr uint32_t kTraceWordsPerQuery=32;

// Keep the production Args and Context ABIs intact. This allocation belongs to
// a single forward binding, survives graph replay and is never host-refreshed
// inside the measured call. Only pointers/immutable version live here.
struct alignas(128) Binding {
    uint64_t pipeline, queueArgs, templates, flags, times, logical;
    uint32_t version, options;
    uint64_t reserved[9];
};
static_assert(sizeof(Binding)==128,"terminal runtime ABI size");
static_assert(offsetof(Binding,version)==48,"terminal runtime version offset");
static_assert(offsetof(Binding,reserved)==56,"terminal runtime reserved offset");

#if defined(__CCE_AICORE__)
#define TERMINAL_OFFSET_DEVICE __aicore__
#else
#define TERMINAL_OFFSET_DEVICE
#endif

// These offsets are shared by the device routing and host regression tests.
TERMINAL_OFFSET_DEVICE constexpr uint64_t QueryIndex(uint32_t row,uint32_t query){return uint64_t(row)*kQueries+query;}
TERMINAL_OFFSET_DEVICE constexpr uint64_t QueryOutputOffset(uint32_t row,uint32_t query){return QueryIndex(row,query)*kTop;}
TERMINAL_OFFSET_DEVICE constexpr uint64_t RowFlagOffset(uint32_t row){return uint64_t(row)*kFlagWordsPerRow;}
TERMINAL_OFFSET_DEVICE constexpr uint64_t QueryTraceOffset(uint32_t row,uint32_t query){return QueryIndex(row,query)*kTraceWordsPerQuery;}
#undef TERMINAL_OFFSET_DEVICE
}
