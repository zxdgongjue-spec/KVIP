// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

#ifndef KVIP_COMPACT_INIT
#define KVIP_COMPACT_INIT 0
#endif
#ifndef KVIP_LOW_MISS_SPARSE_INIT
#define KVIP_LOW_MISS_SPARSE_INIT 0
#endif

#if defined(__CCE_AICORE__)
#define KVIP_ROUND_INIT_DEVICE __aicore__
#else
#define KVIP_ROUND_INIT_DEVICE
#endif

namespace kvip_round_init {
#if KVIP_LOW_MISS_SPARSE_INIT && KVIP_AUTO_ROUTE && KVIP_SELECTED_METADATA && \
    KVIP_LOW_MISS_TAIL && KVIP_LOW_MISS_SKIP && KVIP_STREAMING_LI && \
    !(defined(V9_UBMEM) && V9_UBMEM)
constexpr bool kSparseMapping = true;
#else
constexpr bool kSparseMapping = false;
#endif
KVIP_ROUND_INIT_DEVICE inline constexpr bool LowPostIndexRound(bool streaming, bool noPrefetch, bool postIndex) {
#if KVIP_LOW_MISS_POST_INDEX && !(defined(V9_UBMEM) && V9_UBMEM)
    return streaming && noPrefetch && postIndex;
#else
    return false;
#endif
}
KVIP_ROUND_INIT_DEVICE inline constexpr bool ClearEvents(bool lowPostIndex) {
#if KVIP_SKIP_EXHAUSTED_PREFETCH
    return true;
#else
    return !lowPostIndex;
#endif
}
KVIP_ROUND_INIT_DEVICE inline constexpr bool ClearMapping(bool streaming, bool noPrefetch) {
    // With no admitted prefetch, selected-metadata classification resolves
    // hot/ABC slots from current inputs and never reads the prefetch map.
    // Legacy and high routes retain their full initialization contract.
    return !(kSparseMapping && streaming && noPrefetch);
}
#if KVIP_COMPACT_INIT && !(defined(V9_UBMEM) && V9_UBMEM)
constexpr unsigned kFillWords = 16384;
KVIP_ROUND_INIT_DEVICE inline constexpr unsigned ChangeWords(bool streaming, unsigned prefix, unsigned capacity) {
    // Streaming reads only changes[0:prefix] for prefetch/residency. Final
    // Index updates consume separate, explicitly published event ranges.
    // The extended changes suffix can contain holes even below changeCount
    // (reserved destinations update the prefix instead). It must not be read
    // by this path. Keep full clearing for standalone/legacy update readers.
    return streaming ? prefix : capacity;
}
#else
constexpr unsigned kFillWords = 4096;
KVIP_ROUND_INIT_DEVICE inline constexpr unsigned ChangeWords(bool, unsigned, unsigned capacity) {
    return capacity;
}
#endif
// InitializeRound owns no other TPipe buffers and destroys this allocation
// before the following tail-mapping SIMT call. Every DMA count is in words.
static_assert(kFillWords % 8 == 0 && kFillWords * sizeof(uint32_t) <= 65536,
              "round fill must fit the 64 KiB UB reservation and 32-byte DMA blocks");
}  // namespace kvip_round_init
#undef KVIP_ROUND_INIT_DEVICE
