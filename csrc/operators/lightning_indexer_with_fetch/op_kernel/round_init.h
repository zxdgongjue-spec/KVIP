// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

#ifndef KVIP_COMPACT_INIT
#define KVIP_COMPACT_INIT 0
#endif

#if defined(__CCE_AICORE__)
#define KVIP_ROUND_INIT_DEVICE __aicore__
#else
#define KVIP_ROUND_INIT_DEVICE
#endif

namespace kvip_round_init {
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
