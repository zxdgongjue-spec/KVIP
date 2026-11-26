#pragma once
#include <stdint.h>

namespace rotating {
// Leave room between the dedup table and output for four 256-ID input rows.
constexpr uint32_t kWindowOutputOffset = 23808 / sizeof(uint32_t);
constexpr uint32_t kDirectWindowWords = kWindowOutputOffset - 4 * 256;
constexpr uint32_t kWindowSetWords = 4096;
constexpr uint32_t kDirectWindowTokens = kDirectWindowWords * 32;
static_assert(kWindowSetWords <= kDirectWindowWords && kDirectWindowTokens % 256 == 0);

// One bitmap bit per physical token. GM bitmap transfers require whole 32-byte blocks.
// The bound is exclusive across all Qs, including lookahead. Padded Qs have end zero.
template<class Ends>
#if defined(__CCE_AICORE__)
__aicore__
#endif
inline uint32_t BatchWindowWords(uint32_t history, uint32_t batch, Ends ends) {
    if (history <= kDirectWindowTokens || !ends) return history / 32;
    uint32_t maximum = 0;
    for (uint32_t i = 0; i < 4 * batch; ++i) {
        const uint32_t end = ends[i];
        if (end > maximum) maximum = end;
    }
    if (maximum >= history) return history / 32;
    return maximum ? ((maximum + 255) / 256) * 8 : 8;
}
}
