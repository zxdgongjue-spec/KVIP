#pragma once
#include <cstdint>
#ifndef V9_USER_UB_BYTES
#error "V9_USER_UB_BYTES must match the compiler user-reserved-ub-size option"
#endif
namespace rotating {
// Six uint32 banks, one uint16 first-occurrence-position bank, and control.
// Storing a TopK position (0..8191) instead of its token saves 16 KiB without
// narrowing token IDs. Recover tokens from selected[position] at address gather.
constexpr uint32_t kFinalSlotsOffset = 4 * 8192 + 4096;
constexpr uint32_t kFinalPositionsOffset = kFinalSlotsOffset + 8192;
constexpr uint32_t kFinalControlOffset = kFinalPositionsOffset + 8192;
constexpr uint32_t kFinalScratchWords = kFinalControlOffset + 2560;
constexpr uint32_t kFinalScratchBytes = kFinalScratchWords * sizeof(uint32_t);
static_assert(kFinalScratchBytes <= V9_USER_UB_BYTES,
              "final scratch exceeds the compiler user UB reservation");
static_assert(V9_USER_UB_BYTES + 6144 <= 229376,
              "user plus CANN 9.1 compiler UB exceeds the SIMT limit");
}
