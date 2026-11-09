#pragma once
#include <stdint.h>

// This descriptor belongs to a lease on same-host physical pages. A peer VA
// alone is never accepted as a device address. The host binding keeps the
// mapping owner alive until every using graph has been destroyed.
namespace rotating {
constexpr uint64_t kUbmemMagic = 0x55424d454d563931ULL;
constexpr uint64_t kUbmemVersion = 1;
constexpr uint64_t kUbmemHostLocal = 1;
constexpr uint32_t kUbmemStatusWords = 16;
constexpr uint32_t kUbmemInvalidRegion = 0x10000;
constexpr uint32_t kUbmemInvalidAddress = 0x20000;
constexpr uint32_t kUbmemIncomplete = 0x40000;
struct alignas(128) UbmemRegion {
    uint64_t magic, version, peerBase, mappedBase, bytes, generation;
    uint64_t outputBase, outputBytes, device, locality;
};
static_assert(sizeof(UbmemRegion) == 128, "UBmem region ABI changed");
#if defined(__CCE_AICORE__)
#define V9_ADDRESS_FN __aicore__
#else
#define V9_ADDRESS_FN
#endif
V9_ADDRESS_FN inline bool UbmemRange(uint64_t base, uint64_t bytes) {
    return base && bytes && bytes <= UINT64_MAX - base;
}
V9_ADDRESS_FN inline bool UbmemTranslate(uint64_t peerBase, uint64_t mappedBase,
    uint64_t bytes, uint64_t address, uint64_t width, uint64_t& mapped) {
    if (!UbmemRange(peerBase, bytes) || !UbmemRange(mappedBase, bytes) ||
        address < peerBase || !width || width > bytes) return false;
    uint64_t offset = address - peerBase;
    if (offset > bytes - width) return false;
    mapped = mappedBase + offset;
    return true;
}
#undef V9_ADDRESS_FN
}
