#pragma once
#include <stdint.h>

// Experiment-only, zero opt-in pointer in the padded streaming Args. Each
// writer gets its own cache lines. The fixture survives restores of KV/Index.
namespace fixed_post {
constexpr uint32_t kTop = 2048;
constexpr uint32_t kRecord = 1, kFour = 2, kOne = 3;
struct alignas(128) Window {
    uint32_t count, seen, reserved[30], ids[kTop];
};
struct alignas(128) Query {
    uint32_t count, slotBase, slots, prior, candidates, seen, reserved[26];
    uint32_t winners[kTop], misses[kTop];
};
struct alignas(128) Fixture {
    uint32_t mode, reserved[31];
    Window windows[4];
    Query queries[4];
};
static_assert(sizeof(Fixture) == 4 * (32 + 4*(32+2048) + 4*(32+4096)), "fixed list fixture ABI");
}
