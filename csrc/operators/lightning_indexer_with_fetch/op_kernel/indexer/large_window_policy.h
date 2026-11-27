// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdint.h>

namespace large_window_policy {

#if defined(__CCE_AICORE__)
#define LARGE_WINDOW_DEVICE __aicore__
#else
#define LARGE_WINDOW_DEVICE
#endif

constexpr uint32_t kAivs = 64;
constexpr uint32_t kQueries = 4;
constexpr uint32_t kMaxRows = kAivs / 2;

struct Cost {
    uint32_t q1;
    uint32_t q2;
    uint32_t q4;
};

LARGE_WINDOW_DEVICE constexpr uint32_t Higher(uint32_t x, uint32_t y) {
    return x > y ? x : y;
}

LARGE_WINDOW_DEVICE constexpr Cost Estimate(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    return {Higher(Higher(a, b), Higher(c, d)), Higher(a + b, c + d), a + b + c + d};
}

LARGE_WINDOW_DEVICE constexpr uint32_t QueriesAt(const Cost& cost, uint32_t limit) {
    if (cost.q4 <= limit) return 4;
    if (cost.q2 <= limit) return 2;
    return 1;
}

LARGE_WINDOW_DEVICE inline uint32_t LaneCount(const Cost* costs, uint32_t rows, uint32_t limit) {
    uint32_t lanes = 0;
    for (uint32_t row = 0; row < rows; ++row) {
        lanes += kQueries / QueriesAt(costs[row], limit);
    }
    return lanes;
}

LARGE_WINDOW_DEVICE inline uint32_t Pack(uint32_t row, uint32_t first, uint32_t queries) {
    // Low 3 bits keep 0 available as the inactive-lane sentinel.
    return ((row * kQueries + first) << 3) | queries;
}

LARGE_WINDOW_DEVICE inline void Plan(const Cost* costs, uint32_t rows, uint32_t* lanes) {
    uint32_t lower = 0, allPair = 0;
    for (uint32_t row = 0; row < rows; ++row) {
        lower = Higher(lower, costs[row].q1);
        allPair = Higher(allPair, costs[row].q2);
    }
    // All rows fit in one wave with two AIVs per row. Minimize the
    // heaviest assigned lane, using the widest Q bundle that meets each bound.
    uint32_t limit = allPair;
    while (lower < limit) {
        const uint32_t probe = lower + (limit - lower) / 2;
        if (LaneCount(costs, rows, probe) > kAivs) {
            lower = probe + 1;
        } else {
            limit = probe;
        }
    }
    uint32_t lane = 0;
    for (uint32_t row = 0; row < rows; ++row) {
        // If the mixed layout cannot shorten the estimated critical lane,
        // use the ordinary dense paired layout.
        const uint32_t queries = limit == allPair ? 2 : QueriesAt(costs[row], limit);
        for (uint32_t q = 0; q < kQueries; q += queries) {
            lanes[lane++] = Pack(row, q, queries);
        }
    }
    for (; lane < kAivs; ++lane) lanes[lane] = 0;
}

#undef LARGE_WINDOW_DEVICE
} // namespace large_window_policy
