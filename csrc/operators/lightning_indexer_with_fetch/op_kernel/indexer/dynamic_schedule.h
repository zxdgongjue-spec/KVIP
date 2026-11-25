// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdint.h>

#if defined(__CCE_AICORE__)
#define KVIP_DYN_DEVICE __aicore__
#else
#define KVIP_DYN_DEVICE
#endif

namespace dynamic_li {
constexpr uint32_t kMix = 32;
constexpr uint32_t kAiv = 2 * kMix;
constexpr uint32_t kRows = 16;
constexpr uint32_t kWindows = 4;
constexpr uint32_t kFragments = 3;
constexpr uint32_t kScoreSlots = kMix;
constexpr uint32_t kAckSlots = kScoreSlots + kAiv;
constexpr uint32_t kLines = kAckSlots + kAiv + kMix;
constexpr uint32_t kLineBytes = 128;
constexpr uint32_t kResidentReady = 1;
constexpr uint32_t kStop = UINT32_MAX;

KVIP_DYN_DEVICE inline uint32_t Row(uint32_t job) { return (job >> 4) & 15; }
KVIP_DYN_DEVICE inline uint32_t Window(uint32_t job) { return (job >> 2) & 3; }
KVIP_DYN_DEVICE inline uint32_t Fragment(uint32_t job) { return job & 3; }
KVIP_DYN_DEVICE inline uint32_t Mask(uint32_t window) { return 7U << ((window - 1) * kFragments); }

struct Dispatcher {
    uint32_t nextTicket = 1;
    uint32_t completeServices = 0;
    uint64_t unseen = UINT64_MAX;
    uint64_t idle = 0;
    uint64_t busyService = 0;
    uint32_t busyScore = 0;
    uint32_t assignedServiceRows = 0;
    uint32_t score[kMix]{};
    uint32_t service[kAiv]{};
    uint32_t issued[kRows]{};
    uint32_t finished[kRows]{};
    uint32_t nextWindow[kRows]{};

    KVIP_DYN_DEVICE uint32_t Job(uint32_t row, uint32_t window, uint32_t fragment) {
        return (++nextTicket << 8) | (row << 4) | (window << 2) | fragment;
    }

    template <class Bus>
    KVIP_DYN_DEVICE bool Step(Bus& bus) {
        for (uint64_t pending = unseen; pending; pending &= pending - 1) {
            const uint32_t lane = __builtin_ctzll(pending);
            const uint64_t bit = uint64_t{1} << lane;
            if (bus.AivAck(lane) == kResidentReady) {
                unseen &= ~bit;
                idle |= bit;
            }
        }
        for (uint32_t pending = busyScore; pending; pending &= pending - 1) {
            const uint32_t group = __builtin_ctz(pending);
            const uint32_t job = score[group];
            const uint32_t lane = group * 2;
            if (bus.AicAck(group) == job && bus.AivAck(lane) == job && bus.AivAck(lane + 1) == job) {
                finished[Row(job)] |= 1U << ((Window(job) - 1) * kFragments + Fragment(job));
                busyScore &= ~(1U << group);
                idle |= uint64_t{3} << lane;
            }
        }
        for (uint64_t pending = busyService; pending; pending &= pending - 1) {
            const uint32_t lane = __builtin_ctzll(pending);
            const uint32_t job = service[lane];
            const uint64_t bit = uint64_t{1} << lane;
            if (bus.AivAck(lane) == job) {
                const uint32_t row = Row(job);
                assignedServiceRows &= ~(1U << row);
                ++nextWindow[row];
                ++completeServices;
                busyService &= ~bit;
                idle |= bit;
            }
        }
        if (completeServices == kRows * kWindows) return true;

        // Prefer AIVs without a free peer and those paired with the dispatcher.
        // This leaves more complete peer pairs available to score.
        for (uint32_t row = 0; row < kRows && idle; ++row) {
            const uint32_t epoch = nextWindow[row];
            if (epoch == kWindows || (assignedServiceRows & (1U << row)) ||
                (unseen & (uint64_t{15} << (row * 4))) ||
                (epoch && (finished[row] & Mask(epoch)) != Mask(epoch))) continue;
            const uint64_t peers = ((idle & 0x5555555555555555ULL) << 1) |
                                   ((idle & 0xAAAAAAAAAAAAAAAAULL) >> 1);
            const uint64_t preferred = idle & (~peers | 3);
            const uint32_t lane = __builtin_ctzll(preferred ? preferred : idle);
            const uint64_t bit = uint64_t{1} << lane;
            const uint32_t job = Job(row, epoch, 0);
            service[lane] = job;
            assignedServiceRows |= 1U << row;
            idle &= ~bit;
            busyService |= bit;
            bus.SendService(lane, job);
        }

        // Windows are issued in order of urgency across ready requests.
        uint64_t pairs = idle & (idle >> 1) & 0x5555555555555554ULL;
        for (uint32_t epoch = 1; epoch < kWindows && pairs; ++epoch) {
            for (uint32_t row = 0; row < kRows && pairs; ++row) {
                if ((unseen & (uint64_t{15} << (row * 4))) || epoch > nextWindow[row] + 1) continue;
                uint32_t missing = Mask(epoch) & ~issued[row];
                while (missing && pairs) {
                    const uint32_t lane = __builtin_ctzll(pairs);
                    const uint32_t group = lane / 2;
                    const uint32_t cursor = __builtin_ctz(missing);
                    const uint32_t job = Job(row, epoch, cursor % kFragments);
                    issued[row] |= 1U << cursor;
                    score[group] = job;
                    busyScore |= 1U << group;
                    idle &= ~(uint64_t{3} << lane);
                    pairs &= pairs - 1;
                    missing &= missing - 1;
                    bus.SendScore(group, job);
                }
            }
        }
        return false;
    }
};
}

#undef KVIP_DYN_DEVICE
