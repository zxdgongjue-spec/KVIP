// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "dynamic_schedule.h"

#if defined(__CCE_AICORE__)
#define KVIP_PARTITIONED_DISPATCH_DEVICE __aicore__
#else
#define KVIP_PARTITIONED_DISPATCH_DEVICE
#endif

namespace kvip_partitioned_dispatch {
constexpr uint32_t kRows = 2;
constexpr uint32_t kParts = 4;
constexpr uint32_t kWindows = 4;
constexpr uint32_t kScoreGroups = kRows * kParts;
constexpr uint32_t kScoreAivs = kScoreGroups * 2;
constexpr uint32_t kAicAck = kScoreGroups;
constexpr uint32_t kAivAck = kAicAck + kScoreGroups;
constexpr uint32_t kLines = kAivAck + kScoreAivs;
constexpr uint32_t kJobs = kScoreGroups * (kWindows - 1);

struct Dispatcher {
    uint32_t nextWindow[kScoreGroups]{1, 1, 1, 1, 1, 1, 1, 1};
    uint32_t assigned[kScoreGroups]{};
    uint32_t nextTicket = 0;
    uint32_t completed = 0;

    template<class Bus>
    KVIP_PARTITIONED_DISPATCH_DEVICE bool Step(Bus& bus) {
        for (uint32_t group = 0; group < kScoreGroups; ++group) {
            const uint32_t job = assigned[group];
            const uint32_t peer = group * 2;
            if (!job || bus.Read(kAicAck + group) != job ||
                bus.Read(kAivAck + peer) != job || bus.Read(kAivAck + peer + 1) != job) continue;
            const uint32_t logical = dynamic_li::Row(job) * kParts + dynamic_li::Fragment(job);
            nextWindow[logical] = dynamic_li::Window(job) + 1;
            assigned[group] = 0;
            ++completed;
        }
        if (completed == kJobs) return true;

        for (uint32_t group = 0; group < kScoreGroups; ++group) {
            if (assigned[group]) continue;
            uint32_t oldest = kWindows;
            uint32_t logical = kScoreGroups;
            for (uint32_t candidate = 0; candidate < kScoreGroups; ++candidate) {
                const uint32_t epoch = nextWindow[candidate];
                if (!epoch || epoch >= oldest) continue;
                oldest = epoch;
                logical = candidate;
            }
            if (logical == kScoreGroups) break;
            const uint32_t job = (++nextTicket << 8) |
                ((logical / kParts) << 4) | (oldest << 2) | (logical % kParts);
            // A shard has one monotonic cursor across all windows. Its next
            // window may use a different MIX group only after all peers ACK.
            nextWindow[logical] = 0;
            assigned[group] = job;
            bus.Write(group, job);
        }
        return false;
    }
};
}

#undef KVIP_PARTITIONED_DISPATCH_DEVICE
