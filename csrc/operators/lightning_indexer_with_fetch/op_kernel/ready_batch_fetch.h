// SPDX-License-Identifier: Apache-2.0
// A Q worker may submit published rows without joining every row in its shard.
#pragma once
#include <stdint.h>

#ifndef KVIP_READY_BATCH_FETCH
#define KVIP_READY_BATCH_FETCH 0
#endif
#if KVIP_READY_BATCH_FETCH != 0 && KVIP_READY_BATCH_FETCH != 1
#error "KVIP_READY_BATCH_FETCH must be 0 or 1"
#endif

namespace pipeline_ab {
namespace ready_fetch {
constexpr uint32_t kMaxBatch = 16;
constexpr uint32_t kTop = 2048;
constexpr uint32_t kSlots = 4 * kTop;
constexpr uint32_t kMinimumBatch = 128;
constexpr uint64_t kTimeoutCycles = 5000000000ULL;

struct Publication {
    bool ready;
    uint64_t error;
    uint32_t descriptor, count, safe;
};
struct Completion { bool ready; uint64_t error; };
struct Range { uint32_t begin, count, output; };
struct Submitted { uint64_t error; uint32_t completed; };
struct Result {
    uint64_t error = 0;
    uint32_t completed = 0;
    uint32_t requests = 0;
    uint32_t calls = 0;
};
struct Row {
    bool seen = false;
    uint32_t descriptor = 0, count = 0, safe = 0, next = 0;
};

// IO supplies cache-acquiring nonblocking polls and the existing synchronous
// native BatchGet. Its two Jettys have exactly this worker as owner. No queue
// or descriptor staging is reused while a native call remains outstanding.
template<class IO>
__aicore__ inline Result Run(IO& io, uint32_t batch, uint32_t query,
                            uint32_t lane, uint32_t workers, uint32_t capacity) {
    Result result;
    if (!batch || batch > kMaxBatch || query >= 4 ||
        (workers != 1 && workers != 2 && workers != 4) || capacity < batch * kTop) {
        result.error = io.Invalid();
        return result;
    }
    if (lane >= batch) return result;
    Row rows[kMaxBatch];
    Range ranges[kMaxBatch];
    uint32_t expected = 0;
    uint64_t progress = io.Now();
    while (!result.error) {
        uint32_t size = 0, rangeCount = 0, queued = 0;
        bool pending = false, changed = false;
        // Below one 128-WQE post, allow one additional bounded scan to combine
        // newly published rows. Never wait indefinitely to fill a small batch.
        for (uint32_t scan = 0; scan < 2; ++scan) {
            pending = false;
            for (uint32_t b = lane; b < batch; b += workers) {
                auto& row = rows[b];
                if (!row.seen) {
                    const auto p = io.Query(b);
                    if (!p.ready) { pending = true; continue; }
                    result.error |= p.error;
                    if (p.count > kTop || p.safe > p.count ||
                        p.descriptor != b * kSlots + query * kTop ||
                        p.descriptor > batch * kSlots || p.count > batch * kSlots - p.descriptor)
                        result.error |= io.Invalid();
                    if (result.error) break;
                    row.seen = true;
                    row.descriptor = p.descriptor;
                    row.count = p.count;
                    row.safe = p.safe;
                    expected += p.count;
                    changed = true;
                }
                if (row.next == row.count) continue;
                pending = true;
                // A row contributes at most one range per call, so the existing
                // B-entry range scratch suffices even when both phases are ready.
                if (queued & (1U << b)) continue;
                uint32_t end = row.safe;
                if (row.count > row.safe) {
                    const auto prefetch = io.Prefetch(b);
                    if (prefetch.ready) {
                        result.error |= prefetch.error;
                        if (result.error) break;
                        end = row.count;
                    }
                }
                if (end <= row.next) continue;
                const uint32_t count = end - row.next;
                if (rangeCount >= batch || size > capacity || count > capacity - size) {
                    result.error |= io.Invalid();
                    break;
                }
                ranges[rangeCount++] = {row.descriptor + row.next, count, size};
                size += count;
                row.next = end;
                queued |= 1U << b;
                changed = true;
            }
            if (result.error || size >= kMinimumBatch || !size || !pending) break;
        }
        if (result.error) break;
        if (size) {
            const auto submitted = io.Submit(ranges, rangeCount, size);
            result.error |= submitted.error;
            result.completed += submitted.completed;
            ++result.calls;
            if (result.error) break;
            result.requests += size;
        }
        if (!pending) {
            if (result.requests != expected) result.error |= io.Invalid();
            break;
        }
        if (changed) progress = io.Now();
        else if (io.Now() - progress > kTimeoutCycles) {
            result.error |= io.Timeout();
            break;
        }
    }
    return result;
}
}  // namespace ready_fetch
}  // namespace pipeline_ab
