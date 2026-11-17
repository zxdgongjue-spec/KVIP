// SPDX-License-Identifier: Apache-2.0
// Execute each warp group with real CPU threads. Groups run sequentially:
// their events are disjoint, and thread-zero Profile initialization precedes
// every group's counter. This checks data and warp instructions; CANN tests
// cover full-block barriers, VF completion and cross-kernel GM visibility.
#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>
#define __simt_vf__
#define __aicore__
#define __gm__
#define __ubuf__
#define LAUNCH_BOUND(n)
thread_local struct {
    uint32_t x = 0;
} threadIdx;
thread_local bool serialValidationMode = false;
struct CpuWarpGroup {
    uint32_t width;
    std::barrier<> barrier;
    std::array<uint32_t, 32> values{};
    explicit CpuWarpGroup(uint32_t size) : width(size), barrier(size) {}
};
thread_local CpuWarpGroup *currentWarp = nullptr;
void asc_syncthreads() {
    if (!serialValidationMode) {
        currentWarp->barrier.arrive_and_wait();
    }
}
void asc_threadfence() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
}
template <class T> T asc_ldcg(T *p) {
    return *p;
}
template <class T> void asc_stcg(T *p, T v) {
    *p = v;
}
void asc_atomic_or(uint32_t *p, uint32_t x) {
    std::atomic_ref<uint32_t>(*p).fetch_or(x);
}
void asc_atomic_add(uint32_t *p, uint32_t x) {
    std::atomic_ref<uint32_t>(*p).fetch_add(x);
}
uint32_t asc_shfl(uint32_t value, int source, int width) {
    assert(uint32_t(width) == currentWarp->width && source >= 0 && source < width);
    currentWarp->values[threadIdx.x % currentWarp->width] = value;
    currentWarp->barrier.arrive_and_wait();
    uint32_t result = currentWarp->values[source];
    currentWarp->barrier.arrive_and_wait();
    return result;
}
uint32_t asc_ballot(int32_t predicate) {
    currentWarp->values[threadIdx.x % currentWarp->width] = predicate != 0;
    currentWarp->barrier.arrive_and_wait();
    uint32_t result = 0;
    for (int i = 0; i < int(currentWarp->width); ++i) {
        result |= currentWarp->values[i] << i;
    }
    result <<= (threadIdx.x % 32) / currentWarp->width * currentWarp->width;
    currentWarp->barrier.arrive_and_wait();
    return result;
}
int __popc(uint32_t x) {
    return __builtin_popcount(x);
}
#include "post_index_update.h"

struct TestData {
    uint32_t batch, cores, history, resident, eventStride, eventOffset = 5, addressStride;
    std::vector<uint32_t> prefix, owners, sources, destinations, scales;
    std::vector<uint64_t> keys, addresses, logical, records;
    TestData(uint32_t batchSize, uint32_t coreCount, int scenario)
        : batch(batchSize), cores(coreCount), history(scenario == 3 ? 32768 : 1024),
          resident(history / 2), eventStride(scenario == 3 ? 8199 : 519),
          addressStride(history + 13), prefix(batchSize + 1), owners(batchSize),
          sources(batchSize * eventStride, UINT32_MAX),
          destinations(batchSize * eventStride, UINT32_MAX), scales((batchSize + 1) * history),
          keys((batchSize + 1) * history * 16), addresses((batchSize + 1) * addressStride),
          logical((batchSize + 1) * history), records(coreCount * 16) {
        assert(coreCount >= batchSize && coreCount <= 64 && batchSize <= 16);
        // Only word zero is the error. Neighbouring count/timestamp/state fields
        // are deliberately nonzero so a misplaced half-word read cannot pass.
        for (uint32_t core = 0; core < coreCount; ++core) {
            for (uint32_t word = 1; word < 16; ++word) {
                records[core * 16 + word] = 0xabc0000000000000ULL | (uint64_t(core) << 8) | word;
            }
        }
        for (uint32_t row = 0; row < batchSize; ++row) {
            owners[row] = batchSize - row;
            uint32_t count = scenario == 0    ? 0
                             : scenario == 3  ? 8192
                             : batchSize == 1 ? 241
                             : row % 4 == 0   ? 0
                             : row % 4 == 1   ? 127
                             : row % 4 == 2   ? 1
                                              : 39;
            prefix[row + 1] = prefix[row] + count;
            for (uint32_t event = 0; event < count; ++event) {
                const uint32_t dst = (event * 17) % resident;
                destinations[row * eventStride + eventOffset + event] = dst;
                sources[row * eventStride + eventOffset + event] =
                    scenario == 2 || event % 7 == 0 ? dst : resident + (event * 37) % resident;
            }
        }
        std::iota(keys.begin(), keys.end(), uint64_t(111));
        std::iota(addresses.begin(), addresses.end(), uint64_t(222));
        std::iota(logical.begin(), logical.end(), uint64_t(333));
        std::iota(scales.begin(), scales.end(), uint32_t(444));
    }
};
void ValidatePlans(const TestData &data) {
    serialValidationMode = true;
    for (uint32_t row = 0; row < data.batch; ++row) {
        uint32_t work[4] = {};
        for (uint32_t lane = 0; lane < 4; ++lane) {
            for (threadIdx.x = 0; threadIdx.x < post_index::kThreads; ++threadIdx.x) {
                post_index::ValidatePlan(const_cast<uint32_t *>(data.sources.data()),
                                         const_cast<uint32_t *>(data.destinations.data()),
                                         row * data.eventStride + data.eventOffset,
                                         data.prefix[row + 1] - data.prefix[row], data.resident,
                                         data.history, lane, 4, work);
            }
            assert(!work[0]);
        }
    }
    serialValidationMode = false;
}
uint32_t ApplyOracle(TestData &data, bool ownerNull, int badRow) {
    uint32_t moved = 0;
    for (uint32_t row = 0; row < data.batch; ++row) {
        if (int(row) == badRow) {
            continue;
        }
        const uint32_t owner = ownerNull ? row : data.owners[row];
        for (uint32_t event = 0; event < data.prefix[row + 1] - data.prefix[row]; ++event) {
            uint32_t source = data.sources[row * data.eventStride + data.eventOffset + event],
                     destination =
                         data.destinations[row * data.eventStride + data.eventOffset + event];
            moved += source != destination;
            for (int word = 0; word < 16; ++word) {
                std::swap(data.keys[(owner * data.history + source) * 16 + word],
                          data.keys[(owner * data.history + destination) * 16 + word]);
            }
            std::swap(data.scales[owner * data.history + source],
                      data.scales[owner * data.history + destination]);
            std::swap(data.addresses[owner * data.addressStride + source],
                      data.addresses[owner * data.addressStride + destination]);
            std::swap(data.logical[owner * data.history + source],
                      data.logical[owner * data.history + destination]);
        }
    }
    return moved;
}
void AssertEqual(const TestData &actual, const TestData &expected) {
    assert(actual.keys == expected.keys && actual.scales == expected.scales);
    assert(actual.addresses == expected.addresses && actual.logical == expected.logical);
    assert(actual.sources == expected.sources && actual.destinations == expected.destinations);
    assert(actual.prefix == expected.prefix && actual.owners == expected.owners &&
           actual.records == expected.records);
}
void RunCase(uint32_t batch, uint32_t cores, int scenario, bool ownerNull, bool profile, int badRow,
             uint32_t errorBit = 37) {
    TestData actual(batch, cores, scenario);
    ValidatePlans(actual);
    if (badRow >= 0) {
        uint32_t peer = badRow + (cores - 1 - badRow) / batch * batch;
        // Preserve nonzero 64-bit error semantics.
        actual.records[peer * 16] = uint64_t(1) << errorBit;
    }
    const TestData original = actual;
    TestData once = actual;
    const uint32_t moved = ApplyOracle(once, ownerNull, badRow);
    TestData twice = once;
    assert(ApplyOracle(twice, ownerNull, badRow) == moved);
    AssertEqual(twice, original);
    const uint32_t width = post_index::kEventThreads;
    CpuWarpGroup group(width);
    uint32_t counter = 0, totalMoved = 0;
    std::vector<std::thread> lanes;
    for (uint32_t lane = 0; lane < width; ++lane) {
        lanes.emplace_back([&, lane] {
            currentWarp = &group;
            for (uint32_t pass = 0; pass < 2; ++pass) {
                if (!lane) {
                    totalMoved = 0;
                }
                group.barrier.arrive_and_wait();
                for (uint32_t core = cores; core--;) {
                    if (!lane) {
                        counter = 0;
                    }
                    group.barrier.arrive_and_wait();
                    const uint32_t first = actual.prefix.back() * core / cores,
                                   end = actual.prefix.back() * (core + 1) / cores;
                    for (uint32_t g = 0; g < post_index::kThreads / width; ++g) {
                        threadIdx.x = g * width + lane;
                        group.barrier.arrive_and_wait();
                        if (profile) {
                            post_index::ApplyPlan<true>(
                                actual.sources.data(), actual.destinations.data(),
                                actual.eventOffset, actual.eventStride, actual.history,
                                actual.prefix.data(), batch, first, end,
                                ownerNull ? nullptr : actual.owners.data(), actual.records.data(),
                                cores, actual.keys.data(), actual.scales.data(),
                                actual.addresses.data(), actual.logical.data(),
                                actual.addressStride, &counter);
                        } else {
                            post_index::ApplyPlan<false>(
                                actual.sources.data(), actual.destinations.data(),
                                actual.eventOffset, actual.eventStride, actual.history,
                                actual.prefix.data(), batch, first, end,
                                ownerNull ? nullptr : actual.owners.data(), actual.records.data(),
                                cores, actual.keys.data(), actual.scales.data(),
                                actual.addresses.data(), actual.logical.data(),
                                actual.addressStride, nullptr);
                        }
                        group.barrier.arrive_and_wait();
                    }
                    if (!lane) {
                        assert(counter <= end - first);
                        totalMoved += counter;
                    }
                    group.barrier.arrive_and_wait();
                }
                if (!lane) {
                    assert(totalMoved == (profile ? moved : 0));
                    AssertEqual(actual, pass ? twice : once);
                }
                group.barrier.arrive_and_wait();
            }
        });
    }
    for (auto &t : lanes) {
        t.join();
    }
}
void CheckInvalidPlans() {
    serialValidationMode = true;
    threadIdx.x = 0;
    uint32_t work[4], src, dst;
    for (auto p : std::array<std::pair<uint32_t, uint32_t>, 3>{{{0, 1}, {1024, 0}, {512, 512}}}) {
        src = p.first;
        dst = p.second;
        post_index::ValidatePlan(&src, &dst, 0, 1, 512, 1024, 0, 1, work);
        assert(work[0]);
    }
    src = 3;
    dst = 3;
    post_index::ValidatePlan(&src, &dst, 0, 1, 512, 1024, 0, 1, work);
    assert(!work[0]);
    post_index::ValidatePlan(&src, &dst, 0, 8193, 512, 1024, 0, 1, work);
    assert(work[0]);
    post_index::ValidatePlan(&src, &dst, 0, 1, 512, 1024, 0, 0, work);
    assert(work[0]);
    post_index::ValidatePlan(&src, &dst, 0, 1, 512, 1024, 1, 1, work);
    assert(work[0]);
    serialValidationMode = false;
}
int main() {
    CheckInvalidPlans();
    for (uint32_t batchSize : {1U, 3U, 16U}) {
        for (uint32_t coreCount : {16U, 64U}) {
            for (bool nullOwners : {false, true}) {
                for (bool profiled : {false, true}) {
                    RunCase(batchSize, coreCount, 1, nullOwners, profiled, -1);
                    RunCase(batchSize, coreCount, 1, nullOwners, profiled,
                            batchSize == 1   ? 0
                            : batchSize == 3 ? 1
                                             : 15);
                }
            }
        }
    }
    for (uint32_t row = 0; row < 16; ++row) {
        RunCase(16, 64, 1, false, true, row);
    }
    for (uint32_t bit : {0U, 31U, 32U, 61U, 63U}) {
        RunCase(1, 64, 1, false, true, 0, bit);
        RunCase(16, 64, 1, true, false, 15, bit);
    }
    for (uint32_t batchSize : {1U, 3U, 16U}) {
        RunCase(batchSize, 64, 0, false, true, -1);
        RunCase(batchSize, 64, 2, true, true, -1);
    }
    for (bool profiled : {false, true}) {
        RunCase(1, 64, 3, false, profiled, -1);
        RunCase(1, 64, 3, true, profiled, 0);
    }
    std::cout << "INDEX_WARP_CONTRACT_PASS threads=" << post_index::kThreads
              << " group=" << post_index::kEventThreads
              << " profile=both owner=null+mapped repeated=2 errors=highest_peer" << std::endl;
}
