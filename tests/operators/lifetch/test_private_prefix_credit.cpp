// SPDX-License-Identifier: Apache-2.0
#include <cassert>
#include <cstdint>
#include "../../../csrc/operators/lightning_indexer_with_fetch/op_kernel/terminal/private_prefix_credit.h"

struct PrivateQueue {
    uint32_t issued = 1, read = 0, peak = 1;
    uint32_t failure = 99, timeout = 99;
    uint64_t original = UINT64_MAX - 1, producer = original, consumer = original;

    void Issue(uint32_t index) {
        assert(index == issued);
        ++issued;
        ++producer;
        assert(issued - read <= 2);
        if (issued - read > peak) peak = issued - read;
    }
    terminal_credit::Receipt Consume(uint32_t index) {
        assert(index == read && read < issued);
        if (index == timeout) return {13, false};
        ++read;
        ++consumer;
        return {index == failure ? 11U : 0U, true};
    }
};

int main() {
    for (uint32_t count = 2; count <= 4; ++count) {
        PrivateQueue queue;
        const auto success = terminal_credit::CompletePair(queue, count);
        assert(!success.error && success.issued == count && success.consumed == count);
        assert(success.successful == count && queue.peak == 2);
        assert(queue.consumer == queue.original + count);
        for (uint32_t index = 0; index < count; ++index) {
            PrivateQueue faulty;
            faulty.failure = index;
            const auto failed = terminal_credit::CompletePair(faulty, count);
            assert(failed.error == 11 && failed.issued == failed.consumed);
            assert(failed.issued == (index + 2 < count ? index + 2 : count));
            assert(failed.successful == failed.issued - 1);
            PrivateQueue timed;
            timed.timeout = index;
            const auto stalled = terminal_credit::CompletePair(timed, count);
            assert(stalled.error == 13 && stalled.consumed == index && stalled.successful == index);
            assert(stalled.issued == failed.issued);
        }
    }
    PrivateQueue mixed;
    mixed.failure = 0;
    mixed.timeout = 1;
    const auto failure = terminal_credit::CompletePair(mixed, 4);
    assert(failure.error == 11 && failure.issued == 2 && failure.consumed == 1 && !failure.successful);
}
