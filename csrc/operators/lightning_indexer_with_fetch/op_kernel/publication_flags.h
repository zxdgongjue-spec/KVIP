// SPDX-License-Identifier: Apache-2.0
// Descriptor readiness and service-output completion protect different readers.
#pragma once
#include <stdint.h>

#ifndef KVIP_EARLY_FETCH_READY
#define KVIP_EARLY_FETCH_READY 0
#endif
#if KVIP_EARLY_FETCH_READY != 0 && KVIP_EARLY_FETCH_READY != 1
#error "KVIP_EARLY_FETCH_READY must be 0 or 1"
#endif

namespace pipeline_ab {
#if KVIP_EARLY_FETCH_READY && !V9_UBMEM && KVIP_OUTPUT_WORKER
constexpr bool kEarlyFetchReady = true;
#else
constexpr bool kEarlyFetchReady = false;
#endif
// Slot 5 was an unconsumed producer-completion notification. With early Fetch
// enabled its sole writer becomes OutputWorker. The allocation stays at 16
// cache-line-isolated flags and the existing round initialization clears it.
constexpr uint32_t kOutputsDoneFlag = 5;
constexpr uint32_t kOutputCompletionFlag = kEarlyFetchReady ? kOutputsDoneFlag : 4;
}
