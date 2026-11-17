// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
#ifndef KVIP_INCREMENTAL_TOPK
#define KVIP_INCREMENTAL_TOPK 0
#endif
#if KVIP_INCREMENTAL_TOPK && (KVIP_ROLLING_THRESHOLD || KVIP_WINDOW_REFILTER)
#error "Incremental selection replaces, rather than layers on, rolling/refilter"
#endif
namespace kvip_incremental {
constexpr uint32_t kTop = 2048;
constexpr uint32_t kNeedsFetch = 0x80000000U;
constexpr uint32_t kIdMask = 0x7fffffffU;
constexpr uint32_t kChunk = 2048;
constexpr uint32_t kReadyWord = 8;
constexpr uint32_t kHeader = 32;
constexpr uint32_t kListWords = kHeader + kTop;
// Persistent UB: K scores + chunk scores; K u32 IDs; chunk u16 scan offsets.
// Offsets are relative to the current window (<=65536 scan positions).
constexpr uint32_t kUbBytes = (kTop + kChunk) * 2 + kTop * 4 + kChunk * 2;
}
