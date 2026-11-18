// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
#ifndef KVIP_INCREMENTAL_TOPK
#define KVIP_INCREMENTAL_TOPK 0
#endif
#ifndef KVIP_INCREMENTAL_WINDOW
#define KVIP_INCREMENTAL_WINDOW 0
#endif
#ifndef KVIP_INCREMENTAL_REGISTER
#define KVIP_INCREMENTAL_REGISTER 0
#endif
#ifndef KVIP_INCREMENTAL_PREFILTER
#define KVIP_INCREMENTAL_PREFILTER 0
#endif
#ifndef KVIP_INCREMENTAL_DIRECT
#define KVIP_INCREMENTAL_DIRECT 0
#endif
#ifndef KVIP_INCREMENTAL_TIES
#define KVIP_INCREMENTAL_TIES 0
#endif
#ifndef KVIP_INCREMENTAL_LIGHT
#define KVIP_INCREMENTAL_LIGHT 0
#endif
#ifndef KVIP_INCREMENTAL_INPLACE
#define KVIP_INCREMENTAL_INPLACE 0
#endif
#ifndef KVIP_INCREMENTAL_PROGRESS
#define KVIP_INCREMENTAL_PROGRESS 0
#endif
#ifndef KVIP_INCREMENTAL_SEGMENTS
#define KVIP_INCREMENTAL_SEGMENTS 0
#endif
#ifndef KVIP_WINDOW_PROBE
#define KVIP_WINDOW_PROBE 0
#endif
#ifndef KVIP_SEED_MASK
#define KVIP_SEED_MASK 0
#endif
#ifndef KVIP_SEED_PARALLEL
#define KVIP_SEED_PARALLEL 0
#endif
#if KVIP_SEED_PARALLEL && !KVIP_SEED_MASK
#error "Independent seed preparation requires the seed mask"
#endif
#if KVIP_SEED_MASK && !KVIP_INCREMENTAL_SEGMENTS
#error "Seed mask requires segment lists"
#endif
#ifndef KVIP_CONSUMED_PREFIX
#define KVIP_CONSUMED_PREFIX 0
#endif
#if KVIP_CONSUMED_PREFIX && !KVIP_INCREMENTAL_SEGMENTS
#error "Consumed prefix requires immutable segment lists"
#endif
#if KVIP_INCREMENTAL_SEGMENTS && (!KVIP_INCREMENTAL_LIGHT || !KVIP_INCREMENTAL_TIES || KVIP_INCREMENTAL_PROGRESS || KVIP_INCREMENTAL_INPLACE)
#error "Segment lists require light tie selection and its private staging bank"
#endif
#if KVIP_INCREMENTAL_PROGRESS && !KVIP_INCREMENTAL_WINDOW
#error "Independent progress bank requires fixed-window incremental staging"
#endif
#if KVIP_INCREMENTAL_INPLACE && (!KVIP_INCREMENTAL_LIGHT || !KVIP_INCREMENTAL_TIES)
#error "In-place boundary updates require light state and exact tie updates"
#endif
#if KVIP_INCREMENTAL_LIGHT && !KVIP_INCREMENTAL_DIRECT
#error "Light incremental state requires direct selection"
#endif
#if KVIP_INCREMENTAL_DIRECT && !KVIP_INCREMENTAL_PREFILTER
#error "Direct incremental state requires window prefilter"
#endif
#if KVIP_INCREMENTAL_TIES && !KVIP_INCREMENTAL_DIRECT
#error "Boundary tie updates require direct incremental state"
#endif
#if KVIP_INCREMENTAL_PREFILTER && !KVIP_INCREMENTAL_REGISTER
#error "Window prefilter requires register window selection"
#endif
#if KVIP_INCREMENTAL_WINDOW && !KVIP_INCREMENTAL_TOPK
#error "Window staging requires incremental TopK"
#endif
#if KVIP_INCREMENTAL_REGISTER && !KVIP_INCREMENTAL_WINDOW
#error "Register selection requires incremental window staging"
#endif
#if KVIP_INCREMENTAL_TOPK && (KVIP_ROLLING_THRESHOLD || KVIP_WINDOW_REFILTER)
#error "Incremental selection replaces, rather than layers on, rolling/refilter"
#endif
namespace kvip_incremental {
constexpr uint32_t kTop = 2048;
// Ordinals occupy [11520,19712) for K+up to K compact scores. This
// scratch tail also fits the smallest (6K-trunk) selector allocation.
constexpr uint32_t kMappedWord = 4928;
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
