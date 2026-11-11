// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

namespace kvip_score_trace {
constexpr uint32_t kTrunk = 16384, kRecordWords = 4;
#if defined(__CCE_AICORE__)
#define KVIP_TRACE_FN __aicore__
#else
#define KVIP_TRACE_FN
#define __gm__
#endif
KVIP_TRACE_FN constexpr uint32_t Capacity(uint32_t history) {
    return (history + kTrunk - 1) / kTrunk + 2;
}
KVIP_TRACE_FN constexpr uint32_t QueryWords(uint32_t history) {
    return (Capacity(history) * kRecordWords + 31) / 32 * 32;
}
// One query belongs to one AIV. Its entire record row owns 128B cache lines;
// cleaning a record must never write back a sibling query's scalar stores.
// Slot zero is the resident threshold; later slots correspond to completed
// 16K trunks. The last slot is reserved for a host-derived final threshold.
KVIP_TRACE_FN inline __gm__ uint32_t* Store(__gm__ uint32_t* output,
    uint32_t queries,uint32_t history,uint32_t query,uint32_t validEnd,
    uint32_t appliesFrom,uint32_t key,uint32_t kind) {
    if(!output || query>=queries || validEnd>history || appliesFrom>history ||
       key>65535 || kind<1 || kind>3)return nullptr;
    uint32_t slot=kind==3?(appliesFrom+kTrunk-1)/kTrunk-1:0;
    if(slot>=Capacity(history)-1)return nullptr;
    auto record=output+query*QueryWords(history)+slot*kRecordWords;
    record[0]=validEnd;record[1]=appliesFrom;record[2]=key;record[3]=kind;
    return output+query*QueryWords(history)+(slot*kRecordWords/32)*32;
}
#undef KVIP_TRACE_FN
#if !defined(__CCE_AICORE__)
#undef __gm__
#endif
} // namespace kvip_score_trace
