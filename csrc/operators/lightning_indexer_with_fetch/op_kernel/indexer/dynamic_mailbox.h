// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "dynamic_schedule.h"

namespace dynamic_li {
struct Mailbox {
    GM_ADDR base;

    __aicore__ inline void Flush(uint32_t line) const {
        kvaf_device::CleanCacheLines(base + line * kLineBytes, kLineBytes);
    }
    __aicore__ inline uint32_t Read(uint32_t line) const {
        Flush(line);
        return *reinterpret_cast<__gm__ volatile uint32_t*>(base + line * kLineBytes);
    }
    __aicore__ inline void Write(uint32_t line, uint32_t value) const {
        *reinterpret_cast<__gm__ volatile uint32_t*>(base + line * kLineBytes) = value;
        Flush(line);
    }
    __aicore__ inline uint32_t Score(uint32_t group) const { return Read(group); }
    __aicore__ inline uint32_t Service(uint32_t lane) const { return Read(kScoreSlots + lane); }
    __aicore__ inline uint32_t AivAck(uint32_t lane) const { return Read(kAckSlots + lane); }
    __aicore__ inline uint32_t AicAck(uint32_t group) const { return Read(kAckSlots + kAiv + group); }
    __aicore__ inline void SendScore(uint32_t group, uint32_t job) const { Write(group, job); }
    __aicore__ inline void SendService(uint32_t lane, uint32_t job) const { Write(kScoreSlots + lane, job); }
    __aicore__ inline void CompleteAiv(uint32_t lane, uint32_t job) const { Write(kAckSlots + lane, job); }
    __aicore__ inline void CompleteAic(uint32_t group, uint32_t job) const { Write(kAckSlots + kAiv + group, job); }
};
}
