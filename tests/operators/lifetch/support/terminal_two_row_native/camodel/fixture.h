// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
namespace tiny {
constexpr uint32_t first=7,second=29;
constexpr uint64_t head=(1ULL<<32)-3,cursor=7;
struct alignas(128) Trace {
    uint64_t prepare,refused,initialSq,atPark,parentThird,firstChild,mask;
    uint64_t issued,seen,success,error,earlyQ2,retired,q2Start,q2End,pending;
    uint64_t returnTick,forcedFourth,expectedArrival,ordinaryStart;
};
}
