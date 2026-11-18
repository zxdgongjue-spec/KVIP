// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace pipeline_ab {
namespace high_post_index {
struct Segment {
    uint32_t row, owner, p0, p1, s0, s1, localOffset;
};

__aicore__ inline uint32_t ClipOffset(uint32_t end, uint32_t base, uint32_t size) {
    return end <= base ? 0 : end - base < size ? end - base : size;
}

__aicore__ inline bool PartitionRow(uint32_t start, uint32_t end, uint32_t base,
                                   uint32_t count, uint32_t pairs, Segment& segment) {
    uint32_t left=ClipOffset(start,base,count), right=ClipOffset(end,base,count);
    uint32_t pairWeight=2*pairs, singles=count-pairWeight;
    // A two-tuple exchange belongs entirely to the worker owning its first tuple.
    segment.p0=(ClipOffset(left,0,pairWeight)+1)/2;
    segment.p1=(ClipOffset(right,0,pairWeight)+1)/2;
    segment.s0=ClipOffset(left,pairWeight,singles);
    segment.s1=ClipOffset(right,pairWeight,singles);
    return segment.p1>segment.p0 || segment.s1>segment.s0;
}

} // namespace high_post_index
} // namespace pipeline_ab
