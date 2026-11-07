#pragma once
// SPDX-License-Identifier: Apache-2.0
// Shared unchanged LIDU bitmap count/enumeration; caller owns protection and prefix scan.
namespace kvip_hot_slots {
using namespace AscendC;
__simt_callee__ inline void
CountAvailableBitmapWordsBody(
    __ubuf__ const uint32_t *protectedSlots,
    __ubuf__ int32_t *wordCounts,
    uint32_t budget,
    uint32_t scanBegin,
    uint32_t firstWordCount,
    uint32_t secondWordCount)
{
    const uint32_t threadIndex = Simt::GetThreadIdx();
    const uint32_t threadCount = Simt::GetThreadNum();
    const uint32_t orderedWordCount = firstWordCount + secondWordCount;
    const uint32_t firstWord = scanBegin / 32U;
    for (uint32_t orderedWord = threadIndex;
         orderedWord < orderedWordCount;
         orderedWord += threadCount) {
        uint32_t wordIndex = 0;
        uint32_t segmentBegin = 0;
        uint32_t segmentEnd = scanBegin;
        if (orderedWord < firstWordCount) {
            wordIndex = firstWord + orderedWord;
            segmentBegin = scanBegin;
            segmentEnd = budget;
        } else {
            wordIndex = orderedWord - firstWordCount;
        }

        const uint32_t wordBase = wordIndex * 32U;
        uint32_t validMask = 0xFFFFFFFFU;
        if (segmentBegin > wordBase) {
            validMask &= 0xFFFFFFFFU << (segmentBegin - wordBase);
        }
        if (segmentEnd < wordBase + 32U) {
            const uint32_t validBits = segmentEnd - wordBase;
            validMask &= (1U << validBits) - 1U;
        }

        uint32_t available =
            ~protectedSlots[wordIndex] & validMask;
        uint32_t count = 0;
        while (available != 0U) {
            available &= available - 1U;
            ++count;
        }
        wordCounts[orderedWord] = static_cast<int32_t>(count);
    }
}

__simt_callee__ inline void
WriteAvailableBitmapSlotsBody(
    __ubuf__ const uint32_t *protectedSlots,
    __ubuf__ const int32_t *wordCounts,
    __ubuf__ const int32_t *wordOffsets,
    __ubuf__ int32_t *resolvedSlots,
    uint32_t budget,
    uint32_t scanBegin,
    uint32_t firstWordCount,
    uint32_t secondWordCount,
    uint32_t extraCount,
    uint32_t outputBase = 2048)
{
    const uint32_t threadIndex = Simt::GetThreadIdx();
    const uint32_t threadCount = Simt::GetThreadNum();
    const uint32_t orderedWordCount = firstWordCount + secondWordCount;
    const uint32_t firstWord = scanBegin / 32U;
    for (uint32_t orderedWord = threadIndex;
         orderedWord < orderedWordCount;
         orderedWord += threadCount) {
        const uint32_t prefix = static_cast<uint32_t>(
            wordOffsets[orderedWord]);
        const uint32_t count = static_cast<uint32_t>(
            wordCounts[orderedWord]);
        if (prefix >= extraCount || count == 0) {
            continue;
        }

        uint32_t wordIndex = 0;
        uint32_t segmentBegin = 0;
        uint32_t segmentEnd = scanBegin;
        if (orderedWord < firstWordCount) {
            wordIndex = firstWord + orderedWord;
            segmentBegin = scanBegin;
            segmentEnd = budget;
        } else {
            wordIndex = orderedWord - firstWordCount;
        }

        const uint32_t wordBase = wordIndex * 32U;
        uint32_t validMask = 0xFFFFFFFFU;
        if (segmentBegin > wordBase) {
            validMask &= 0xFFFFFFFFU << (segmentBegin - wordBase);
        }
        if (segmentEnd < wordBase + 32U) {
            const uint32_t validBits = segmentEnd - wordBase;
            validMask &= (1U << validBits) - 1U;
        }

        uint32_t available =
            ~protectedSlots[wordIndex] & validMask;
        uint32_t outputIndex = prefix;
        while (available != 0U && outputIndex < extraCount) {
            // Keep the original bitmap while locating the first set bit.
            // Clearing a bit from a progressively shifted value would leave
            // bits at positions >= 16 uncleared and repeat the same slot.
            uint32_t probe = available;
            uint32_t bit = 0;
            if ((probe & 0x0000FFFFU) == 0) {
                probe >>= 16;
                bit += 16;
            }
            if ((probe & 0x000000FFU) == 0) {
                probe >>= 8;
                bit += 8;
            }
            if ((probe & 0x0000000FU) == 0) {
                probe >>= 4;
                bit += 4;
            }
            if ((probe & 0x00000003U) == 0) {
                probe >>= 2;
                bit += 2;
            }
            if ((probe & 0x00000001U) == 0) {
                bit += 1;
            }
            resolvedSlots[outputBase + outputIndex] =
                static_cast<int32_t>(wordBase + bit);
            ++outputIndex;
            available &= ~(1U << bit);
        }
    }
}


__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void CountAvailableBitmapWordsSimt(
    __ubuf__ const uint32_t* bits,__ubuf__ int32_t* counts,uint32_t budget,
    uint32_t begin,uint32_t first,uint32_t second) {
  CountAvailableBitmapWordsBody(bits,counts,budget,begin,first,second);
}
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void WriteAvailableBitmapSlotsSimt(
    __ubuf__ const uint32_t* bits,__ubuf__ const int32_t* counts,__ubuf__ const int32_t* offsets,
    __ubuf__ int32_t* slots,uint32_t budget,uint32_t begin,uint32_t first,uint32_t second,
    uint32_t count) {
  WriteAvailableBitmapSlotsBody(bits,counts,offsets,slots,budget,begin,first,second,count,2048);
}
}  // namespace kvip_hot_slots
