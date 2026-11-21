// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_functions.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"

namespace kvip_logical_cutoff {
using namespace AscendC;
constexpr uint32_t kThreads = 512;
constexpr uint32_t kTile = 8192;
constexpr uint32_t kTop = 2048;

__simd_vf__ inline void MarkEqual(__ubuf__ uint16_t* scores, __ubuf__ uint32_t* words,
                                  uint16_t count, uint16_t cutoff) {
    using namespace MicroAPI;
    auto all = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg even, odd, lo, hi, valid;
    RegTensor<uint16_t> value;
    RegTensor<int16_t> index;
    for (uint16_t i = 0; i + 256 <= count; i += 256) {
        LoadAlign(value,scores+i);
        CompareScalar<uint16_t, CMPMODE::EQ>(lo,value,cutoff,all);
        LoadAlign(value,scores+i+128);
        CompareScalar<uint16_t, CMPMODE::EQ>(hi,value,cutoff,all);
        MaskDeInterleave<uint8_t>(even,odd,lo,hi);
        StoreAlign(words+i/32,even);
    }
    if (count % 256) {
        const uint16_t i=count/256*256;
        Arange(index, int16_t(i));
        CompareScalar<uint16_t, CMPMODE::LT>(valid, reinterpret_cast<RegTensor<uint16_t>&>(index), count, all);
        LoadAlign(value, scores + i);
        CompareScalar<uint16_t, CMPMODE::EQ>(lo, value, cutoff, all);
        MaskAnd(lo, lo, valid, all);
        Arange(index, int16_t(i+128));
        CompareScalar<uint16_t, CMPMODE::LT>(valid, reinterpret_cast<RegTensor<uint16_t>&>(index), count, all);
        LoadAlign(value, scores + i + 128);
        CompareScalar<uint16_t, CMPMODE::EQ>(hi, value, cutoff, all);
        MaskAnd(hi, hi, valid, all);
        // A B16 predicate uses every second bit. Pack two 128-lane predicates into one 32-byte bitmap.
        MaskDeInterleave<uint8_t>(even, odd, lo, hi);
        StoreAlign(words + i/32, even);
    }
}

// The selected scores and IDs are paired. Preserve every strictly higher score at its original output position.
__simd_vf__ inline void EqualSlots(__ubuf__ uint16_t* scores, __ubuf__ uint16_t* slots,
                                   __ubuf__ uint32_t* count, uint16_t cutoff) {
    using namespace MicroAPI;
    auto all = CreateMask<uint16_t, MaskPattern::ALL>();
    auto first = CreateMask<uint32_t, MaskPattern::VL1>();
    RegTensor<uint16_t> value, packed, ones, sums;
    RegTensor<int16_t> index;
    RegTensor<uint32_t> total;
    MaskReg equal;
    UnalignRegForStore store;
    Duplicate(sums, uint16_t(0));
    ClearSpr<SpecialPurposeReg::AR>();
    for (uint16_t i = 0; i < kTop; i += 128) {
        Arange(index, int16_t(i));
        LoadAlign(value, scores + i);
        CompareScalar<uint16_t, CMPMODE::EQ>(equal, value, cutoff, all);
        Duplicate<uint16_t,MaskMergeMode::ZEROING>(ones, uint16_t(1), equal);
        Add(sums, sums, ones, all);
        Squeeze<uint16_t,MicroAPI::GatherMaskMode::STORE_REG>(packed, reinterpret_cast<RegTensor<uint16_t>&>(index), equal);
        StoreUnAlign<uint16_t,PostLiteral::POST_MODE_UPDATE>(slots, packed, store);
    }
    StoreUnAlignPost(slots, store);
    ReduceSum(total, sums, all);
    StoreAlign(count, total, first);
}

__simt_callee__ inline uint32_t Exclusive(uint32_t n, __ubuf__ uint32_t* warps) {
    uint32_t tid = threadIdx.x, lane = tid % 32, warp = tid / 32, scan = n;
    for (uint32_t d = 1; d < 32; d *= 2) {
        uint32_t left = asc_shfl_up(scan, d);
        if (lane >= d) scan += left;
    }
    if (lane == 31) warps[warp] = scan;
    asc_syncthreads();
    if (!warp) {
        uint32_t v = lane < kThreads/32 ? warps[lane] : 0;
        for (uint32_t d = 1; d < 32; d *= 2) {
            uint32_t left = asc_shfl_up(v, d);
            if (lane >= d) v += left;
        }
        warps[lane] = v;
    }
    asc_syncthreads();
    uint32_t offset = scan - n + (warp ? warps[warp-1] : 0);
    asc_syncthreads();
    return offset;
}

__simt_callee__ inline uint32_t Tagged(uint32_t p, uint32_t resident, __gm__ uint32_t* hot,
                                       __gm__ int32_t* reserved, uint32_t tailBegin, uint32_t tailEnd) {
    if (p >= tailBegin && p < tailEnd) return p | 0x40000000U;
    if (p < resident && (!hot || asc_ldcg(hot+p)) && (!reserved || asc_ldcg(reserved+p) < 0))
        return p | 0x80000000U;
    return p;
}

// A serving binding may use nonlocal logical IDs. Find the cutoff member's 64-bit key by
// MSD radix selection; all passes walk only the set bits, and no candidate list can overflow.
__simt_callee__ inline void WideLogical(__gm__ uint64_t* logical, __ubuf__ uint32_t* marks, uint32_t words,
        __ubuf__ uint16_t* slots, uint32_t quota, __ubuf__ uint32_t* ids, __ubuf__ uint32_t* work,
        uint32_t resident, __gm__ uint32_t* hot, __gm__ int32_t* reserved, uint32_t tailBegin, uint32_t tailEnd) {
    uint32_t tid = threadIdx.x, rank = quota;
    uint64_t prefix = 0, prefixMask = 0;
    for (int shift = 56; shift >= 0; shift -= 8) {
        if (tid < 256) work[tid] = 0;
        asc_syncthreads();
        for (uint32_t w = tid; w < words; w += kThreads) {
            for (uint32_t bits = marks[w]; bits; bits &= bits-1) {
                uint32_t p = 32*w + __builtin_ctz(bits);
                uint64_t key = asc_ldcg(logical+p);
                if ((key & prefixMask) == prefix) asc_atomic_add(work + uint32_t((key>>shift) & 255), 1U);
            }
        }
        asc_syncthreads();
        if (!tid) {
            uint32_t bucket = 0, remaining = rank;
            while (bucket < 255 && remaining > work[bucket]) remaining -= work[bucket++];
            work[256] = bucket;
            work[257] = remaining;
        }
        asc_syncthreads();
        prefix |= uint64_t(work[256]) << shift;
        prefixMask |= uint64_t(255) << shift;
        rank = work[257];
        asc_syncthreads();
    }
    // rank members at the boundary are needed. Order identical keys by physical position.
    uint32_t span = (words + kThreads-1)/kThreads, begin = tid*span, end = begin+span;
    if (end > words) end = words;
    uint32_t lower = 0, equal = 0;
    for (uint32_t w = begin; w < end; ++w) {
        for (uint32_t bits = marks[w]; bits; bits &= bits-1) {
            uint64_t key = asc_ldcg(logical + 32*w + __builtin_ctz(bits));
            lower += key < prefix;
            equal += key == prefix;
        }
    }
    uint32_t lowerPos = Exclusive(lower, work), equalPos = Exclusive(equal, work);
    for (uint32_t w = begin; w < end; ++w) {
        for (uint32_t bits = marks[w]; bits; bits &= bits-1) {
            uint32_t p = 32*w + __builtin_ctz(bits);
            uint64_t key = asc_ldcg(logical+p);
            uint32_t pos = quota;
            if (key < prefix) pos = lowerPos++;
            if (key == prefix) pos = quota-rank+equalPos++;
            if (pos < quota) ids[slots[pos]] = Tagged(p,resident,hot,reserved,tailBegin,tailEnd);
        }
    }
    asc_syncthreads();
}

// A compact list of nonempty bitmap words lets sparse rows read each logical key once.
__simd_vf__ inline void CompactWords(__ubuf__ uint32_t* marks, __ubuf__ uint32_t* output,
                                      __ubuf__ uint32_t* count, uint16_t words) {
    using namespace MicroAPI;
    auto all = CreateMask<uint32_t, MaskPattern::ALL>();
    auto first = CreateMask<uint32_t, MaskPattern::VL1>();
    RegTensor<uint32_t> bits, packed, ones, sums, total;
    RegTensor<int32_t> index;
    MaskReg take;
    UnalignRegForStore store;
    uint32_t left = words;
    Duplicate(sums, 0U);
    ClearSpr<SpecialPurposeReg::AR>();
    for (uint16_t w = 0; w < words; w += 64) {
        MaskReg valid = UpdateMask<uint32_t>(left);
        LoadAlign(bits, marks + w);
        CompareScalar<uint32_t, CMPMODE::NE>(take, bits, 0U, valid);
        Arange(index, int32_t(w));
        Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(
            packed, reinterpret_cast<RegTensor<uint32_t>&>(index), take);
        StoreUnAlign<uint32_t, PostLiteral::POST_MODE_UPDATE>(output, packed, store);
        Duplicate<uint32_t, MaskMergeMode::ZEROING>(ones, 1U, take);
        Add(sums, sums, ones, all);
    }
    StoreUnAlignPost(output, store);
    ReduceSum(total, sums, all);
    StoreAlign(count, total, first);
}

__simt_callee__ inline bool SparseLogical(__gm__ uint64_t* logical, __ubuf__ uint32_t* marks,
        uint32_t history, __ubuf__ uint16_t* slots, uint32_t quota, __ubuf__ uint32_t* ids,
        __ubuf__ uint32_t* work, uint32_t nonempty, uint32_t resident, __gm__ uint32_t* hot,
        __gm__ int32_t* reserved, uint32_t tailBegin, uint32_t tailEnd) {
    if (nonempty > kThreads) return false;
    const uint32_t tid = threadIdx.x;
    uint32_t w = tid < nonempty ? work[tid] : 0;
    uint32_t bits = tid < nonempty ? marks[w] : 0;
    const uint32_t logicalWords = (history+31)/32;
    auto prefix = work + logicalWords;
    auto ctl = prefix + kThreads;
    // The fallback owns ctl[32] and can start on an earlier warp.
    auto sparseFallback = ctl + 33;
    uint32_t slot = Exclusive(__popc(bits), ctl);
    uint32_t size = ctl[kThreads/32-1];
    if (size == quota) return true;
    if (size > kThreads) return false;

    for (; bits; bits &= bits-1) prefix[slot++] = 32*w + __builtin_ctz(bits);
    asc_syncthreads();
    uint32_t physical = tid < size ? prefix[tid] : 0;
    uint64_t key = tid < size ? asc_ldcg(logical+physical) : 0;
    for (uint32_t i=tid; i<logicalWords; i+=kThreads) work[i] = 0;
    if (!tid) *sparseFallback = 0;
    asc_syncthreads();
    if (tid < size) {
        if (key < history) {
            uint32_t bit = 1U << (uint32_t(key) % 32);
            if (asc_atomic_or(work + uint32_t(key)/32,bit) & bit) asc_atomic_or(sparseFallback,1U);
        } else asc_atomic_or(sparseFallback,1U);
    }
    asc_syncthreads();
    if (*sparseFallback) return false;
    const uint32_t span = (logicalWords+kThreads-1)/kThreads;
    uint32_t n = 0;
    for (uint32_t i=tid*span; i<(tid+1)*span && i<logicalWords; ++i) n+=__popc(work[i]);
    prefix[tid] = Exclusive(n,ctl);
    asc_syncthreads();
    if (tid < size) {
        const uint32_t word = uint32_t(key)/32, group = word/span;
        uint32_t rank = prefix[group];
        if (rank < quota) {
            for (uint32_t i=group*span; i<word; ++i) rank+=__popc(work[i]);
            rank+=__popc(work[word] & ((1U<<(uint32_t(key)%32))-1));
            if (rank < quota) ids[slots[rank]] = Tagged(physical,resident,hot,reserved,tailBegin,tailEnd);
        }
    }
    asc_syncthreads();
    return true;
}
__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads) inline void Select(
        __gm__ uint64_t* logical, __ubuf__ uint32_t* marks, uint32_t words, uint32_t history,
        __ubuf__ uint16_t* slots, __ubuf__ uint32_t* count, __ubuf__ uint32_t* ids, __ubuf__ uint32_t* work,
        uint32_t resident, __gm__ uint32_t* hot, __gm__ int32_t* reserved, uint32_t tailBegin, uint32_t tailEnd) {
    uint32_t tid = threadIdx.x, quota = count[0];
    if (!quota) return;
    if (words > kTile/32 &&
        SparseLogical(logical,marks,history,slots,quota,ids,work,count[8],resident,hot,reserved,tailBegin,tailEnd)) return;
    const uint32_t logicalWords = (history+31)/32;
    auto prefix = work + logicalWords;
    auto ctl = prefix + kThreads;
    for (uint32_t w = tid; w < logicalWords; w += kThreads) work[w] = 0;
    if (!tid) ctl[32] = 0;
    asc_syncthreads();
    for (uint32_t w = tid; w < words; w += kThreads) {
        for (uint32_t bits = marks[w]; bits; bits &= bits-1) {
            uint32_t p = 32*w + __builtin_ctz(bits);
            uint64_t key = asc_ldcg(logical+p);
            if (key < history) {
                uint32_t bit = 1U << (uint32_t(key)%32);
                if (asc_atomic_or(work + uint32_t(key)/32, bit) & bit) asc_atomic_or(ctl+32,1U);
            } else asc_atomic_or(ctl+32,1U);
        }
    }
    asc_syncthreads();
    if (ctl[32]) {
        WideLogical(logical,marks,words,slots,quota,ids,work,resident,hot,reserved,tailBegin,tailEnd);
        return;
    }
    const uint32_t span = (logicalWords + kThreads-1)/kThreads;
    uint32_t n = 0;
    for (uint32_t w = tid*span; w < (tid+1)*span && w < logicalWords; ++w) n += __popc(work[w]);
    prefix[tid] = Exclusive(n,ctl);
    asc_syncthreads();
    if (ctl[kThreads/32-1] == quota) return;
    for (uint32_t w = tid; w < words; w += kThreads) {
        for (uint32_t bits = marks[w]; bits; bits &= bits-1) {
            uint32_t p = 32*w + __builtin_ctz(bits), key = uint32_t(asc_ldcg(logical+p));
            uint32_t word = key/32, group = word/span, rank = prefix[group];
            if (rank >= quota) continue;
            for (uint32_t v = group*span; v < word; ++v) rank += __popc(work[v]);
            rank += __popc(work[word] & ((1U<<(key%32))-1));
            if (rank < quota) ids[slots[rank]] = Tagged(p,resident,hot,reserved,tailBegin,tailEnd);
        }
    }
    asc_syncthreads();
}

__aicore__ inline void Repair(GlobalTensor<uint16_t> raw, GlobalTensor<uint64_t> logical,
        LocalTensor<uint16_t> chosenScores, LocalTensor<uint32_t> taggedIds,
        LocalTensor<uint16_t> tile, LocalTensor<uint32_t> scratch, uint32_t valid, uint32_t history,
        uint16_t cutoff, uint32_t resident, __gm__ uint32_t* hot, __gm__ int32_t* reserved,
        uint32_t tailBegin, uint32_t tailEnd) {
    // Maximum supported history is 210944. Physical bitmap, slots and count use <=30528 B.
    // After score DMA the two 16 KiB tile banks become the logical bitmap and prefix counters.
    const uint32_t words = (valid+31)/32, alignedWords = ((words+31)/32)*32;
    auto slots = scratch[alignedWords].template ReinterpretCast<uint16_t>();
    auto count = scratch[alignedWords+1024];
    EqualSlots(reinterpret_cast<__ubuf__ uint16_t*>(chosenScores.GetPhyAddr()),
               reinterpret_cast<__ubuf__ uint16_t*>(slots.GetPhyAddr()),
               reinterpret_cast<__ubuf__ uint32_t*>(count.GetPhyAddr()),cutoff);
    for (uint32_t base = 0; base < valid; base += kTile) {
        uint32_t n = valid-base < kTile ? valid-base : kTile;
        uint32_t bank = (base/kTile)%2, event = bank ? EVENT_ID3 : EVENT_ID5;
        if (base >= 2*kTile) WaitFlag<HardEvent::V_MTE2>(event);
        auto buf = tile[bank*kTile];
        DataCopyExtParams copy{1, n*uint32_t(sizeof(uint16_t)),0,0,0};
        DataCopyPadExtParams<uint16_t> pad{true,0,0,0};
        DataCopyPad(buf,raw[base],copy,pad);
        SetFlag<HardEvent::MTE2_V>(event);WaitFlag<HardEvent::MTE2_V>(event);
        MarkEqual(reinterpret_cast<__ubuf__ uint16_t*>(buf.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ uint32_t*>(scratch[base/32].GetPhyAddr()),uint16_t(n),cutoff);
        SetFlag<HardEvent::V_MTE2>(event);
    }
    if (valid) WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
    if (valid > kTile) WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
    PipeBarrier<PIPE_ALL>();
    // A single score tile already fits the direct bitmap traversal. Compacting
    // its word indices costs another vector pass and synchronization.
    if (words > kTile/32) {
        CompactWords(reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr()),
                     reinterpret_cast<__ubuf__ uint32_t*>(tile.GetPhyAddr()),
                     reinterpret_cast<__ubuf__ uint32_t*>(count[8].GetPhyAddr()), uint16_t(words));
        PipeBarrier<PIPE_ALL>();
    }
    Simt::VF_CALL<Select>(Simt::Dim3{kThreads,1,1},const_cast<__gm__ uint64_t*>(logical.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr()),words,history,
        reinterpret_cast<__ubuf__ uint16_t*>(slots.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(count.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(taggedIds.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(tile.GetPhyAddr()),resident,hot,reserved,tailBegin,tailEnd);
    PipeBarrier<PIPE_ALL>();
}
} // namespace kvip_logical_cutoff
