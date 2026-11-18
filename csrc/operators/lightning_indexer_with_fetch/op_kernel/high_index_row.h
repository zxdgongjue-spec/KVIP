// SPDX-License-Identifier: Apache-2.0
#pragma once

// Included inside high_post_index after TransferSegments. All lanes see the
// same immutable row plan. Each owns a separate slice of swaps, cycle edges,
// and invalidations. Distinct active rows must have distinct physical owners.
template<class RowJoin>
__aicore__ inline uint64_t TransferRow(__gm__ Args* a, const Segment& part,
        uint64_t stride, bool spill, __ubuf__ Tuple* local, bool sharedCycles,
        __gm__ uint64_t* ticks, bool profile, RowJoin& join) {
    if (part.p0 != part.p1 || part.s0 != part.s1)
        TransferSegments<false>(a,&part,1,a->history,stride,spill,local);
    // The swaps are disjoint. Long cycles can cross lanes: every original
    // must be in private UB or private spill before the first overwrite.
    if (sharedCycles) {
        uint64_t error=join.Gathered();
        if (error) return error;
    }
    if (profile) ticks[12]=GetSystemCycle();
    if (part.s0 != part.s1)
        TransferSegments<true>(a,&part,1,a->history,stride,spill,local);
    return 0;
}

template<class RowJoin>
__aicore__ inline uint64_t ApplyRowIndex(__gm__ Args* a, __gm__ Context* ctx,
        uint32_t b, uint32_t owner, uint32_t lane, uint32_t lanes,
        uint32_t pairs, uint32_t singles, uint32_t invalidations, uint64_t addressStride,
        __gm__ uint64_t* ticks, bool profile, RowJoin& join) {
    // These bounds are properties of the plan and are uniform for its lanes.
    if (pairs>kCapacity/2 || singles>kCapacity-2*pairs || invalidations>kCapacity)
        return kvaf_device::kInvalidArgs;
    Segment part{b,owner,pairs*lane/lanes,pairs*(lane+1)/lanes,
                 singles*lane/lanes,singles*(lane+1)/lanes,0};
    const uint32_t moves=2*pairs+singles;
    // Swaps and longer cycles are disjoint in the composed permutation.
    // When all long edges fit one lane's usual share and its UB, keep
    // their originals together so the other three lanes need no gather join.
    // Charge the owning lane for those moves when distributing pair swaps.
    const bool privateCycles=singles && singles<=kLocalTuples && singles*lanes<=moves;
    if (privateCycles) {
        const uint32_t first=moves*lane/(2*lanes);
        const uint32_t end=moves*(lane+1)/(2*lanes);
        part.p0=first<pairs ? first : pairs;
        part.p1=end<pairs ? end : pairs;
        part.s0=0;
        part.s1=lane+1==lanes ? singles : 0;
    }
    const bool sharedCycles=singles && !privateCycles;
    uint32_t i0=invalidations*lane/lanes, i1=invalidations*(lane+1)/lanes;
    if (i0!=i1) {
        auto edges=reinterpret_cast<__gm__ Edge*>(a->highIndex+uint64_t(b)*kRowBytes);
        auto payload=reinterpret_cast<__gm__ Tuple*>(edges+kCapacity);
        auto valid=reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+uint64_t(owner)*a->resident;
        Simt::VF_CALL<CommitInvalidations>(Simt::Dim3{256,1,1},&payload[0].padding,i0,i1,valid);
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
    uint32_t localCount=part.s1-part.s0;
    bool spill=localCount>kLocalTuples;
    uint64_t error;
    if (!localCount || spill) {
        // The spill specialization never forms an address from a null UB
        // pointer. Empty long lists do not allocate a dummy payload.
        error=TransferRow(a,part,addressStride,true,nullptr,sharedCycles,ticks,profile,join);
    } else {
        TPipe pipe;
        TBuf<TPosition::VECCALC> payload;
        pipe.InitBuffer(payload,((localCount*sizeof(Tuple)+31)/32)*32);
        auto local=reinterpret_cast<__ubuf__ Tuple*>(payload.Get<uint32_t>().GetPhyAddr());
        error=TransferRow(a,part,addressStride,false,local,sharedCycles,ticks,profile,join);
        pipe.Destroy();
    }
    if (profile) {
        uint32_t pairCount=part.p1-part.p0;
        ticks[2]=0x50494831U; ticks[5]=GetSystemCycle();
        ticks[9]=2*pairCount+localCount; ticks[10]=ticks[9];
        ticks[16]=pairCount; ticks[17]=localCount;
        ticks[18]=spill ? localCount : 0; ticks[19]=i1-i0;
    }
    return error;
}
